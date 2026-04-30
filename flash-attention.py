from typing import Dict
import kdt
import torch

def calculate_num_jobs(task_args: dict[str, int]) -> int:
    BLOCK_SIZE_QO = 128
    return task_args["S_qo"] // BLOCK_SIZE_QO

@kdt.kernel(num_jobs_calculator=calculate_num_jobs)
def flash_attention_kernel(task_args: Dict[str, int], io_tensors: Dict[str, kdt.Tile]):
    e = 2.7182818284590451
    inf = -3.4e38
    BLOCK_SIZE_QO = 128
    BLOCK_SIZE_KV = 128
    
    NUM_STAGES = 2

    S_qo = task_args['S_qo']
    S_kv = task_args['S_kv']
    D = task_args['D']
    num_qo_blocks = S_qo // BLOCK_SIZE_QO
    num_kv_blocks = S_kv // BLOCK_SIZE_KV

    job_id = kdt.get_job_id()
    Q_start = job_id * BLOCK_SIZE_QO
    Q_end = Q_start + BLOCK_SIZE_QO

    Q_tile = kdt.alloc_spm((BLOCK_SIZE_QO, D), dtype='float32')
    K_tiles = kdt.alloc_spm((NUM_STAGES, BLOCK_SIZE_KV, D), dtype='float32')
    V_tiles = kdt.alloc_spm((NUM_STAGES, BLOCK_SIZE_KV, D), dtype='float32')
    QK_tile = kdt.alloc_spm((BLOCK_SIZE_QO, BLOCK_SIZE_KV), dtype='float32')
    O_tile = kdt.alloc_spm((BLOCK_SIZE_QO, D), dtype='float32', init_value = 0)
    
    global_rowmax = kdt.alloc_spm((BLOCK_SIZE_QO,), dtype='float32', init_value=inf)
    global_exp_rowsum = kdt.alloc_spm((BLOCK_SIZE_QO,), dtype='float32', init_value=0)

    local_rowmax = kdt.alloc_spm((BLOCK_SIZE_QO,), dtype='float32')
    local_exp_rowsum = kdt.alloc_spm((BLOCK_SIZE_QO,), dtype='float32')
    
    local_rowmax_diff = kdt.alloc_spm((BLOCK_SIZE_QO,), dtype='float32')
    global_rowmax_diff = kdt.alloc_spm((BLOCK_SIZE_QO,), dtype='float32')
    
    global_rowmax_new = kdt.alloc_spm((BLOCK_SIZE_QO,), dtype='float32')
    global_exp_rowsum_new = kdt.alloc_spm((BLOCK_SIZE_QO,), dtype='float32')

    tmp1 = kdt.alloc_spm((BLOCK_SIZE_QO,), dtype='float32')

    Q_global = io_tensors['Q']
    K_global = io_tensors['K']
    V_global = io_tensors['V']
    O_global = io_tensors['O']
    
    kdt.load(Q_global[Q_start:Q_end, :], Q_tile)

    for kv_id in range(0, NUM_STAGES):
        if kv_id < num_kv_blocks:
            stage = kv_id % NUM_STAGES
            KV_start = kv_id * BLOCK_SIZE_KV
            KV_end = KV_start + BLOCK_SIZE_KV
            kdt.load(K_global[KV_start:KV_end, :], K_tiles[stage, :, :])
            kdt.load(V_global[KV_start:KV_end, :], V_tiles[stage, :, :])
        if kv_id == 0:
            kdt.matmul(Q_tile, kdt.transpose(K_tiles[0], 0, 1), QK_tile, accumulate=False)

    for kv_id in range(0, num_kv_blocks):
        stage = kv_id % NUM_STAGES
        K_tile = K_tiles[stage]
        V_tile = V_tiles[stage]
        
        prefetch_kv_id = kv_id + (NUM_STAGES - 1)
        prefetch_stage = prefetch_kv_id % NUM_STAGES
        if kv_id != 0 and prefetch_kv_id < num_kv_blocks:
            KV_start = prefetch_kv_id * BLOCK_SIZE_KV
            KV_end = KV_start + BLOCK_SIZE_KV
            kdt.load(K_global[KV_start:KV_end, :], K_tiles[prefetch_stage])
            kdt.load(V_global[KV_start:KV_end, :], V_tiles[prefetch_stage])
        
        kdt.reduce(QK_tile, 1, 'max', local_rowmax)
        
        kdt.sub(QK_tile, kdt.broadcast_to(kdt.unsqueeze(local_rowmax, 1), 1, BLOCK_SIZE_KV), QK_tile)
        QK_tile_sub = QK_tile
        kdt.exp(QK_tile_sub, QK_tile_sub, e)
        QK_tile_sub_exp = QK_tile_sub
        
        kdt.matmul(QK_tile_sub_exp, V_tile, V_tile)
        PV_tile = V_tile
        
        # PV_tile = matmul(..., ...) overlap w/ VXM ops under
        kdt.reduce(QK_tile_sub_exp, 1, 'sum', local_exp_rowsum)

        kdt.max(local_rowmax, global_rowmax, global_rowmax_new)

        kdt.sub(global_rowmax, global_rowmax_new, global_rowmax_diff)
        kdt.sub(local_rowmax, global_rowmax_new, local_rowmax_diff)
        kdt.exp(local_rowmax_diff, local_rowmax_diff, e)
        kdt.exp(global_rowmax_diff, global_rowmax_diff, e)
        global_rowmax_diff_exp = global_rowmax_diff
        local_rowmax_diff_exp = local_rowmax_diff
        kdt.mul(global_rowmax_diff_exp, global_exp_rowsum, tmp1)
        kdt.fma(local_rowmax_diff_exp, local_exp_rowsum, tmp1, global_exp_rowsum_new)
        kdt.mul(O_tile, kdt.broadcast_to(kdt.unsqueeze(tmp1, 1), 1, D), O_tile)
        # PV_tile = matmul(..., ...) will overlap w/ VXM ops above
        
        next_kv_id = kv_id + 1
        if next_kv_id < num_kv_blocks:
            next_stage = next_kv_id % NUM_STAGES
            # type: MXM
            kdt.matmul(Q_tile, kdt.transpose(K_tiles[next_stage], 0, 1), QK_tile, accumulate=False)
            # Q @ K^T (MXM) of next stage overlap w/ below mul & add & div VXM operations

        # type: VXM
        kdt.mul(PV_tile, kdt.broadcast_to(kdt.unsqueeze(local_rowmax_diff_exp, 1), 1, D), PV_tile)
        kdt.add(O_tile, PV_tile, O_tile)
        kdt.div(O_tile, kdt.broadcast_to(kdt.unsqueeze(global_exp_rowsum_new, 1), 1, D), O_tile)
        kdt.copy(global_rowmax_new, global_rowmax)
        kdt.copy(global_exp_rowsum_new, global_exp_rowsum)


    kdt.store(O_tile, O_global[Q_start:Q_end, :])




def get_kernel(task_id: int) -> kdt.KernelFunction:
    if task_id == 4:
        return flash_attention_kernel
    return None
