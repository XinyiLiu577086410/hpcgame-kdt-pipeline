from itertools import accumulate
from typing import Dict
import kdt
import torch

def calculate_num_jobs(task_args: dict[str, int]) -> int:
    BLK_QO = 128
    return task_args["S_qo"] // BLK_QO

@kdt.kernel(num_jobs_calculator=calculate_num_jobs)
def flash_attention_kernel(task_args: Dict[str, int], io_tensors: Dict[str, kdt.Tile]):
    e = 2.7182818284590451
    inf = -3.4e38
    BLK_QO = 128
    BLK_KV = 128
    
    S_qo = task_args['S_qo']
    S_kv = task_args['S_kv']
    D = task_args['D']
    num_qo_blocks = S_qo // BLK_QO
    num_kv_blocks = S_kv // BLK_KV

    job_id = kdt.get_job_id()
    Q_start = job_id * BLK_QO
    Q_end = Q_start + BLK_QO

    Q_tile = kdt.alloc_spm((BLK_QO, D), dtype='float32')
    K_tiles = kdt.alloc_spm((2, BLK_KV, D), dtype='float32')
    V_tiles = kdt.alloc_spm((2, BLK_KV, D), dtype='float32')
    QK_score = kdt.alloc_spm((2, BLK_QO, BLK_KV), dtype='float32')
    O_tile = kdt.alloc_spm((BLK_QO, D), dtype='float32', init_value = 0)
    
    global_rowmax = kdt.alloc_spm((BLK_QO,), dtype='float32', init_value=inf)
    global_rowsum = kdt.alloc_spm((BLK_QO,), dtype='float32', init_value=0)

    local_rowmax = kdt.alloc_spm((BLK_QO,), dtype='float32')
    local_rowsum = kdt.alloc_spm((BLK_QO,), dtype='float32')

    scale_online_softmax = kdt.alloc_spm((BLK_QO,), dtype='float32')
    
    global_rowmax_new = kdt.alloc_spm((BLK_QO,), dtype='float32')
    global_rowsum_new = kdt.alloc_spm((BLK_QO,), dtype='float32')

    tmp1 = kdt.alloc_spm((BLK_QO,), dtype='float32')

    Q_global = io_tensors['Q']
    K_global = io_tensors['K']
    V_global = io_tensors['V']
    O_global = io_tensors['O']
    
    kdt.load(Q_global[Q_start:Q_end, :], Q_tile)
    kdt.load(K_global[0:BLK_KV, :], K_tiles[0, :, :])
    kdt.matmul(Q_tile, kdt.transpose(K_tiles[0], 0, 1), QK_score[0], accumulate=False)

    for kv_id in range(0, num_kv_blocks):
        stage = kv_id % 2
        K_tile = K_tiles[stage]
        V_tile = V_tiles[stage]
        # prefetch V_tile is not preferred because V_tile is use after Q@K which is long enough to overlap with the loading time of V_tile
        # prefetch V_tile creates dependency across loops like kv_id = 0 and kv_id = 2, which makes the pipeline stall
        kdt.load(V_global[kv_id*BLK_KV:(kv_id+1)*BLK_KV], V_tile)
        
        prefetch_kv_id = kv_id + 1
        prefetch_stage = prefetch_kv_id % 2
        if prefetch_kv_id < num_kv_blocks:
            KV_start = prefetch_kv_id * BLK_KV
            KV_end = KV_start + BLK_KV
            kdt.load(K_global[KV_start:KV_end, :], K_tiles[prefetch_stage])
        
        next_kv_id = kv_id + 1
        if next_kv_id < num_kv_blocks:
            next_stage = next_kv_id % 2
            # type: MXM
            kdt.matmul(Q_tile, kdt.transpose(K_tiles[next_stage], 0, 1), QK_score[next_stage], accumulate=False)
            # Q @ K^T (MXM) of next stage overlap w/ below mul & add & div VXM operations
        
        kdt.reduce(QK_score[stage], 1, 'max', local_rowmax)
        kdt.max(local_rowmax, global_rowmax, global_rowmax_new)
        
        kdt.sub(QK_score[stage], kdt.broadcast_to(kdt.unsqueeze(global_rowmax_new, 1), 1, BLK_KV), QK_score[stage])
        kdt.exp(QK_score[stage], QK_score[stage], e)

        # PV_tile = matmul(..., ...) WILL NOT overlap w/ VXM ops under because they share the same QK_score !!!
        kdt.reduce(QK_score[stage], 1, 'sum', local_rowsum)

        kdt.sub(global_rowmax, global_rowmax_new, scale_online_softmax)
        kdt.exp(scale_online_softmax, scale_online_softmax, e)
        kdt.fma(scale_online_softmax, global_rowsum, local_rowsum, global_rowsum_new)
        # fa v2 said here should be div?
        kdt.mul(O_tile, kdt.broadcast_to(kdt.unsqueeze(scale_online_softmax, 1), 1, D), O_tile)
        # PV_tile = matmul(..., ...) will overlap w/ VXM ops above
        

        # type: VXM
        kdt.matmul(QK_score[stage], V_tile, O_tile, accumulate = True)
        kdt.copy(global_rowmax_new, global_rowmax)
        kdt.copy(global_rowsum_new, global_rowsum)


    kdt.div(O_tile, kdt.broadcast_to(kdt.unsqueeze(global_rowsum, 1), 1, D), O_tile)
    kdt.store(O_tile, O_global[Q_start:Q_end, :])




def get_kernel(task_id: int) -> kdt.KernelFunction:
    if task_id == 4:
        return flash_attention_kernel
    return None
