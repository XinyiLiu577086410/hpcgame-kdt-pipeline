from itertools import accumulate
from typing import Dict

import kdt
import torch

def calculate_num_jobs(task_args: dict[str, int]) -> int:
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    return (task_args["M"] // BLOCK_SIZE_M) * (task_args["N"] // BLOCK_SIZE_N)

@kdt.kernel(num_jobs_calculator=calculate_num_jobs)
def matmul_kernel(task_args: Dict[str, int], io_tensors: Dict[str, kdt.Tile]):
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    N = task_args['N']
    M = task_args['M']
    K = task_args['K']
    num_m_blocks = M // BLOCK_SIZE_M  # 计算需要处理的块数
    num_n_blocks = N // BLOCK_SIZE_N  # 计算需要处理的块数
    num_k_blocks = K // BLOCK_SIZE_K  # 计算需要处理的块数
    # 分配 SPM 上的数据块
    job_id = kdt.get_job_id()
    job_id_m = job_id % num_m_blocks
    job_id_n = job_id // num_m_blocks
    M_start = job_id_m * BLOCK_SIZE_M
    M_end = M_start + BLOCK_SIZE_M
    N_start = job_id_n * BLOCK_SIZE_N
    N_end = N_start + BLOCK_SIZE_N

    NUM_STAGES = 3
    Ab_tile = kdt.alloc_spm((NUM_STAGES, BLOCK_SIZE_M, BLOCK_SIZE_K), dtype='float32')
    Bb_tile = kdt.alloc_spm((NUM_STAGES, BLOCK_SIZE_K, BLOCK_SIZE_N), dtype='float32')
    As_tile = kdt.alloc_spm((NUM_STAGES, BLOCK_SIZE_M, 1), dtype='float32')
    Bs_tile = kdt.alloc_spm((NUM_STAGES, 1, BLOCK_SIZE_N), dtype='float32')
    scale_temp = kdt.alloc_spm((2, BLOCK_SIZE_M, BLOCK_SIZE_N), dtype='float32', init_value = 0.0)
    c_tile_temp = kdt.alloc_spm((2, BLOCK_SIZE_M, BLOCK_SIZE_N), dtype='float32', init_value = 0.0)
    c_tile = kdt.alloc_spm((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype='float32', init_value = 0.0)

    for stage in range(0, NUM_STAGES - 1):
        k = stage
        K_start = k * BLOCK_SIZE_K
        K_end = K_start + BLOCK_SIZE_K
        kdt.load(io_tensors['Ab'][M_start:M_end, K_start:K_end], Ab_tile[stage, :, :])
        kdt.load(io_tensors['Bb'][K_start:K_end, N_start:N_end], Bb_tile[stage, :, :])
        kdt.load(io_tensors['As'][M_start:M_end, k:k+1], As_tile[stage, :, :])
        kdt.load(io_tensors['Bs'][k:k+1, N_start:N_end], Bs_tile[stage, :, :])

    for k in range(num_k_blocks):
        stage = k % NUM_STAGES
        K_start = k * BLOCK_SIZE_K
        K_end = K_start + BLOCK_SIZE_K
    
        kdt.matmul(Ab_tile[stage], Bb_tile[stage], c_tile_temp[k%2], accumulate = False)
        kdt.mul(
            kdt.broadcast_to(As_tile[stage], 1, BLOCK_SIZE_N),
            kdt.broadcast_to(Bs_tile[stage], 0, BLOCK_SIZE_M), 
            scale_temp[k%2]
        )
        next_k = k + NUM_STAGES -1
        next_stage = next_k % NUM_STAGES
        if next_k < num_k_blocks: 
            next_K_start = next_k * BLOCK_SIZE_K
            next_K_end = next_K_start + BLOCK_SIZE_K
            kdt.load(io_tensors['Ab'][M_start:M_end, next_K_start:next_K_end], Ab_tile[next_stage, :, :])
            kdt.load(io_tensors['Bb'][next_K_start:next_K_end, N_start:N_end], Bb_tile[next_stage, :, :])
            kdt.load(io_tensors['As'][M_start:M_end, next_k:next_k+1], As_tile[next_stage, :, :])
            kdt.load(io_tensors['Bs'][next_k:next_k+1, N_start:N_end], Bs_tile[next_stage, :, :])
        if k != 0:
            kdt.fma(
                scale_temp[k%2^1],
                c_tile_temp[k%2^1],
                c_tile,
                c_tile
            )
    kdt.fma(
        scale_temp[num_k_blocks%2^1],
        c_tile_temp[num_k_blocks%2^1],
        c_tile,
        c_tile
    )
    kdt.store(c_tile, io_tensors["C"][M_start:M_end,N_start:N_end])

def get_kernel(task_id: int) -> kdt.KernelFunction:
    if task_id == 3:
        return matmul_kernel
    return None
