from itertools import accumulate
from typing import Dict

import kdt
import torch

def calculate_num_jobs(task_args: dict[str, int]) -> int:
    BLOCK_SIZE = 128
    return (task_args["M"] // BLOCK_SIZE) * (task_args["N"] // BLOCK_SIZE)

@kdt.kernel(num_jobs_calculator=calculate_num_jobs)
def matmul_kernel(task_args: Dict[str, int], io_tensors: Dict[str, kdt.Tile]):
    BLOCK_SIZE = 128
    BLOCK_K_SIZE = 256
    N = task_args['N']
    M = task_args['M']
    K = task_args['K']
    num_k_blocks = K // BLOCK_K_SIZE  # 计算需要处理的块数
    num_m_blocks = M // BLOCK_SIZE  # 计算需要处理的块数
    num_n_blocks = N // BLOCK_SIZE  # 计算需要处理的块数
    # 分配 SPM 上的数据块
    job_id = kdt.get_job_id()
    job_id_m = job_id % num_m_blocks
    job_id_n = job_id // num_m_blocks
    M_base = job_id_m * BLOCK_SIZE
    N_base = job_id_n * BLOCK_SIZE

    a_tile = kdt.alloc_spm((BLOCK_SIZE, BLOCK_K_SIZE), dtype='float32')
    b_tile = kdt.alloc_spm((BLOCK_K_SIZE, BLOCK_SIZE), dtype='float32')
    c_tile = kdt.alloc_spm((BLOCK_SIZE, BLOCK_SIZE), dtype='float32', init_value = 0.0)

    for k_id in range(num_k_blocks):
        K_base = k_id * BLOCK_K_SIZE
        kdt.load(io_tensors["A"][M_base:M_base+BLOCK_SIZE,K_base:K_base+BLOCK_K_SIZE], a_tile)
        kdt.load(io_tensors["B"][K_base:K_base+BLOCK_K_SIZE,N_base:N_base+BLOCK_SIZE], b_tile)
        kdt.matmul(a_tile, b_tile, c_tile, accumulate=True)
    kdt.store(c_tile, io_tensors["C"][M_base:M_base+BLOCK_SIZE,N_base:N_base+BLOCK_SIZE])

def get_kernel(task_id: int) -> kdt.KernelFunction:
    if task_id == 2:
        return matmul_kernel
    return None

def main():
    M, N, K = 512, 1024, 2560
    a = torch.randn((M, K), dtype=torch.float32).clamp(-1, 1)
    b = torch.randn((K, N), dtype=torch.float32).clamp(-1, 1)
    c = torch.zeros((M, N), dtype=torch.float32)

    task_args = {'M': M, 'N': N, 'K': K}
    io_tensors = {'A': a, 'B': b, 'C': c}

    compiled_kernel = matmul_kernel.compile(task_args, io_tensors)

    tpu_spec = kdt.TPUSpec(num_sms=32, load_store_latency=1000, spm_size=384*1024)
    num_cycles = kdt.launch_kernel(compiled_kernel, io_tensors, tpu_spec)

    c_ref = a @ b
    print("Result C:", c)
    print(torch.isclose(c, c_ref, atol=1e-5, rtol=2e-3))
    assert torch.allclose(c, c_ref, atol=1e-5, rtol=2e-3), "Result incorrect!"
    print(f"Kernel executed in {num_cycles} cycles.")
    
if __name__ == '__main__':
    main()