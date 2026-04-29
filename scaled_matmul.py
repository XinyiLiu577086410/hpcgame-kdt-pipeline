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
    as_tile = kdt.alloc_spm((BLOCK_SIZE, BLOCK_K_SIZE // 64), dtype='float32')
    bs_tile = kdt.alloc_spm((BLOCK_K_SIZE // 64, BLOCK_SIZE), dtype='float32')

    kdt.load(io_tensors["Ab"][M_base:M_base+BLOCK_SIZE,0:0+BLOCK_K_SIZE//2], a_tile[:,0:BLOCK_K_SIZE//2])
    kdt.load(io_tensors["As"][M_base:M_base+BLOCK_SIZE,0//64:(0+BLOCK_K_SIZE//2)//64], as_tile[:,0:BLOCK_K_SIZE//2//64])
    kdt.load(io_tensors["Bb"][0:0+BLOCK_K_SIZE//2,N_base:N_base+BLOCK_SIZE], b_tile[0:BLOCK_K_SIZE//2,:])
    kdt.load(io_tensors["Bs"][0//64:(0+BLOCK_K_SIZE//2)//64,N_base:N_base+BLOCK_SIZE], bs_tile[0:BLOCK_K_SIZE//2//64,:])
    for k_id in range(num_k_blocks):
        K_base = k_id * BLOCK_K_SIZE
        kdt.mul(a_tile[:, 0:64], kdt.broadcast_to(as_tile[:, 0:1], 1, 64), a_tile[:, 0:64])
        kdt.mul(a_tile[:, 64:128], kdt.broadcast_to(as_tile[:, 1:2], 1, 64), a_tile[:, 64:128])
        kdt.mul(b_tile[0:64, :], kdt.broadcast_to(bs_tile[0:1, :], 0, 64), b_tile[0:64, :])
        kdt.mul(b_tile[64:128, :], kdt.broadcast_to(bs_tile[1:2, :], 0, 64), b_tile[64:128, :])
        kdt.matmul(a_tile[:,0:BLOCK_K_SIZE//2], b_tile[0:BLOCK_K_SIZE//2,:], c_tile, accumulate=True)
        kdt.load(io_tensors["Ab"][M_base:M_base+BLOCK_SIZE,K_base+BLOCK_K_SIZE//2:K_base+BLOCK_K_SIZE], a_tile[:,BLOCK_K_SIZE//2:BLOCK_K_SIZE])
        kdt.load(io_tensors["As"][M_base:M_base+BLOCK_SIZE,(K_base+BLOCK_K_SIZE//2)//64:(K_base+BLOCK_K_SIZE)//64], as_tile[:,BLOCK_K_SIZE//2//64:BLOCK_K_SIZE//64])
        kdt.load(io_tensors["Bb"][K_base+BLOCK_K_SIZE//2:K_base+BLOCK_K_SIZE,N_base:N_base+BLOCK_SIZE], b_tile[BLOCK_K_SIZE//2:BLOCK_K_SIZE,:])
        kdt.load(io_tensors["Bs"][(K_base+BLOCK_K_SIZE//2)//64:(K_base+BLOCK_K_SIZE)//64,N_base:N_base+BLOCK_SIZE], bs_tile[BLOCK_K_SIZE//2//64:BLOCK_K_SIZE//64,:])
        kdt.mul(a_tile[:, 128:192], kdt.broadcast_to(as_tile[:, 2:3], 1, 64), a_tile[:, 128:192])
        if k_id + 1 < num_k_blocks:
            K_base_nxt = K_base + BLOCK_K_SIZE
            kdt.load(io_tensors["Ab"][M_base:M_base+BLOCK_SIZE,K_base_nxt:K_base_nxt+BLOCK_K_SIZE//2], a_tile[:,0:BLOCK_K_SIZE//2])
            kdt.load(io_tensors["As"][M_base:M_base+BLOCK_SIZE,K_base_nxt//64:(K_base_nxt+BLOCK_K_SIZE//2)//64], as_tile[:,0:BLOCK_K_SIZE//2//64])
            kdt.load(io_tensors["Bb"][K_base_nxt:K_base_nxt+BLOCK_K_SIZE//2,N_base:N_base+BLOCK_SIZE], b_tile[0:BLOCK_K_SIZE//2,:])
            kdt.load(io_tensors["Bs"][K_base_nxt//64:(K_base_nxt+BLOCK_K_SIZE//2)//64,N_base:N_base+BLOCK_SIZE], bs_tile[0:BLOCK_K_SIZE//2//64,:])
        kdt.mul(a_tile[:, 192:256], kdt.broadcast_to(as_tile[:, 3:4], 1, 64), a_tile[:, 192:256])
        kdt.mul(b_tile[128:192, :], kdt.broadcast_to(bs_tile[2:3, :], 0, 64), b_tile[128:192, :])
        kdt.mul(b_tile[192:256, :], kdt.broadcast_to(bs_tile[3:4, :], 0, 64), b_tile[192:256, :])
        kdt.matmul(a_tile[:,BLOCK_K_SIZE//2:BLOCK_K_SIZE], b_tile[BLOCK_K_SIZE//2:BLOCK_K_SIZE,:], c_tile, accumulate=True)
    kdt.store(c_tile, io_tensors["C"][M_base:M_base+BLOCK_SIZE,N_base:N_base+BLOCK_SIZE])

def get_kernel(task_id: int) -> kdt.KernelFunction:
    if task_id == 3:
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