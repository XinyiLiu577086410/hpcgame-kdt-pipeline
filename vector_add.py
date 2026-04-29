from typing import Dict

import kdt
import torch

def calculate_num_jobs(task_args: dict[str, int]) -> int:
    return 1

@kdt.kernel(num_jobs_calculator=calculate_num_jobs)
def vector_add_kernel(task_args: Dict[str, int], io_tensors: Dict[str, kdt.Tile]):
    BLOCK_SIZE = 8192
    vec_size = task_args['N']
    num_blocks = vec_size // BLOCK_SIZE  # 计算需要处理的块数
    # 分配 SPM 上的数据块
    a_tile = kdt.alloc_spm((BLOCK_SIZE,), dtype='float32')
    b_tile = kdt.alloc_spm((BLOCK_SIZE,), dtype='float32')
    c_tile = kdt.alloc_spm((BLOCK_SIZE,), dtype='float32')
    d_tile = kdt.alloc_spm((BLOCK_SIZE,), dtype='float32')

    kdt.load(io_tensors['a'][0 * BLOCK_SIZE: (0 + 1) * BLOCK_SIZE], a_tile)
    kdt.load(io_tensors['b'][0 *  BLOCK_SIZE: (0 + 1) * BLOCK_SIZE], b_tile)
    for i in range(num_blocks // 2):
        j = i * 2
        # 执行向量加法
        kdt.add(a_tile, b_tile, b_tile)
        # 加载输入数据到 SPM
        kdt.load(io_tensors['a'][(j + 1) * BLOCK_SIZE: (j + 2) * BLOCK_SIZE], c_tile)
        kdt.load(io_tensors['b'][(j + 1) * BLOCK_SIZE: (j + 2) * BLOCK_SIZE], d_tile)
        # 执行向量加法
        kdt.add(c_tile, d_tile, d_tile)
        # 存储结果回显存
        kdt.store(b_tile, io_tensors['c'][j * BLOCK_SIZE: (j + 1) * BLOCK_SIZE])
        if j + 2 < num_blocks:
            kdt.load(io_tensors['a'][(j + 2) * BLOCK_SIZE: (j + 3) * BLOCK_SIZE], a_tile)
            kdt.load(io_tensors['b'][(j + 2) * BLOCK_SIZE: (j + 3) * BLOCK_SIZE], b_tile)
        kdt.store(d_tile, io_tensors['c'][(j + 1) * BLOCK_SIZE: (j + 2) * BLOCK_SIZE])

def get_kernel(task_id: int) -> kdt.KernelFunction:
    if task_id == 1:
        return vector_add_kernel
    return None


def main():
    vec_size = 16  # 向量大小
    a = torch.randn(vec_size, dtype=torch.float32)  # 输入向量 a
    b = torch.randn(vec_size, dtype=torch.float32)  # 输入向量 b
    c = torch.empty(vec_size, dtype=torch.float32)  # 输出向量 c

    task_args = {'vec_size': vec_size}
    io_tensors = {'a': a, 'b': b, 'c': c}

    compiled_kernel = vector_add_kernel.compile(task_args, io_tensors)
    # compiled_kernel.print_ir()  # 打印中间表示（IR）

    tpu_spec = kdt.TPUSpec(num_sms=1, load_store_latency=100)
    num_cycles = kdt.launch_kernel(compiled_kernel, io_tensors, tpu_spec)

    print("Result c:", c)
    assert torch.allclose(c, a + b), "结果不正确！"
    print(f"Kernel executed in {num_cycles} cycles.")
    
if __name__ == '__main__':
    main()