"""
Kernel launch function.
"""

import os
from typing import Dict, Any, List, Optional
import torch

from kdt.language.errors import CompilationError
from kdt.language.frontend import CompiledKernel, IOTensorMeta
from kdt.simulator import Simulator, TPUSpec

def launch_kernel(
    kernel_func: CompiledKernel,
    io_tensors: Dict[str, torch.Tensor],
    tpu_spec: TPUSpec,
    profile_path: Optional[str] = None,
    profile_jobs: Optional[List[int]] = None,
) -> int:
    """
    Launch a KDT-DSL kernel.

    Args:
        kernel_func: Kernel function decorated with @kdt.kernel and compiled via `.compile()`
        task_args: Dictionary of task arguments
        io_tensors: Dictionary mapping tensor names to PyTorch tensors
        profile_path: Optional path to write a Perfetto/Chrome-trace JSON profile.
            Can also be set through the ``KDT_PROFILE_PATH`` environment variable.
        profile_jobs: Optional list of job ids to profile (default: all jobs).
            Can also be set through the ``KDT_PROFILE_JOBS`` environment variable
            as a comma separated list of integers.

    Returns:
        int: Number of cycles taken to execute the kernel
    """
    # Check that kernel_func is a CompiledKenrel
    if not isinstance(kernel_func, CompiledKernel):
        raise CompilationError(
            "First argument must be a CompiledKernel"
        )

    if profile_path is None:
        profile_path = os.environ.get("KDT_PROFILE_PATH") or None
    if profile_jobs is None:
        env_jobs = os.environ.get("KDT_PROFILE_JOBS")
        if env_jobs:
            profile_jobs = [int(x) for x in env_jobs.replace(" ", "").split(",") if x != ""]

    # Validate input/output tensors match kernel expectations
    expected_io_tensors = kernel_func.io_tensors_meta
    for name, meta in expected_io_tensors.items():
        if name not in io_tensors:
            raise ValueError(f"IO tensor '{name}' expected by kernel but not provided")
        tensor = io_tensors[name]
        cur_tensor_meta = IOTensorMeta.from_tensor(tensor)
        if cur_tensor_meta.shape != meta.shape or cur_tensor_meta.dtype != meta.dtype:
            raise ValueError(
                f"IO tensor '{name}' has shape {cur_tensor_meta.shape} and dtype {cur_tensor_meta.dtype}, "
                f"but kernel expects shape {meta.shape} and dtype {meta.dtype}"
            )
        if tensor.device.type != 'cpu':
            raise ValueError(f"IO tensor '{name}' must be on CPU for simulation")
        
    for name in io_tensors.keys():
        if name not in expected_io_tensors:
            raise ValueError(f"IO tensor '{name}' provided but not expected by kernel")

    # Execute kernel using NumPy simulator
    num_cycles = Simulator.simulate(
        kernel_func,
        io_tensors,
        tpu_spec,
        profile_path=profile_path,
        profile_jobs=profile_jobs,
    )
    
    return num_cycles
