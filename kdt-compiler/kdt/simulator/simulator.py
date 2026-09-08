"""
Main simulator for KDT-DSL IR.

Executes CompiledKernel using NumPy on CPU, handling multiple jobs sequentially.
"""

import time
import numpy as np
import torch
from typing import Dict, List
from kdt.language.frontend import CompiledKernel
from .execution_context import ExecutionContext
from .execution_visitor import ExecutionVisitor
from .tpu_spec import TPUSpec

class Simulator:
    """
    Simulator for KDT-DSL IR.

    Executes a CompiledKernel on CPU using NumPy, simulating the KDT-TPU execution model.

    Simulate execution cycles at the same time, return the ending cycle of every job
    """

    @staticmethod
    def simulate(compiled_kernel: CompiledKernel, io_tensors: Dict[str, torch.Tensor], tpu_spec: TPUSpec, time_limit: float = 20.0,
                 profile_path: str = None, profile_jobs: List[int] = None) -> int:
        """
        Simulate execution of a compiled kernel.

        Args:
            compiled_kernel: Compiled kernel with IR and task arguments
            io_tensors: Dictionary mapping tensor names to PyTorch tensors
            tpu_spec: Specification of the TPU hardware, including mem latency, VXM/MXM trpt, etc.
            time_limit: Maximum allowed simulation time in seconds
            profile_path: If set, write a Perfetto/Chrome-trace JSON file to this path.
            profile_jobs: If set, only profile these job ids (default: all jobs).
        """
        kernel_ir = compiled_kernel.kernel_ir
        start_lineno = compiled_kernel.start_lineno

        simulation_start_time = time.time()
        simulation_end_time_limit = simulation_start_time + time_limit

        tracer = None
        if profile_path is not None:
            from .profiler import TraceRecorder
            tracer = TraceRecorder()
            if profile_jobs is not None:
                profile_jobs = set(profile_jobs)

        # Execute each job sequentially
        cycle_usages = []
        for job_id in range(kernel_ir.num_jobs):
            if time.time() > simulation_end_time_limit:
                raise TimeoutError(f"Simulation exceeded time limit of {time_limit} seconds.")
            context = ExecutionContext(job_id, start_lineno)

            for tile_storage in kernel_ir.io_tile_storage_defs:
                tensor = io_tensors[tile_storage.name]
                array = tensor.detach().numpy()
                context.allocate_global_storage(tile_storage, array)
            
            for tile_storage in kernel_ir.spm_tile_storage_defs:
                context.allocate_spm_storage(tile_storage)

            if context.allocated_spm_size > tpu_spec.spm_size:
                raise RuntimeError(f"SPM size exceeded: job id {job_id}, allocated {context.allocated_spm_size} bytes, but SPM size is {tpu_spec.spm_size} bytes.")

            visitor = ExecutionVisitor(context, tpu_spec, simulation_end_time_limit)
            if tracer is not None and (profile_jobs is None or job_id in profile_jobs):
                visitor.set_tracer(tracer)
            kernel_ir.accept(visitor)

            cycle_usages.append(visitor.get_cycle_usage())

        # NOTE. Since we've assigned input/output tensors as in-place numpy array views, we don't need to update output tensors

        # Calculate the cycle usage of the entire kernel from all jobs
        # print(cycle_usages)
        sm_job_end_time = [0 for _ in range(tpu_spec.num_sms)]
        job_sm = [0 for _ in range(kernel_ir.num_jobs)]
        job_start = [0 for _ in range(kernel_ir.num_jobs)]
        for job_id in range(kernel_ir.num_jobs):
            sm_idx = sm_job_end_time.index(min(sm_job_end_time))
            job_sm[job_id] = sm_idx
            job_start[job_id] = sm_job_end_time[sm_idx]
            sm_job_end_time[sm_idx] += cycle_usages[job_id]
        final_cycle_usage = max(sm_job_end_time)

        if tracer is not None:
            from .profiler import write_perfetto_trace
            write_perfetto_trace(
                tracer.events,
                job_sm,
                job_start,
                cycle_usages,
                profile_path,
                tpu_spec.num_sms,
            )

        return final_cycle_usage
