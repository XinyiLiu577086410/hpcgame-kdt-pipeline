"""
KDT-DSL IR Simulator

This module provides a functional simulator that executes KDT-DSL IR using NumPy on CPU.
"""

from .simulator import Simulator
from .execution_context import ExecutionContext
from .execution_visitor import ExecutionVisitor
from .tpu_spec import TPUSpec
from .profiler import TraceRecorder, write_perfetto_trace

__all__ = [
    "Simulator",
    "ExecutionContext",
    "ExecutionVisitor",
    "TPUSpec",
    "TraceRecorder",
    "write_perfetto_trace",
]