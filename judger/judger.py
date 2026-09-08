from typing import Callable
import argparse
import importlib
import dataclasses
import sys
import os

import lib
from task1 import test_task1
from task2 import test_task2
from task3 import test_task3
from task4 import test_task4

@dataclasses.dataclass
class TaskSpec:
    task_name: str
    tester: Callable    # test_taskX
    num_testcases: int
    score: float

all_tasks = [
    TaskSpec('vector add', test_task1, 2, 10),
    TaskSpec('matmul', test_task2, 2, 20),
    TaskSpec('matmul w/ fine grain scale', test_task3, 2, 30),
    TaskSpec('flash attention', test_task4, 4, 40),
]

def main(args: argparse.Namespace):
    if not os.path.exists(args.kernel_impl_path):
        raise FileNotFoundError(f"Kernel implementation path '{args.kernel_impl_path}' does not exist.")
    if os.path.isdir(args.kernel_impl_path):
        raise IsADirectoryError(f"Kernel implementation path '{args.kernel_impl_path}' is a directory, expected a file.")
    try:
        # Add into sys.path the directory containing the player's kernel implementation
        kernel_impl_dir = os.path.dirname(os.path.abspath(args.kernel_impl_path))
        sys.path.insert(0, kernel_impl_dir)
        # Derive module name from the file name
        module_name = os.path.splitext(os.path.basename(args.kernel_impl_path))[0]
        players_module = importlib.import_module(module_name)
        if not hasattr(players_module, 'get_kernel'):
            raise AttributeError(f"Module '{module_name}' does not have a 'get_kernel' function.")
        players_kernel_getter = players_module.get_kernel
    except ImportError as e:
        print(f"Error: Could not import module '{args.kernel_impl_path}'. Error: {e}")
        return
    
    if args.task is None:
        tasks_to_test = [x+1 for x in range(len(all_tasks))]
    else:
        if args.task <= 0 or args.task > len(all_tasks):
            raise ValueError(f"Invalid task ID {args.task}. Must be between 1 and {len(all_tasks)}.")
        tasks_to_test = [args.task]
    
    total_score = 0.0
    for task_idx in tasks_to_test:
        print(f"Running tests for Task {task_idx}...")
        task_spec = all_tasks[task_idx-1]
        players_kernel = players_kernel_getter(task_idx)
        score_of_task = 0.0

        for testcase_idx in range(task_spec.num_testcases):
            if args.profile_path is not None:
                os.makedirs(args.profile_path, exist_ok=True)
                lib.PROFILE_PATH = os.path.join(
                    args.profile_path, f"task{task_idx}_case{testcase_idx}.json"
                )
            score = task_spec.tester(testcase_idx, players_kernel, args.print_ir)
            score_of_task += score * (1.0 / task_spec.num_testcases)
        cur_task_score = score_of_task / 100.0 * task_spec.score
        total_score += cur_task_score
        print(f"Task {task_idx} ({task_spec.task_name}) score: {cur_task_score:.1f} / {task_spec.score}")
    print(f"Total score: {total_score:.1f} / 100.0")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KDT-TPU Kernel Judger")
    parser.add_argument('--task', type=int, help="Task ID to test (e.g., 1 for Task 1)")
    parser.add_argument('--kernel-impl-path', type=str, required=True, help="Path to the player's kernel implementation")
    parser.add_argument('--print-ir', action='store_true')
    parser.add_argument('--profile-path', type=str, default=None,
                        help="Directory to write Perfetto/Chrome-trace JSON profiles into.")
    parser.add_argument('--profile-jobs', type=str, default=None,
                        help="Comma separated job ids to profile (default: all jobs).")
    parsed_args = parser.parse_args()

    if parsed_args.profile_jobs:
        lib.PROFILE_JOBS = [int(x) for x in parsed_args.profile_jobs.replace(" ", "").split(",") if x != ""]

    main(parsed_args)

