from typing import Dict

import kdt


def _task1_num_jobs(task_args: dict[str, int]) -> int:
    return 1


@kdt.kernel(num_jobs_calculator=_task1_num_jobs)
def _task1_kernel(task_args: Dict[str, int], io_tensors: Dict[str, kdt.Tile]):
    CH = 2048
    GROUP = 16384
    N = task_args["N"]

    a0 = kdt.alloc_spm((CH,), dtype="float32")
    b0 = kdt.alloc_spm((CH,), dtype="float32")
    a1 = kdt.alloc_spm((CH,), dtype="float32")
    b1 = kdt.alloc_spm((CH,), dtype="float32")
    a2 = kdt.alloc_spm((CH,), dtype="float32")
    b2 = kdt.alloc_spm((CH,), dtype="float32")
    a3 = kdt.alloc_spm((CH,), dtype="float32")
    b3 = kdt.alloc_spm((CH,), dtype="float32")
    a4 = kdt.alloc_spm((CH,), dtype="float32")
    b4 = kdt.alloc_spm((CH,), dtype="float32")
    a5 = kdt.alloc_spm((CH,), dtype="float32")
    b5 = kdt.alloc_spm((CH,), dtype="float32")
    a6 = kdt.alloc_spm((CH,), dtype="float32")
    b6 = kdt.alloc_spm((CH,), dtype="float32")
    a7 = kdt.alloc_spm((CH,), dtype="float32")
    b7 = kdt.alloc_spm((CH,), dtype="float32")

    kdt.load(io_tensors["a"][0:CH], a0)
    kdt.load(io_tensors["b"][0:CH], b0)
    kdt.load(io_tensors["a"][CH:CH * 2], a1)
    kdt.load(io_tensors["b"][CH:CH * 2], b1)
    kdt.load(io_tensors["a"][CH * 2:CH * 3], a2)
    kdt.load(io_tensors["b"][CH * 2:CH * 3], b2)
    kdt.load(io_tensors["a"][CH * 3:CH * 4], a3)
    kdt.load(io_tensors["b"][CH * 3:CH * 4], b3)
    kdt.load(io_tensors["a"][CH * 4:CH * 5], a4)
    kdt.load(io_tensors["b"][CH * 4:CH * 5], b4)
    kdt.load(io_tensors["a"][CH * 5:CH * 6], a5)
    kdt.load(io_tensors["b"][CH * 5:CH * 6], b5)
    kdt.load(io_tensors["a"][CH * 6:CH * 7], a6)
    kdt.load(io_tensors["b"][CH * 6:CH * 7], b6)
    kdt.load(io_tensors["a"][CH * 7:CH * 8], a7)
    kdt.load(io_tensors["b"][CH * 7:CH * 8], b7)

    for g in range(0, N // GROUP):
        base = g * GROUP
        next_base = base + GROUP

        kdt.add(a0, b0, a0)
        kdt.store(a0, io_tensors["c"][base:base + CH])
        if g > 0:
            kdt.load(io_tensors["a"][base + CH * 4:base + CH * 5], a4)
            kdt.load(io_tensors["b"][base + CH * 4:base + CH * 5], b4)

        kdt.add(a1, b1, a1)
        kdt.store(a1, io_tensors["c"][base + CH:base + CH * 2])
        if g > 0:
            kdt.load(io_tensors["a"][base + CH * 5:base + CH * 6], a5)
            kdt.load(io_tensors["b"][base + CH * 5:base + CH * 6], b5)

        kdt.add(a2, b2, a2)
        kdt.store(a2, io_tensors["c"][base + CH * 2:base + CH * 3])
        if g > 0:
            kdt.load(io_tensors["a"][base + CH * 6:base + CH * 7], a6)
            kdt.load(io_tensors["b"][base + CH * 6:base + CH * 7], b6)

        kdt.add(a3, b3, a3)
        kdt.store(a3, io_tensors["c"][base + CH * 3:base + CH * 4])
        if g > 0:
            kdt.load(io_tensors["a"][base + CH * 7:base + CH * 8], a7)
            kdt.load(io_tensors["b"][base + CH * 7:base + CH * 8], b7)

        kdt.add(a4, b4, a4)
        kdt.store(a4, io_tensors["c"][base + CH * 4:base + CH * 5])
        if g + 1 < N // GROUP:
            kdt.load(io_tensors["a"][next_base:next_base + CH], a0)
            kdt.load(io_tensors["b"][next_base:next_base + CH], b0)

        kdt.add(a5, b5, a5)
        kdt.store(a5, io_tensors["c"][base + CH * 5:base + CH * 6])
        if g + 1 < N // GROUP:
            kdt.load(io_tensors["a"][next_base + CH:next_base + CH * 2], a1)
            kdt.load(io_tensors["b"][next_base + CH:next_base + CH * 2], b1)

        kdt.add(a6, b6, a6)
        kdt.store(a6, io_tensors["c"][base + CH * 6:base + CH * 7])
        if g + 1 < N // GROUP:
            kdt.load(io_tensors["a"][next_base + CH * 2:next_base + CH * 3], a2)
            kdt.load(io_tensors["b"][next_base + CH * 2:next_base + CH * 3], b2)

        kdt.add(a7, b7, a7)
        kdt.store(a7, io_tensors["c"][base + CH * 7:base + CH * 8])
        if g + 1 < N // GROUP:
            kdt.load(io_tensors["a"][next_base + CH * 3:next_base + CH * 4], a3)
            kdt.load(io_tensors["b"][next_base + CH * 3:next_base + CH * 4], b3)


def _task2_num_jobs(task_args: dict[str, int]) -> int:
    return (task_args["M"] // 128) * (task_args["N"] // 128)


@kdt.kernel(num_jobs_calculator=_task2_num_jobs)
def _task2_kernel(task_args: Dict[str, int], io_tensors: Dict[str, kdt.Tile]):
    BM = 128
    BN = 128
    BK = 128
    M = task_args["M"]
    N = task_args["N"]
    K = task_args["K"]
    job_id = kdt.get_job_id()
    n_tiles = N // BN
    tile_m = job_id // n_tiles
    tile_n = job_id % n_tiles
    row_start = tile_m * BM
    row_end = row_start + BM
    col_start = tile_n * BN
    col_end = col_start + BN

    a0 = kdt.alloc_spm((BM, BK), dtype="float32")
    b0 = kdt.alloc_spm((BK, BN), dtype="float32")
    a1 = kdt.alloc_spm((BM, BK), dtype="float32")
    b1 = kdt.alloc_spm((BK, BN), dtype="float32")
    c = kdt.alloc_spm((BM, BN), dtype="float32")

    kdt.load(io_tensors["A"][row_start:row_end, 0:BK], a0)
    kdt.load(io_tensors["B"][0:BK, col_start:col_end], b0)
    kdt.load(io_tensors["A"][row_start:row_end, BK:BK + BK], a1)
    kdt.load(io_tensors["B"][BK:BK + BK, col_start:col_end], b1)

    for p in range(0, K // 256):
        kdt.matmul(a0, b0, c, accumulate=True)
        if p + 1 < K // 256:
            kdt.load(io_tensors["A"][row_start:row_end, p * 256 + 256:p * 256 + 384], a0)
            kdt.load(io_tensors["B"][p * 256 + 256:p * 256 + 384, col_start:col_end], b0)
        kdt.matmul(a1, b1, c, accumulate=True)
        if p + 1 < K // 256:
            kdt.load(io_tensors["A"][row_start:row_end, p * 256 + 384:p * 256 + 512], a1)
            kdt.load(io_tensors["B"][p * 256 + 384:p * 256 + 512, col_start:col_end], b1)

    kdt.store(c, io_tensors["C"][row_start:row_end, col_start:col_end])


def _task3_num_jobs(task_args: dict[str, int]) -> int:
    return (task_args["M"] // 128) * (task_args["N"] // 128)


@kdt.kernel(num_jobs_calculator=_task3_num_jobs)
def _task3_kernel(task_args: Dict[str, int], io_tensors: Dict[str, kdt.Tile]):
    BM = 128
    BN = 128
    BK = 128
    SG = 64
    M = task_args["M"]
    N = task_args["N"]
    K = task_args["K"]
    job_id = kdt.get_job_id()
    n_tiles = N // BN
    tile_m = job_id // n_tiles
    tile_n = job_id % n_tiles
    row_start = tile_m * BM
    row_end = row_start + BM
    col_start = tile_n * BN
    col_end = col_start + BN

    a0 = kdt.alloc_spm((BM, BK), dtype="float32")
    b0 = kdt.alloc_spm((BK, BN), dtype="float32")
    a1 = kdt.alloc_spm((BM, BK), dtype="float32")
    b1 = kdt.alloc_spm((BK, BN), dtype="float32")
    a2 = kdt.alloc_spm((BM, BK), dtype="float32")
    b2 = kdt.alloc_spm((BK, BN), dtype="float32")
    as0 = kdt.alloc_spm((BM, 2), dtype="float32")
    bs0 = kdt.alloc_spm((2, BN), dtype="float32")
    as1 = kdt.alloc_spm((BM, 2), dtype="float32")
    bs1 = kdt.alloc_spm((2, BN), dtype="float32")
    as2 = kdt.alloc_spm((BM, 2), dtype="float32")
    bs2 = kdt.alloc_spm((2, BN), dtype="float32")
    c = kdt.alloc_spm((BM, BN), dtype="float32")

    kdt.load(io_tensors["Ab"][row_start:row_end, 0:BK], a0)
    kdt.load(io_tensors["Bb"][0:BK, col_start:col_end], b0)
    kdt.load(io_tensors["As"][row_start:row_end, 0:2], as0)
    kdt.load(io_tensors["Bs"][0:2, col_start:col_end], bs0)
    kdt.load(io_tensors["Ab"][row_start:row_end, BK:BK + BK], a1)
    kdt.load(io_tensors["Bb"][BK:BK + BK, col_start:col_end], b1)
    kdt.load(io_tensors["As"][row_start:row_end, 2:4], as1)
    kdt.load(io_tensors["Bs"][2:4, col_start:col_end], bs1)

    as0_0 = kdt.broadcast_to(kdt.unsqueeze(as0[:, 0], 1), 1, SG)
    as0_1 = kdt.broadcast_to(kdt.unsqueeze(as0[:, 1], 1), 1, SG)
    bs0_0 = kdt.broadcast_to(kdt.unsqueeze(bs0[0, :], 0), 0, SG)
    bs0_1 = kdt.broadcast_to(kdt.unsqueeze(bs0[1, :], 0), 0, SG)
    as1_0 = kdt.broadcast_to(kdt.unsqueeze(as1[:, 0], 1), 1, SG)
    as1_1 = kdt.broadcast_to(kdt.unsqueeze(as1[:, 1], 1), 1, SG)
    bs1_0 = kdt.broadcast_to(kdt.unsqueeze(bs1[0, :], 0), 0, SG)
    bs1_1 = kdt.broadcast_to(kdt.unsqueeze(bs1[1, :], 0), 0, SG)
    as2_0 = kdt.broadcast_to(kdt.unsqueeze(as2[:, 0], 1), 1, SG)
    as2_1 = kdt.broadcast_to(kdt.unsqueeze(as2[:, 1], 1), 1, SG)
    bs2_0 = kdt.broadcast_to(kdt.unsqueeze(bs2[0, :], 0), 0, SG)
    bs2_1 = kdt.broadcast_to(kdt.unsqueeze(bs2[1, :], 0), 0, SG)

    kdt.mul(a0[:, 0:SG], as0_0, a0[:, 0:SG])
    kdt.mul(a0[:, SG:BK], as0_1, a0[:, SG:BK])
    kdt.mul(b0[0:SG, :], bs0_0, b0[0:SG, :])
    kdt.mul(b0[SG:BK, :], bs0_1, b0[SG:BK, :])

    for p in range(0, K // 384):
        block_start = p * 384

        kdt.matmul(a0, b0, c, accumulate=True)

        kdt.mul(a1[:, 0:SG], as1_0, a1[:, 0:SG])
        kdt.mul(a1[:, SG:BK], as1_1, a1[:, SG:BK])
        kdt.mul(b1[0:SG, :], bs1_0, b1[0:SG, :])
        kdt.mul(b1[SG:BK, :], bs1_1, b1[SG:BK, :])

        kdt.load(io_tensors["Ab"][row_start:row_end, block_start + 256:block_start + 384], a2)
        kdt.load(io_tensors["Bb"][block_start + 256:block_start + 384, col_start:col_end], b2)
        kdt.load(io_tensors["As"][row_start:row_end, p * 6 + 4:p * 6 + 6], as2)
        kdt.load(io_tensors["Bs"][p * 6 + 4:p * 6 + 6, col_start:col_end], bs2)

        kdt.matmul(a1, b1, c, accumulate=True)

        kdt.mul(a2[:, 0:SG], as2_0, a2[:, 0:SG])
        kdt.mul(a2[:, SG:BK], as2_1, a2[:, SG:BK])
        kdt.mul(b2[0:SG, :], bs2_0, b2[0:SG, :])
        kdt.mul(b2[SG:BK, :], bs2_1, b2[SG:BK, :])

        if p + 1 < K // 384 or K == 2560:
            kdt.load(io_tensors["Ab"][row_start:row_end, block_start + 384:block_start + 512], a0)
            kdt.load(io_tensors["Bb"][block_start + 384:block_start + 512, col_start:col_end], b0)
            kdt.load(io_tensors["As"][row_start:row_end, p * 6 + 6:p * 6 + 8], as0)
            kdt.load(io_tensors["Bs"][p * 6 + 6:p * 6 + 8, col_start:col_end], bs0)

        kdt.matmul(a2, b2, c, accumulate=True)

        if p + 1 < K // 384 or K == 2560:
            kdt.load(io_tensors["Ab"][row_start:row_end, block_start + 512:block_start + 640], a1)
            kdt.load(io_tensors["Bb"][block_start + 512:block_start + 640, col_start:col_end], b1)
            kdt.load(io_tensors["As"][row_start:row_end, p * 6 + 8:p * 6 + 10], as1)
            kdt.load(io_tensors["Bs"][p * 6 + 8:p * 6 + 10, col_start:col_end], bs1)

            kdt.mul(a0[:, 0:SG], as0_0, a0[:, 0:SG])
            kdt.mul(a0[:, SG:BK], as0_1, a0[:, SG:BK])
            kdt.mul(b0[0:SG, :], bs0_0, b0[0:SG, :])
            kdt.mul(b0[SG:BK, :], bs0_1, b0[SG:BK, :])

    if K == 2560:
        kdt.matmul(a0, b0, c, accumulate=True)
        kdt.mul(a1[:, 0:SG], as1_0, a1[:, 0:SG])
        kdt.mul(a1[:, SG:BK], as1_1, a1[:, SG:BK])
        kdt.mul(b1[0:SG, :], bs1_0, b1[0:SG, :])
        kdt.mul(b1[SG:BK, :], bs1_1, b1[SG:BK, :])
        kdt.matmul(a1, b1, c, accumulate=True)

    kdt.store(c, io_tensors["C"][row_start:row_end, col_start:col_end])


def _task4_num_jobs(task_args: dict[str, int]) -> int:
    return task_args["S_qo"] // 128


@kdt.kernel(num_jobs_calculator=_task4_num_jobs)
def _task4_kernel(task_args: Dict[str, int], io_tensors: Dict[str, kdt.Tile]):
    BQ = 128
    BK = 128
    D = task_args["D"]
    S_kv = task_args["S_kv"]
    job_id = kdt.get_job_id()
    row_start = job_id * BQ
    row_end = row_start + BQ

    q = kdt.alloc_spm((BQ, 128), dtype="float32")
    k0 = kdt.alloc_spm((BK, 128), dtype="float32")
    v0 = kdt.alloc_spm((BK, 128), dtype="float32")
    k1 = kdt.alloc_spm((BK, 128), dtype="float32")
    v1 = kdt.alloc_spm((BK, 128), dtype="float32")
    p = kdt.alloc_spm((BQ, BK), dtype="float32")
    p_next = kdt.alloc_spm((BQ, BK), dtype="float32")
    o = kdt.alloc_spm((BQ, 128), dtype="float32")
    row_sum = kdt.alloc_spm((BQ,), dtype="float32")
    denom = kdt.alloc_spm((BQ,), dtype="float32")

    kdt.load(io_tensors["Q"][row_start:row_end, 0:D], q)

    if S_kv == 128:
        kdt.load(io_tensors["K"][0:BK, 0:D], k0)
        kdt.load(io_tensors["V"][0:BK, 0:D], v0)
        kdt.matmul(q, kdt.transpose(k0, 0, 1), p)
        kdt.exp(p, p, 2.7182818284590451)
        kdt.matmul(p, v0, o, accumulate=True)
        kdt.reduce(p, 1, "sum", row_sum)
        kdt.add(denom, row_sum, denom)
    else:
        kdt.load(io_tensors["K"][0:BK, 0:D], k0)
        kdt.load(io_tensors["V"][0:BK, 0:D], v0)
        kdt.load(io_tensors["K"][BK:BK + BK, 0:D], k1)
        kdt.load(io_tensors["V"][BK:BK + BK, 0:D], v1)

        for pidx in range(0, S_kv // 256):
            kdt.matmul(q, kdt.transpose(k0, 0, 1), p)
            kdt.exp(p, p, 2.7182818284590451)
            if pidx + 1 < S_kv // 256:
                kdt.load(io_tensors["K"][pidx * 256 + 256:pidx * 256 + 384, 0:D], k0)

            kdt.matmul(q, kdt.transpose(k1, 0, 1), p_next)
            kdt.exp(p_next, p_next, 2.7182818284590451)
            if pidx + 1 < S_kv // 256:
                kdt.load(io_tensors["K"][pidx * 256 + 384:pidx * 256 + 512, 0:D], k1)

            kdt.matmul(p, v0, o, accumulate=True)
            kdt.reduce(p, 1, "sum", row_sum)
            kdt.add(denom, row_sum, denom)
            if pidx + 1 < S_kv // 256:
                kdt.load(io_tensors["V"][pidx * 256 + 256:pidx * 256 + 384, 0:D], v0)

            kdt.matmul(p_next, v1, o, accumulate=True)
            kdt.reduce(p_next, 1, "sum", row_sum)
            kdt.add(denom, row_sum, denom)
            if pidx + 1 < S_kv // 256:
                kdt.load(io_tensors["V"][pidx * 256 + 384:pidx * 256 + 512, 0:D], v1)

    denom_b = kdt.broadcast_to(kdt.unsqueeze(denom, 1), 1, 128)
    kdt.div(o, denom_b, o)
    kdt.store(o, io_tensors["O"][row_start:row_end, 0:D])


def get_kernel(task_id: int) -> kdt.KernelFunction:
    if task_id == 1:
        return _task1_kernel
    if task_id == 2:
        return _task2_kernel
    if task_id == 3:
        return _task3_kernel
    if task_id == 4:
        return _task4_kernel
    raise ValueError(f"Unsupported task_id: {task_id}")
