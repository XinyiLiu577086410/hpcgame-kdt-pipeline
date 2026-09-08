import math

import kdt


# ======================================================================
# Task 1: Vector add
# ======================================================================


def _calc_jobs_vector_add(task_args):
    return 1


@kdt.kernel(num_jobs_calculator=_calc_jobs_vector_add)
def vector_add_kernel(task_args, io_tensors):
    N = task_args['N']
    B = 2048
    NSTG = 5
    num_blocks = N // B

    a_buf = kdt.alloc_spm((NSTG, B), dtype='float32')
    b_buf = kdt.alloc_spm((NSTG, B), dtype='float32')
    c_buf = kdt.alloc_spm((NSTG, B), dtype='float32')

    for s in range(NSTG - 1):
        kdt.load(io_tensors['a'][s * B:(s + 1) * B], a_buf[s])
        kdt.load(io_tensors['b'][s * B:(s + 1) * B], b_buf[s])

    for k in range(num_blocks):
        nxt = k + (NSTG - 1)
        if nxt < num_blocks:
            kdt.load(io_tensors['a'][nxt * B:(nxt + 1) * B], a_buf[nxt % NSTG])
            kdt.load(io_tensors['b'][nxt * B:(nxt + 1) * B], b_buf[nxt % NSTG])
        kdt.add(a_buf[k % NSTG], b_buf[k % NSTG], c_buf[k % NSTG])
        kdt.store(c_buf[k % NSTG], io_tensors['c'][k * B:(k + 1) * B])


# ======================================================================
# Task 2: Matrix multiplication
# ======================================================================


def _calc_jobs_matmul(task_args):
    M = task_args['M']
    N = task_args['N']
    return (M // 128) * (N // 128)


@kdt.kernel(num_jobs_calculator=_calc_jobs_matmul)
def matmul_kernel(task_args, io_tensors):
    M = task_args['M']
    N = task_args['N']
    K = task_args['K']
    BM = 128
    BN = 128
    BK = 64
    NSTG = 4
    job = kdt.get_job_id()
    num_blk_m = M // BM
    num_blk_n = N // BN
    m_blk = job // num_blk_n
    n_blk = job % num_blk_n
    m0 = m_blk * BM
    n0 = n_blk * BN

    A = io_tensors['A'][m0:m0 + BM, :]
    B = io_tensors['B'][:, n0:n0 + BN]

    a_buf = kdt.alloc_spm((NSTG, BM, BK), dtype='float32')
    b_buf = kdt.alloc_spm((NSTG, BK, BN), dtype='float32')
    c_acc = kdt.alloc_spm((BM, BN), dtype='float32', init_value=0.0)

    num_k = K // BK
    for s in range(NSTG - 1):
        kdt.load(A[:, s * BK:(s + 1) * BK], a_buf[s])
        kdt.load(B[s * BK:(s + 1) * BK, :], b_buf[s])

    for k in range(num_k):
        nxt = k + (NSTG - 1)
        if nxt < num_k:
            kdt.load(A[:, nxt * BK:(nxt + 1) * BK], a_buf[nxt % NSTG])
            kdt.load(B[nxt * BK:(nxt + 1) * BK, :], b_buf[nxt % NSTG])
        kdt.matmul(a_buf[k % NSTG], b_buf[k % NSTG], c_acc, accumulate=True)

    kdt.store(c_acc, io_tensors['C'][m0:m0 + BM, n0:n0 + BN])


# ======================================================================
# Task 3: Matrix multiplication with fine-grained scale
# ======================================================================


def _calc_jobs_matmul_scale(task_args):
    M = task_args['M']
    N = task_args['N']
    return (M // 128) * (N // 128)


@kdt.kernel(num_jobs_calculator=_calc_jobs_matmul_scale)
def matmul_scale_kernel(task_args, io_tensors):
    M = task_args['M']
    N = task_args['N']
    K = task_args['K']
    BM = 128
    BN = 128
    BK = 64
    SGR = 64
    NSTG = 6
    job = kdt.get_job_id()
    num_blk_m = M // BM
    num_blk_n = N // BN
    m_blk = job // num_blk_n
    n_blk = job % num_blk_n
    m0 = m_blk * BM
    n0 = n_blk * BN

    Ab = io_tensors['Ab'][m0:m0 + BM, :]
    As_g = io_tensors['As'][m0:m0 + BM, :]
    Bb = io_tensors['Bb'][:, n0:n0 + BN]
    Bs_g = io_tensors['Bs'][:, n0:n0 + BN]

    ab_buf = kdt.alloc_spm((NSTG, BM, BK), dtype='float32')
    as_buf = kdt.alloc_spm((NSTG, BM, BK // SGR), dtype='float32')
    bb_buf = kdt.alloc_spm((NSTG, BK, BN), dtype='float32')
    bs_buf = kdt.alloc_spm((NSTG, BK // SGR, BN), dtype='float32')
    c_acc = kdt.alloc_spm((BM, BN), dtype='float32', init_value=0.0)

    num_k = K // BK
    NUM_PRO = NSTG - 1
    # prologue: load raw chunks 0..NUM_PRO-1, scale chunks 0 and 1
    for s in range(NUM_PRO):
        if s < num_k:
            kdt.load(Ab[:, s * BK:(s + 1) * BK], ab_buf[s])
            kdt.load(As_g[:, (s * BK) // SGR:(s * BK + BK) // SGR], as_buf[s])
            kdt.load(Bb[s * BK:(s + 1) * BK, :], bb_buf[s])
            kdt.load(Bs_g[(s * BK) // SGR:(s * BK + BK) // SGR, :], bs_buf[s])
    if num_k > 0:
        as_bc0 = kdt.broadcast_to(as_buf[0], 1, BK)
        bs_bc0 = kdt.broadcast_to(bs_buf[0], 0, BK)
        kdt.mul(ab_buf[0], as_bc0, ab_buf[0])
        kdt.mul(bb_buf[0], bs_bc0, bb_buf[0])
    if num_k > 1:
        as_bc1 = kdt.broadcast_to(as_buf[1], 1, BK)
        bs_bc1 = kdt.broadcast_to(bs_buf[1], 0, BK)
        kdt.mul(ab_buf[1], as_bc1, ab_buf[1])
        kdt.mul(bb_buf[1], bs_bc1, bb_buf[1])

    for k in range(num_k):
        kdt.matmul(ab_buf[k % NSTG], bb_buf[k % NSTG], c_acc, accumulate=True)
        nxt = k + NUM_PRO
        if nxt < num_k:
            kdt.load(Ab[:, nxt * BK:(nxt + 1) * BK], ab_buf[nxt % NSTG])
            kdt.load(As_g[:, (nxt * BK) // SGR:(nxt * BK + BK) // SGR], as_buf[nxt % NSTG])
            kdt.load(Bb[nxt * BK:(nxt + 1) * BK, :], bb_buf[nxt % NSTG])
            kdt.load(Bs_g[(nxt * BK) // SGR:(nxt * BK + BK) // SGR, :], bs_buf[nxt % NSTG])
        sc = k + 2
        if sc < num_k:
            as_bc = kdt.broadcast_to(as_buf[sc % NSTG], 1, BK)
            bs_bc = kdt.broadcast_to(bs_buf[sc % NSTG], 0, BK)
            kdt.mul(ab_buf[sc % NSTG], as_bc, ab_buf[sc % NSTG])
            kdt.mul(bb_buf[sc % NSTG], bs_bc, bb_buf[sc % NSTG])

    kdt.store(c_acc, io_tensors['C'][m0:m0 + BM, n0:n0 + BN])


# ======================================================================
# Task 4: Flash Attention
# ======================================================================


def _calc_jobs_attn(task_args):
    S_qo = task_args['S_qo']
    return S_qo // 128


@kdt.kernel(num_jobs_calculator=_calc_jobs_attn)
def flash_attn_kernel(task_args, io_tensors):
    S_qo = task_args['S_qo']
    S_kv = task_args['S_kv']
    D = task_args['D']
    BQ = 128
    BK = 128
    E = 2.7182818284590451
    job = kdt.get_job_id()
    q0 = job * BQ

    Q = io_tensors['Q'][q0:q0 + BQ, :]
    KV_chunks = S_kv // BK

    q_buf = kdt.alloc_spm((BQ, D), dtype='float32')
    o_buf = kdt.alloc_spm((BQ, D), dtype='float32', init_value=0.0)
    m_buf = kdt.alloc_spm((BQ,), dtype='float32', init_value=-1.0e38)
    l_buf = kdt.alloc_spm((BQ,), dtype='float32', init_value=0.0)

    k_buf = kdt.alloc_spm((2, BK, D), dtype='float32')
    v_buf = kdt.alloc_spm((2, BK, D), dtype='float32')

    s_buf = kdt.alloc_spm((2, BQ, BK), dtype='float32')
    p_buf = kdt.alloc_spm((BQ, BK), dtype='float32')

    m_new = kdt.alloc_spm((BQ,), dtype='float32')
    rowmax = kdt.alloc_spm((BQ,), dtype='float32')
    alpha = kdt.alloc_spm((BQ,), dtype='float32')
    rsum = kdt.alloc_spm((BQ,), dtype='float32')
    tmp = kdt.alloc_spm((BQ,), dtype='float32')

    kdt.load(Q, q_buf)
    kdt.load(io_tensors['K'][0:BK, :], k_buf[0])
    kdt.load(io_tensors['V'][0:BK, :], v_buf[0])
    kdt.matmul(q_buf, kdt.transpose(k_buf[0], 0, 1), s_buf[0], accumulate=False)
    if KV_chunks > 1:
        kdt.load(io_tensors['K'][BK:2 * BK, :], k_buf[1])
        kdt.load(io_tensors['V'][BK:2 * BK, :], v_buf[1])

    for c in range(KV_chunks):
        nxt = c + 1
        if nxt < KV_chunks:
            kdt.matmul(q_buf, kdt.transpose(k_buf[nxt % 2], 0, 1), s_buf[nxt % 2], accumulate=False)
        pre = c + 2
        if pre < KV_chunks:
            kdt.load(io_tensors['K'][pre * BK:(pre + 1) * BK, :], k_buf[pre % 2])

        kdt.copy(s_buf[c % 2], p_buf)
        kdt.reduce(p_buf, 1, 'max', rowmax)
        kdt.max(m_buf, rowmax, m_new)

        m_bc = kdt.broadcast_to(kdt.unsqueeze(m_new, 1), 1, BK)
        kdt.sub(p_buf, m_bc, p_buf)
        kdt.exp(p_buf, p_buf, E)

        kdt.sub(m_buf, m_new, tmp)
        kdt.exp(tmp, alpha, E)

        alpha_bc = kdt.broadcast_to(kdt.unsqueeze(alpha, 1), 1, D)
        kdt.mul(o_buf, alpha_bc, o_buf)
        kdt.matmul(p_buf, v_buf[c % 2], o_buf, accumulate=True)

        kdt.reduce(p_buf, 1, 'sum', rsum)
        kdt.fma(l_buf, alpha, rsum, l_buf)
        kdt.copy(m_new, m_buf)

        if pre < KV_chunks:
            kdt.load(io_tensors['V'][pre * BK:(pre + 1) * BK, :], v_buf[pre % 2])

    l_bc = kdt.broadcast_to(kdt.unsqueeze(l_buf, 1), 1, D)
    kdt.div(o_buf, l_bc, o_buf)
    kdt.store(o_buf, io_tensors['O'][q0:q0 + BQ, :])


def get_kernel(task_id: int) -> kdt.KernelFunction:
    if task_id == 1:
        return vector_add_kernel
    elif task_id == 2:
        return matmul_kernel
    elif task_id == 3:
        return matmul_scale_kernel
    elif task_id == 4:
        return flash_attn_kernel
    else:
        raise ValueError(f"Invalid task_id: {task_id}")
