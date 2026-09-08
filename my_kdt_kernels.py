import kdt


# =====================================================================
# Task 1: vector add
# =====================================================================

def _num_jobs_task1(task_args):
    return 1


@kdt.kernel(num_jobs_calculator=_num_jobs_task1)
def task1_kernel(task_args, io_tensors):
    BLOCK = 2048
    DEPTH = 4
    N = task_args['N']
    num_chunks = N // BLOCK
    a_global = io_tensors['a']
    b_global = io_tensors['b']
    c_global = io_tensors['c']

    a_buf = kdt.alloc_spm((DEPTH, BLOCK))
    b_buf = kdt.alloc_spm((DEPTH, BLOCK))
    c_buf = kdt.alloc_spm((DEPTH, BLOCK))

    for p in range(DEPTH - 1):
        kdt.load(a_global[p * BLOCK:(p + 1) * BLOCK], a_buf[p])
        kdt.load(b_global[p * BLOCK:(p + 1) * BLOCK], b_buf[p])

    for i in range(DEPTH - 1, num_chunks):
        kdt.load(a_global[i * BLOCK:(i + 1) * BLOCK], a_buf[i % DEPTH])
        kdt.load(b_global[i * BLOCK:(i + 1) * BLOCK], b_buf[i % DEPTH])
        j = i - DEPTH + 1
        kdt.add(a_buf[j % DEPTH], b_buf[j % DEPTH], c_buf[j % DEPTH])
        kdt.store(c_buf[j % DEPTH], c_global[j * BLOCK:(j + 1) * BLOCK])

    for i in range(num_chunks - DEPTH + 1, num_chunks):
        kdt.add(a_buf[i % DEPTH], b_buf[i % DEPTH], c_buf[i % DEPTH])
        kdt.store(c_buf[i % DEPTH], c_global[i * BLOCK:(i + 1) * BLOCK])


# =====================================================================
# Task 2: matmul
# =====================================================================

def _num_jobs_task2(task_args):
    BM = 128
    BN = 128
    return (task_args['M'] // BM) * (task_args['N'] // BN)


@kdt.kernel(num_jobs_calculator=_num_jobs_task2)
def task2_kernel(task_args, io_tensors):
    BM = 128
    BN = 128
    BK = 128
    M = task_args['M']
    N = task_args['N']
    K = task_args['K']
    num_n = N // BN
    num_k = K // BK

    job_id = kdt.get_job_id()
    m_idx = job_id // num_n
    n_idx = job_id % num_n
    m0 = m_idx * BM
    n0 = n_idx * BN

    a_global = io_tensors['A']
    b_global = io_tensors['B']
    c_global = io_tensors['C']

    a_buf = kdt.alloc_spm((2, BM, BK))
    b_buf = kdt.alloc_spm((2, BK, BN))
    c_tile = kdt.alloc_spm((BM, BN))
    kdt.fill(c_tile, 0.0)

    kdt.load(a_global[m0:m0 + BM, 0:BK], a_buf[0])
    kdt.load(b_global[0:BK, n0:n0 + BN], b_buf[0])

    for k in range(1, num_k):
        kdt.load(a_global[m0:m0 + BM, k * BK:(k + 1) * BK], a_buf[k % 2])
        kdt.load(b_global[k * BK:(k + 1) * BK, n0:n0 + BN], b_buf[k % 2])
        kdt.matmul(a_buf[(k - 1) % 2], b_buf[(k - 1) % 2], c_tile, accumulate=True)

    kdt.matmul(a_buf[(num_k - 1) % 2], b_buf[(num_k - 1) % 2], c_tile, accumulate=True)
    kdt.store(c_tile, c_global[m0:m0 + BM, n0:n0 + BN])


# =====================================================================
# Task 3: matmul with fine-grained scale
# =====================================================================

def _num_jobs_task3(task_args):
    BM = 128
    BN = 128
    return (task_args['M'] // BM) * (task_args['N'] // BN)


@kdt.kernel(num_jobs_calculator=_num_jobs_task3)
def task3_kernel(task_args, io_tensors):
    BM = 128
    BN = 128
    BK = 64
    DEPTH = 6
    DEQ_DELAY = 2
    MM_DELAY = 4
    M = task_args['M']
    N = task_args['N']
    K = task_args['K']
    num_n = N // BN
    num_k = K // BK

    job_id = kdt.get_job_id()
    m_idx = job_id // num_n
    n_idx = job_id % num_n
    m0 = m_idx * BM
    n0 = n_idx * BN

    ab_global = io_tensors['Ab']
    as_global = io_tensors['As']
    bb_global = io_tensors['Bb']
    bs_global = io_tensors['Bs']
    c_global = io_tensors['C']

    a_buf = kdt.alloc_spm((DEPTH, BM, BK))
    b_buf = kdt.alloc_spm((DEPTH, BK, BN))
    as_tile = kdt.alloc_spm((BM, num_k))
    bs_tile = kdt.alloc_spm((num_k, BN))
    c_tile = kdt.alloc_spm((BM, BN))
    kdt.fill(c_tile, 0.0)

    kdt.load(as_global[m0:m0 + BM, :], as_tile)
    kdt.load(bs_global[:, n0:n0 + BN], bs_tile)

    for t in range(num_k + MM_DELAY):
        if t < num_k:
            kdt.load(ab_global[m0:m0 + BM, t * BK:(t + 1) * BK], a_buf[t % DEPTH])
            kdt.load(bb_global[t * BK:(t + 1) * BK, n0:n0 + BN], b_buf[t % DEPTH])
        if t >= DEQ_DELAY and t < num_k + DEQ_DELAY:
            d = t - DEQ_DELAY
            kdt.mul(a_buf[d % DEPTH], kdt.broadcast_to(as_tile[:, d:d + 1], 1, BK), a_buf[d % DEPTH])
            kdt.mul(b_buf[d % DEPTH], kdt.broadcast_to(bs_tile[d:d + 1, :], 0, BK), b_buf[d % DEPTH])
        if t >= MM_DELAY:
            mm = t - MM_DELAY
            kdt.matmul(a_buf[mm % DEPTH], b_buf[mm % DEPTH], c_tile, accumulate=True)

    kdt.store(c_tile, c_global[m0:m0 + BM, n0:n0 + BN])


# =====================================================================
# Task 4: flash attention
# =====================================================================

def _num_jobs_task4(task_args):
    BQ = 128
    return task_args['S_qo'] // BQ


@kdt.kernel(num_jobs_calculator=_num_jobs_task4)
def task4_kernel(task_args, io_tensors):
    BQ = 128
    BK = 128
    DEPTH = 2
    S_DELAY = 1
    P_DELAY = 2
    E = 2.7182818284590451
    S_qo = task_args['S_qo']
    S_kv = task_args['S_kv']
    D = task_args['D']
    num_blocks = S_kv // BK

    job_id = kdt.get_job_id()
    m0 = job_id * BQ

    q_global = io_tensors['Q']
    k_global = io_tensors['K']
    v_global = io_tensors['V']
    o_global = io_tensors['O']

    q_tile = kdt.alloc_spm((BQ, D))
    o_acc = kdt.alloc_spm((BQ, D))
    s_buf = kdt.alloc_spm((DEPTH, BQ, BK))
    k_buf = kdt.alloc_spm((DEPTH, BK, D))
    v_buf = kdt.alloc_spm((DEPTH, BK, D))
    m_old = kdt.alloc_spm((BQ,))
    m_new = kdt.alloc_spm((BQ,))
    l_old = kdt.alloc_spm((BQ,))
    l_new = kdt.alloc_spm((BQ,))
    rowred = kdt.alloc_spm((BQ,))
    tmp1 = kdt.alloc_spm((BQ,))
    alpha = kdt.alloc_spm((BQ,))

    kdt.load(q_global[m0:m0 + BQ, :], q_tile)
    kdt.fill(o_acc, 0.0)
    kdt.fill(m_old, -1e30)
    kdt.fill(l_old, 0.0)

    for t in range(num_blocks + P_DELAY):
        if t >= S_DELAY and t < num_blocks + S_DELAY:
            si = t - S_DELAY
            kdt.matmul(q_tile, kdt.transpose(k_buf[si % DEPTH], 0, 1), s_buf[si % DEPTH])
        if t >= P_DELAY and t < num_blocks + P_DELAY:
            b = t - P_DELAY
            s = s_buf[b % DEPTH]
            v = v_buf[b % DEPTH]
            kdt.reduce(s, 1, 'max', rowred)
            kdt.max(m_old, rowred, m_new)
            kdt.sub(m_old, m_new, tmp1)
            kdt.exp(tmp1, alpha, E)
            kdt.copy(m_new, m_old)
            kdt.sub(s, kdt.broadcast_to(kdt.unsqueeze(m_new, 1), 1, BK), s)
            kdt.exp(s, s, E)
            kdt.reduce(s, 1, 'sum', rowred)
            kdt.mul(alpha, l_old, tmp1)
            kdt.add(tmp1, rowred, l_new)
            kdt.copy(l_new, l_old)
            kdt.mul(o_acc, kdt.broadcast_to(kdt.unsqueeze(alpha, 1), 1, D), o_acc)
            kdt.matmul(s, v, o_acc, accumulate=True)
        if t < num_blocks:
            kdt.load(k_global[t * BK:(t + 1) * BK, :], k_buf[t % DEPTH])
            kdt.load(v_global[t * BK:(t + 1) * BK, :], v_buf[t % DEPTH])

    kdt.div(o_acc, kdt.broadcast_to(kdt.unsqueeze(l_old, 1), 1, D), o_acc)
    kdt.store(o_acc, o_global[m0:m0 + BQ, :])


# =====================================================================
# Entry point
# =====================================================================

def get_kernel(task_id: int) -> kdt.KernelFunction:
    if task_id == 1:
        return task1_kernel
    if task_id == 2:
        return task2_kernel
    if task_id == 3:
        return task3_kernel
    if task_id == 4:
        return task4_kernel
    raise ValueError(f"Unknown task_id: {task_id}")
