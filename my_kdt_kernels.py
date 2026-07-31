import kdt


# ==================== Task 1: vector add ====================

def _t1_num_jobs(task_args):
    return 1


@kdt.kernel(num_jobs_calculator=_t1_num_jobs)
def _kernel_t1(task_args, io_tensors):
    N = task_args['N']
    C = 2048
    niter = N // C

    a0 = kdt.alloc_spm((C,))
    a1 = kdt.alloc_spm((C,))
    a2 = kdt.alloc_spm((C,))
    a3 = kdt.alloc_spm((C,))
    a4 = kdt.alloc_spm((C,))
    b0 = kdt.alloc_spm((C,))
    b1 = kdt.alloc_spm((C,))
    b2 = kdt.alloc_spm((C,))
    b3 = kdt.alloc_spm((C,))
    b4 = kdt.alloc_spm((C,))
    c0 = kdt.alloc_spm((C,))
    c1 = kdt.alloc_spm((C,))
    c2 = kdt.alloc_spm((C,))
    c3 = kdt.alloc_spm((C,))
    c4 = kdt.alloc_spm((C,))

    kdt.load(io_tensors['a'][0: C], a0)
    kdt.load(io_tensors['a'][C: 2 * C], a1)
    kdt.load(io_tensors['a'][2 * C: 3 * C], a2)
    kdt.load(io_tensors['a'][3 * C: 4 * C], a3)
    kdt.load(io_tensors['b'][0: C], b0)
    kdt.load(io_tensors['b'][C: 2 * C], b1)
    kdt.load(io_tensors['b'][2 * C: 3 * C], b2)
    kdt.load(io_tensors['b'][3 * C: 4 * C], b3)

    for i in range(niter):
        if i % 5 == 0:
            if i < niter - 4:
                kdt.load(io_tensors['a'][(i + 4) * C: (i + 5) * C], a4)
                kdt.load(io_tensors['b'][(i + 4) * C: (i + 5) * C], b4)
            kdt.add(a0, b0, c0)
            kdt.store(c0, io_tensors['c'][i * C: (i + 1) * C])
        elif i % 5 == 1:
            if i < niter - 4:
                kdt.load(io_tensors['a'][(i + 4) * C: (i + 5) * C], a0)
                kdt.load(io_tensors['b'][(i + 4) * C: (i + 5) * C], b0)
            kdt.add(a1, b1, c1)
            kdt.store(c1, io_tensors['c'][i * C: (i + 1) * C])
        elif i % 5 == 2:
            if i < niter - 4:
                kdt.load(io_tensors['a'][(i + 4) * C: (i + 5) * C], a1)
                kdt.load(io_tensors['b'][(i + 4) * C: (i + 5) * C], b1)
            kdt.add(a2, b2, c2)
            kdt.store(c2, io_tensors['c'][i * C: (i + 1) * C])
        elif i % 5 == 3:
            if i < niter - 4:
                kdt.load(io_tensors['a'][(i + 4) * C: (i + 5) * C], a2)
                kdt.load(io_tensors['b'][(i + 4) * C: (i + 5) * C], b2)
            kdt.add(a3, b3, c3)
            kdt.store(c3, io_tensors['c'][i * C: (i + 1) * C])
        else:
            if i < niter - 4:
                kdt.load(io_tensors['a'][(i + 4) * C: (i + 5) * C], a3)
                kdt.load(io_tensors['b'][(i + 4) * C: (i + 5) * C], b3)
            kdt.add(a4, b4, c4)
            kdt.store(c4, io_tensors['c'][i * C: (i + 1) * C])


# ==================== Task 2: matmul ====================

def _t2_num_jobs(task_args):
    return (task_args['M'] // 128) * (task_args['N'] // 128)


@kdt.kernel(num_jobs_calculator=_t2_num_jobs)
def _kernel_t2(task_args, io_tensors):
    M = task_args['M']
    N = task_args['N']
    K = task_args['K']
    BM = 128
    BN = 128
    BK = 128
    NITER = K // BK

    A1 = kdt.alloc_spm((BM, BK))
    B1 = kdt.alloc_spm((BK, BN))
    A2 = kdt.alloc_spm((BM, BK))
    B2 = kdt.alloc_spm((BK, BN))
    C = kdt.alloc_spm((BM, BN), init_value=0)

    job_id = kdt.get_job_id()
    n_n = N // BN
    mi = job_id // n_n
    ni = job_id % n_n

    kdt.load(io_tensors['A'][mi * BM: (mi + 1) * BM, 0: BK], A1)
    kdt.load(io_tensors['B'][0: BK, ni * BN: (ni + 1) * BN], B1)

    for ki in range(NITER):
        if ki % 2 == 0:
            kdt.matmul(A1, B1, C, True)
            if ki < NITER - 1:
                kdt.load(io_tensors['A'][mi * BM: (mi + 1) * BM, (ki + 1) * BK: (ki + 2) * BK], A2)
                kdt.load(io_tensors['B'][(ki + 1) * BK: (ki + 2) * BK, ni * BN: (ni + 1) * BN], B2)
        else:
            kdt.matmul(A2, B2, C, True)
            if ki < NITER - 1:
                kdt.load(io_tensors['A'][mi * BM: (mi + 1) * BM, (ki + 1) * BK: (ki + 2) * BK], A1)
                kdt.load(io_tensors['B'][(ki + 1) * BK: (ki + 2) * BK, ni * BN: (ni + 1) * BN], B1)

    kdt.store(C, io_tensors['C'][mi * BM: (mi + 1) * BM, ni * BN: (ni + 1) * BN])


# ==================== Task 3: scaled matmul ====================

def _t3_num_jobs(task_args):
    return (task_args['M'] // 128) * (task_args['N'] // 128)


@kdt.kernel(num_jobs_calculator=_t3_num_jobs)
def _kernel_t3(task_args, io_tensors):
    M = task_args['M']
    N = task_args['N']
    K = task_args['K']
    BM = 128
    BN = 128
    BK = 128
    NITER = K // BK

    Ab1 = kdt.alloc_spm((BM, BK))
    Bb1 = kdt.alloc_spm((BK, BN))
    Ab2 = kdt.alloc_spm((BM, BK))
    Bb2 = kdt.alloc_spm((BK, BN))
    As1 = kdt.alloc_spm((BM, 2))
    Bs1 = kdt.alloc_spm((2, BN))
    As2 = kdt.alloc_spm((BM, 2))
    Bs2 = kdt.alloc_spm((2, BN))
    S0 = kdt.alloc_spm((BM, BN))
    S1 = kdt.alloc_spm((BM, BN))
    scale_tile = kdt.alloc_spm((BM, BN))
    C = kdt.alloc_spm((BM, BN), init_value=0)

    job_id = kdt.get_job_id()
    n_n = N // BN
    mi = job_id // n_n
    ni = job_id % n_n

    kdt.load(io_tensors['Ab'][mi * BM: (mi + 1) * BM, 0: BK], Ab1)
    kdt.load(io_tensors['Bb'][0: BK, ni * BN: (ni + 1) * BN], Bb1)
    kdt.load(io_tensors['As'][mi * BM: (mi + 1) * BM, 0: 2], As1)
    kdt.load(io_tensors['Bs'][0: 2, ni * BN: (ni + 1) * BN], Bs1)

    for ki in range(NITER):
        if ki % 2 == 0:
            kdt.matmul(Ab1[:, 0: 64], Bb1[0: 64, :], S0)
            if ki < NITER - 1:
                kdt.load(io_tensors['Ab'][mi * BM: (mi + 1) * BM, (ki + 1) * BK: (ki + 2) * BK], Ab2)
                kdt.load(io_tensors['Bb'][(ki + 1) * BK: (ki + 2) * BK, ni * BN: (ni + 1) * BN], Bb2)
                kdt.load(io_tensors['As'][mi * BM: (mi + 1) * BM, 2 * ki + 2: 2 * ki + 4], As2)
                kdt.load(io_tensors['Bs'][2 * ki + 2: 2 * ki + 4, ni * BN: (ni + 1) * BN], Bs2)
            kdt.matmul(Ab1[:, 64: 128], Bb1[64: 128, :], S1)
            kdt.mul(kdt.broadcast_to(kdt.unsqueeze(As1[:, 0], 1), 1, BN),
                    kdt.broadcast_to(kdt.unsqueeze(Bs1[0, :], 0), 0, BM), scale_tile)
            kdt.fma(S0, scale_tile, C, C)
            kdt.mul(kdt.broadcast_to(kdt.unsqueeze(As1[:, 1], 1), 1, BN),
                    kdt.broadcast_to(kdt.unsqueeze(Bs1[1, :], 0), 0, BM), scale_tile)
            kdt.fma(S1, scale_tile, C, C)
        else:
            kdt.matmul(Ab2[:, 0: 64], Bb2[0: 64, :], S0)
            if ki < NITER - 1:
                kdt.load(io_tensors['Ab'][mi * BM: (mi + 1) * BM, (ki + 1) * BK: (ki + 2) * BK], Ab1)
                kdt.load(io_tensors['Bb'][(ki + 1) * BK: (ki + 2) * BK, ni * BN: (ni + 1) * BN], Bb1)
                kdt.load(io_tensors['As'][mi * BM: (mi + 1) * BM, 2 * ki + 2: 2 * ki + 4], As1)
                kdt.load(io_tensors['Bs'][2 * ki + 2: 2 * ki + 4, ni * BN: (ni + 1) * BN], Bs1)
            kdt.matmul(Ab2[:, 64: 128], Bb2[64: 128, :], S1)
            kdt.mul(kdt.broadcast_to(kdt.unsqueeze(As2[:, 0], 1), 1, BN),
                    kdt.broadcast_to(kdt.unsqueeze(Bs2[0, :], 0), 0, BM), scale_tile)
            kdt.fma(S0, scale_tile, C, C)
            kdt.mul(kdt.broadcast_to(kdt.unsqueeze(As2[:, 1], 1), 1, BN),
                    kdt.broadcast_to(kdt.unsqueeze(Bs2[1, :], 0), 0, BM), scale_tile)
            kdt.fma(S1, scale_tile, C, C)

    kdt.store(C, io_tensors['C'][mi * BM: (mi + 1) * BM, ni * BN: (ni + 1) * BN])


# ==================== Task 4: flash attention ====================

def _t4_num_jobs(task_args):
    return task_args['S_qo'] // 128


@kdt.kernel(num_jobs_calculator=_t4_num_jobs)
def _kernel_t4(task_args, io_tensors):
    S_qo = task_args['S_qo']
    S_kv = task_args['S_kv']
    D = task_args['D']
    BM = 128
    BN = 128
    NB = S_kv // BN

    Q_tile = kdt.alloc_spm((BM, D))
    O_tile = kdt.alloc_spm((BM, D), init_value=0)
    S0 = kdt.alloc_spm((BM, BN))
    S1 = kdt.alloc_spm((BM, BN))
    K1 = kdt.alloc_spm((BN, D))
    V1 = kdt.alloc_spm((BN, D))
    K2 = kdt.alloc_spm((BN, D))
    V2 = kdt.alloc_spm((BN, D))

    m_tile = kdt.alloc_spm((BM,), init_value=-1e38)
    l_tile = kdt.alloc_spm((BM,), init_value=0)
    d1_tile = kdt.alloc_spm((BM,))
    rescale_tile = kdt.alloc_spm((BM,))
    rm_tile = kdt.alloc_spm((BM,))
    rs_tile = kdt.alloc_spm((BM,))
    z_tile = kdt.alloc_spm((BM,), init_value=0)

    job_id = kdt.get_job_id()

    kdt.load(io_tensors['Q'][job_id * BM: (job_id + 1) * BM, :], Q_tile)
    kdt.load(io_tensors['K'][0: BN, :], K1)
    kdt.load(io_tensors['V'][0: BN, :], V1)
    if NB >= 2:
        kdt.load(io_tensors['K'][BN: 2 * BN, :], K2)
        kdt.load(io_tensors['V'][BN: 2 * BN, :], V2)

    for bi in range(NB // 2):
        if bi % 2 == 0:
            kdt.matmul(Q_tile, kdt.transpose(K1, 0, 1), S0)
            kdt.matmul(Q_tile, kdt.transpose(K2, 0, 1), S1)
            kdt.reduce(S0, 1, 'max', rm_tile)
            kdt.sub(m_tile, rm_tile, d1_tile)
            kdt.min(d1_tile, z_tile, d1_tile)
            kdt.exp(d1_tile, rescale_tile, 2.7182818284590451)
            kdt.max(m_tile, rm_tile, m_tile)
            kdt.mul(O_tile, kdt.broadcast_to(kdt.unsqueeze(rescale_tile, 1), 1, D), O_tile)
            kdt.sub(S0, kdt.broadcast_to(kdt.unsqueeze(m_tile, 1), 1, BN), S0)
            kdt.exp(S0, S0, 2.7182818284590451)
            kdt.mul(l_tile, rescale_tile, l_tile)
            kdt.reduce(S0, 1, 'sum', rs_tile)
            kdt.add(l_tile, rs_tile, l_tile)
            kdt.matmul(S0, V1, O_tile, True)
            kdt.reduce(S1, 1, 'max', rm_tile)
            kdt.sub(m_tile, rm_tile, d1_tile)
            kdt.min(d1_tile, z_tile, d1_tile)
            kdt.exp(d1_tile, rescale_tile, 2.7182818284590451)
            kdt.max(m_tile, rm_tile, m_tile)
            kdt.sub(S1, kdt.broadcast_to(kdt.unsqueeze(m_tile, 1), 1, BN), S1)
            kdt.exp(S1, S1, 2.7182818284590451)
            kdt.mul(l_tile, rescale_tile, l_tile)
            kdt.reduce(S1, 1, 'sum', rs_tile)
            kdt.add(l_tile, rs_tile, l_tile)
            kdt.mul(O_tile, kdt.broadcast_to(kdt.unsqueeze(rescale_tile, 1), 1, D), O_tile)
            kdt.matmul(S1, V2, O_tile, True)
            if bi < NB // 2 - 1:
                kdt.load(io_tensors['K'][(2 * bi + 2) * BN: (2 * bi + 3) * BN, :], K1)
                kdt.load(io_tensors['V'][(2 * bi + 2) * BN: (2 * bi + 3) * BN, :], V1)
                kdt.load(io_tensors['K'][(2 * bi + 3) * BN: (2 * bi + 4) * BN, :], K2)
                kdt.load(io_tensors['V'][(2 * bi + 3) * BN: (2 * bi + 4) * BN, :], V2)
        else:
            kdt.matmul(Q_tile, kdt.transpose(K2, 0, 1), S0)
            kdt.matmul(Q_tile, kdt.transpose(K1, 0, 1), S1)
            kdt.reduce(S0, 1, 'max', rm_tile)
            kdt.sub(m_tile, rm_tile, d1_tile)
            kdt.min(d1_tile, z_tile, d1_tile)
            kdt.exp(d1_tile, rescale_tile, 2.7182818284590451)
            kdt.max(m_tile, rm_tile, m_tile)
            kdt.mul(O_tile, kdt.broadcast_to(kdt.unsqueeze(rescale_tile, 1), 1, D), O_tile)
            kdt.sub(S0, kdt.broadcast_to(kdt.unsqueeze(m_tile, 1), 1, BN), S0)
            kdt.exp(S0, S0, 2.7182818284590451)
            kdt.mul(l_tile, rescale_tile, l_tile)
            kdt.reduce(S0, 1, 'sum', rs_tile)
            kdt.add(l_tile, rs_tile, l_tile)
            kdt.matmul(S0, V2, O_tile, True)
            kdt.reduce(S1, 1, 'max', rm_tile)
            kdt.sub(m_tile, rm_tile, d1_tile)
            kdt.min(d1_tile, z_tile, d1_tile)
            kdt.exp(d1_tile, rescale_tile, 2.7182818284590451)
            kdt.max(m_tile, rm_tile, m_tile)
            kdt.sub(S1, kdt.broadcast_to(kdt.unsqueeze(m_tile, 1), 1, BN), S1)
            kdt.exp(S1, S1, 2.7182818284590451)
            kdt.mul(l_tile, rescale_tile, l_tile)
            kdt.reduce(S1, 1, 'sum', rs_tile)
            kdt.add(l_tile, rs_tile, l_tile)
            kdt.mul(O_tile, kdt.broadcast_to(kdt.unsqueeze(rescale_tile, 1), 1, D), O_tile)
            kdt.matmul(S1, V1, O_tile, True)
            if bi < NB // 2 - 1:
                kdt.load(io_tensors['K'][(2 * bi + 2) * BN: (2 * bi + 3) * BN, :], K2)
                kdt.load(io_tensors['V'][(2 * bi + 2) * BN: (2 * bi + 3) * BN, :], V2)
                kdt.load(io_tensors['K'][(2 * bi + 3) * BN: (2 * bi + 4) * BN, :], K1)
                kdt.load(io_tensors['V'][(2 * bi + 3) * BN: (2 * bi + 4) * BN, :], V1)

    if NB % 2 == 1:
        kdt.matmul(Q_tile, kdt.transpose(K1, 0, 1), S0)
        kdt.reduce(S0, 1, 'max', rm_tile)
        kdt.sub(m_tile, rm_tile, d1_tile)
        kdt.min(d1_tile, z_tile, d1_tile)
        kdt.exp(d1_tile, rescale_tile, 2.7182818284590451)
        kdt.max(m_tile, rm_tile, m_tile)
        kdt.mul(O_tile, kdt.broadcast_to(kdt.unsqueeze(rescale_tile, 1), 1, D), O_tile)
        kdt.sub(S0, kdt.broadcast_to(kdt.unsqueeze(m_tile, 1), 1, BN), S0)
        kdt.exp(S0, S0, 2.7182818284590451)
        kdt.mul(l_tile, rescale_tile, l_tile)
        kdt.reduce(S0, 1, 'sum', rs_tile)
        kdt.add(l_tile, rs_tile, l_tile)
        kdt.matmul(S0, V1, O_tile, True)

    kdt.div(O_tile, kdt.broadcast_to(kdt.unsqueeze(l_tile, 1), 1, D), O_tile)
    kdt.store(O_tile, io_tensors['O'][job_id * BM: (job_id + 1) * BM, :])


def get_kernel(task_id: int):
    if task_id == 1:
        return _kernel_t1
    elif task_id == 2:
        return _kernel_t2
    elif task_id == 3:
        return _kernel_t3
    elif task_id == 4:
        return _kernel_t4
    else:
        raise ValueError(f"Unknown task id: {task_id}")
