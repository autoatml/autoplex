import numpy as np
from scipy.sparse.linalg import LinearOperator, svds
from quippy import descriptors
from autoplex.utilities import flatten
from multiprocessing import Pool


def parallel_calc_descriptor_vec(atom, selected_descriptor):

    desc_object = descriptors.Descriptor(selected_descriptor)
    atom.info["descriptor_vec"] = desc_object.calc(atom)['data']

    return atom


def cur_select(atoms, selected_descriptor, kernel_exp, select_nums, stochastic=True):

    if isinstance(atoms[0], list):
        print('flattening')
        fatoms = flatten(atoms, recursive=True)
    else:
        fatoms = atoms
    
    with Pool() as pool:
        ats = pool.starmap(parallel_calc_descriptor_vec, [(atom, selected_descriptor) for atom in fatoms])

    if isinstance(ats, list) & (len(ats) != 0):  #waiting until all soap vectors are calculated

        at_descs = np.array([at.info["descriptor_vec"] for at in ats]).T
        if kernel_exp > 0.0:
            m = np.matmul((np.squeeze(at_descs)).T,
                        np.squeeze(at_descs)) ** kernel_exp
        else:
            m = at_descs

        def descriptor_svd(at_descs, num, do_vectors='vh'):
            def mv(v):
                return np.dot(at_descs, v)

            def rmv(v):
                return np.dot(at_descs.T, v)

            A = LinearOperator(at_descs.shape, matvec=mv,
                            rmatvec=rmv, matmat=mv)
            return svds(A, k=num, return_singular_vectors=do_vectors)

        (_, _, vt) = descriptor_svd(
            m, min(max(1, int(select_nums / 2)), min(m.shape) - 1))
        c_scores = np.sum(vt ** 2, axis=0) / vt.shape[0]
        if stochastic:
            selected = sorted(np.random.choice(
                range(len(ats)), size=select_nums, replace=False, p=c_scores))
        else:
            selected = sorted(np.argsort(c_scores)[-select_nums:])

        selected_atoms = [ats[i] for i in selected]
        
        for at in selected_atoms:
            del at.info["descriptor_vec"]
    
        return selected_atoms
    