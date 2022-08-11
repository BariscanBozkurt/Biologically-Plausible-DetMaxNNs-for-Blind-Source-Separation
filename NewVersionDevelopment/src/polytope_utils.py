####### POLYTOPE OPERATIONS UTILITIES ##############
import numpy as np
import itertools
import pypoman

def generate_practical_polytope(dim, antisparse_dims, nonnegative_dims, relative_sparse_dims_list):
    A = []
    b = []
    for j in antisparse_dims:
        row1 = [0 for _ in range(dim)]
        row2 = row1.copy()
        row1[j] = 1
        A.append(row1)
        b.append(1)
        row2[j] = -1
        A.append(row2)
        b.append(1)

    for j in nonnegative_dims:
        row1 = [0 for _ in range(dim)]
        row2 = row1.copy()
        row1[j] = 1
        A.append(row1)
        b.append(1)
        row2[j] = -1
        A.append(row2)
        b.append(0)

    for relative_sparse_dims in relative_sparse_dims_list:
        row = np.zeros(dim)
        pm_one = [[1,-1] for _ in range(relative_sparse_dims.shape[0])]
        for i in itertools.product(*pm_one):
            row_copy = row.copy()
            row_copy[relative_sparse_dims] = i
            A.append(list(row_copy))
            b.append(1)
    A = np.array(A)
    b = np.array(b)
    vertices = pypoman.compute_polytope_vertices(A, b)
    V = np.array([list(v) for v in vertices]).T
    return (A,b), V
