####### POLYTOPE OPERATIONS UTILITIES ##############
import itertools
import numpy as np
import networkx as nx
import pypoman
from scipy.spatial import ConvexHull
from scipy import optimize
from sympy import Plane, Point3D
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from scipy.spatial import HalfspaceIntersection


def generate_practical_polytope(
    dim, antisparse_dims, nonnegative_dims, relative_sparse_dims_list
):
    """Generate a practical polytope in the form of.

       $\\left\\{\\mathbf{s}\\in \\mathbb{R}^n\\ \\middle\vert  s_i\\in[-1,1] \\, \forall i\\in \\mathcal{I}_s,\\, s_i\\in[0,1] \\,
       \forall i\\in \\mathcal{I}_+, \\, \\left\\|\vs_{\\mathcal{J}_k}\right\\|_1\\le 1, \\, \\mathcal{J}_k\\subseteq \\mathbb{Z}_n, \\,
       k\\in\\mathbb{Z}_L  \right\\}$
    Args:
        dim (int): Dimension of the polytope
        antisparse_dims (numpy array): Antisparse dimension indices
        nonnegative_dims (numpy array): Nonnegative antisparse dimension indices
        relative_sparse_dims_list (list of numpy arrays): Collection of mutually sparse dimension indices

    Returns:
        A (numpy array), b (numpy array): Inequalty constraints of the polytope, i.e., A @ x < b
        V (numpy array)                 : Vertices of the resulting polytope

    Example:
    >>> dim = 3
    >>> signed_dims = np.array([0,1])
    >>> nn_dims = np.array([2])
    >>> sparse_dims_list = [np.array([0,1]),np.array([1,2])]
    >>> (A,b), V = generate_practical_polytope(dim, signed_dims, nn_dims, sparse_dims_list)
    >>> print(A)
        [[ 1.  0.  0.]
        [-1.  0.  0.]
        [ 0.  1.  0.]
        [ 0. -1.  0.]
        [ 0.  0.  1.]
        [ 0.  0. -1.]
        [ 1.  1.  0.]
        [ 1. -1.  0.]
        [-1.  1.  0.]
        [-1. -1.  0.]
        [ 0.  1.  1.]
        [ 0.  1. -1.]
        [ 0. -1.  1.]
        [ 0. -1. -1.]]
    >>> print(b)
        [1 1 1 1 1 0 1 1 1 1 1 1 1 1]
    >>> print(V)
        [[ 1.  1.  0.  0. -1. -1.]
        [ 0.  0.  1. -1.  0.  0.]
        [ 1.  0.  0.  0.  0.  1.]]
    >>> np.linalg.norm(A @ V - b[:,np.newaxis] > 0 )
        0.0
    """
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
        pm_one = [[1, -1] for _ in range(relative_sparse_dims.shape[0])]
        for i in itertools.product(*pm_one):
            row_copy = row.copy()
            row_copy[relative_sparse_dims] = i
            A.append(list(row_copy))
            b.append(1)
    A = np.array(A)
    b = np.array(b)
    vertices = pypoman.compute_polytope_vertices(A, b)
    V = np.array([list(v) for v in vertices]).T
    return (A, b), V

def ProjectOntoPolytopeCanonical(y, A, b, x0, print_message = False):
    """
    Projection of a vector y onto a convex polytope defined by \{x∈Rn:Ax≤b\},
    with initial guess of x0.
    The optimization problem can be defined in matrix notation as follows:
    
    minimize_x || y - x ||^2

    subject to:
        Ax <= b
        
    We solve this problem using scipy's optimize.minimize function with Sequential Least Squares Programming (SLSQP)

    Args:
        y (numpy array): Given point to be projected onto the polytope
        A (numpy array): Matrix defining the polytope (by inequality Ax≤b)
        b (numpy array): Vector defining the polytope (by inequality Ax≤b)
        x0 (numpy array): Initial guess for the projection
        print_message (bool, optional): Print the message of scipy's optimization function (If optimization is successful or not). Defaults to False.

    Example:
    >>> dim = 3
    >>> signed_dims = np.array([0, 1, 2])
    >>> nn_dims = np.array([])
    >>> sparse_dims_list = []
    >>> (A, b), V = generate_practical_polytope(dim, signed_dims, nn_dims, sparse_dims_list)
    >>> y = np.array([-2, 1.1, 0.3])
    >>> ProjectOntoPolytopeCanonical(y, A, b, np.zeros(y.shape[0]), True)
    Output: 
    Optimization terminated successfully
    array([-1. ,  1. ,  0.3]
    """
    def loss(x): 
        return (0.5 * (x - y).T @ (x-y))
    
    def jac(x): # Jacobion of the loss function
        return (x - y)
    
    cons = {'type':'ineq',
            'fun':lambda x: b - np.dot(A,x),
            'jac':lambda x: -A}
    
    opt = {'disp':False}
    
    res_cons = optimize.minimize(loss, x0, jac=jac, constraints=cons,
                                 method='SLSQP', options=opt)
    
    if print_message:
        print(res_cons.message)
    return res_cons.x # Return the projected point

############ POLYTOPE PLOTTING FUNCTIONS ##########################
def simplify(triangles):
    """Simplify an iterable of triangles such that adjacent and coplanar triangles form
    a single face.

    Each triangle is a set of 3 points in 3D space.
    """

    # create a graph in which nodes represent triangles;
    # nodes are connected if the corresponding triangles are adjacent and coplanar
    G = nx.Graph()
    G.add_nodes_from(range(len(triangles)))
    for ii, a in enumerate(triangles):
        for jj, b in enumerate(triangles):
            if (
                ii < jj
            ):  # test relationships only in one way as adjacency and co-planarity are bijective
                if is_adjacent(a, b):
                    if is_coplanar(a, b, np.pi / 180.0):
                        G.add_edge(ii, jj)

    # triangles that belong to a connected component can be combined
    components = list(nx.connected_components(G))
    simplified = [
        set(flatten(triangles[index] for index in component))
        for component in components
    ]

    # need to reorder nodes so that patches are plotted correctly
    reordered = [reorder(face) for face in simplified]

    return reordered

def is_coplanar(a, b, tolerance_in_radians=0):
    a1, a2, a3 = a
    b1, b2, b3 = b
    plane_a = Plane(Point3D(a1), Point3D(a2), Point3D(a3))
    plane_b = Plane(Point3D(b1), Point3D(b2), Point3D(b3))
    if not tolerance_in_radians:  # only accept exact results
        return plane_a.is_coplanar(plane_b)
    else:
        angle = plane_a.angle_between(plane_b).evalf()
        angle %= np.pi  # make sure that angle is between 0 and np.pi
        return (angle - tolerance_in_radians <= 0.0) or ((np.pi - angle) - tolerance_in_radians <= 0.0)

def is_adjacent(a, b):
    return len(set(a) & set(b)) == 2  # i.e. triangles share 2 points and hence a side

flatten = lambda l: [item for sublist in l for item in sublist]

def get_distance(v1, v2):
    v2 = np.array(list(v2))
    difference = v2 - v1
    ssd = np.sum(difference**2, axis=1)
    return np.sqrt(ssd)

def reorder(vertices):
    """
    Reorder nodes such that the resulting path corresponds to the "hull" of the set of points.
    Note:
    -----
    Not tested on edge cases, and likely to break.
    Probably only works for convex shapes.
    """
    if len(vertices) <= 3:  # just a triangle
        return vertices
    else:
        # take random vertex (here simply the first)
        reordered = [vertices.pop()]
        # get next closest vertex that is not yet reordered
        # repeat until only one vertex remains in original list
        vertices = list(vertices)
        while len(vertices) > 1:
            idx = np.argmin(get_distance(reordered[-1], vertices))
            v = vertices.pop(idx)
            reordered.append(v)
        # add remaining vertex to output
        reordered += vertices
        return reordered

def Plot3DPolytope(
    verts, azim=50, elev=10, dist=10, xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1
):
    hull = ConvexHull(verts)
    faces = hull.simplices

    ax = a3.Axes3D(plt.figure())
    ax.dist = dist
    ax.azim = azim
    ax.elev = elev

    ax.set_xlim3d([xmin, xmax])
    ax.set_ylim3d([ymin, ymax])
    ax.set_zlim3d([zmin, zmax])

    triangles = []
    for s in faces:
        sq = [
            (verts[s[0], 0], verts[s[0], 1], verts[s[0], 2]),
            (verts[s[1], 0], verts[s[1], 1], verts[s[1], 2]),
            (verts[s[2], 0], verts[s[2], 1], verts[s[2], 2]),
        ]
        triangles.append(sq)

    new_faces = simplify(triangles)
    for sq in new_faces:
        f = a3.art3d.Poly3DCollection([list(sq)], alpha=0.4)
        f.set_color("b")  # colors.rgb2hex(sp.rand(3)))
        f.set_edgecolor("k")
        f.set_alpha(0.4)
        ax.add_collection3d(f)

def simpleplot3dpoly(verts, azim=0, elev=10, dist=10, color="b", figsize=(8, 5)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    hull = ConvexHull(verts)
    # draw the polygons of the convex hull
    for s in hull.simplices:
        tri = Poly3DCollection([verts[s]])
        tri.set_color(color)
        tri.set_alpha(0.5)
        ax.add_collection3d(tri)
    # draw the vertices
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], marker="o", color="purple")

    ax.view_init(elev=elev, azim=azim)
    ax.dist = dist
    plt.show()
