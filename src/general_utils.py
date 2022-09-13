#######  GENERAL UTILITY FUNCTIONS ####################
import itertools
import logging
import os
import sys
import time

import numpy as np
from IPython.display import Math, display


class Timer:
    """A simple timer class for performance profiling Taken from https://github.com/flat
    ironinstitute/online_psp/blob/master/online_psp/online_psp_simulations.py.

    Usage:
    with Timer() as t:
        DO SOMETHING HERE
    print('Above (DO SOMETHING HERE) took %f sec.' % (t.interval))
    """

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


class HiddenPrints:
    """
    Example:

    with HiddenPrints():
        print("This will not be printed")

    print("This will be printed as before")
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


########## Logging Utilities #########################
def create_my_logger(
    logger_level="INFO",
    logger_name="logger",
    create_logger_folder=True,
    logger_folder_name="Loggers",
):
    """Generate a logger file and returns the logger to use it in a python script.

    Args:
        logger_level (str, optional): Logging level. Please see: https://docs.python.org/3/library/logging.html#logging-levels.
                                      Defaults to "INFO".
        logger_name (str, optional): Logger file name. Defaults to "logger".
        create_logger_folder (bool, optional): If true, it creates a new folder to save the logger file. Defaults to True.
        logger_folder_name (str, optional): The name of the folder to keep the logger file if a new folder will be created.
                                             Defaults to "Loggers".

    Returns:
        _type_: Logger

    Example:
    >>> l = create_my_logger(logger_level = "INFO", logger_name = "logger",
                             create_logger_folder = True, logger_folder_name = "Loggers")
    >>> l.info("Here I am")
    """
    if create_logger_folder:
        if not os.path.exists(logger_folder_name):
            os.mkdir(logger_folder_name)
    logger = logging.getLogger(__name__)
    if logger_level == "INFO":
        logger.setLevel(logging.INFO)
    elif logger_level == "WARNING":
        logger.setLevel(logging.WARNING)
    elif logger_level == "ERROR":
        logger.setLevel(logging.ERROR)
    elif logger_level == "CRITICAL":
        logger.setLevel(logging.CRITICAL)
    else:
        print(
            "Wrong logging level has been entered. Please try INFO, WARNING, ERROR, or CRITICAL."
        )

    formatter = logging.Formatter("%(asctime)s | | %(levelname)s | | %(message)s")
    logger_file_name = os.path.join(logger_folder_name, logger_name)
    file_handler = logging.FileHandler(logger_file_name, "w")

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


########### Set Theoretic Functions ##################
def findsubsets(S, m):
    """Find subsets of given set with given number of elements.

    Args:
        S (set or list): A list of elements
        m (int): Number of elements of the subsets

    Returns:
        _type_ (set): All possible subsets of S with m elements

    Examples:
    >>> findsubsets({0,1,2,3}, 2)
    {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}

    >>> findsubsets([0,1,2,3], 3)
    {(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)}
    """
    return set(itertools.combinations(S, m))


def swapped(l, p1, p2):
    """Swap the elements of l in indices p1 and p2.

    Args:
        l (list or numpy array): A list of elements whose elements to be swapped
        p1 (int): Swapping position 1
        p2 (int): Swappint position 2

    Returns:
        _type_ (list or numpy array): Resulting swapped list

    Examples:
    >>> swapped([0,1,2,3,4], 1,3)
    [0, 3, 2, 1, 4]
    >>> swapped(np.array([0,1,2,3,4]), 1,2)
    array([0, 2, 1, 3, 4])
    """
    r = l[:]
    r[p1], r[p2] = r[p2], r[p1]
    return r


def transpositions(A):
    """Returns all transpositions of A.

    Args:
        A (list or numpy array):

    Returns:
        _type_ (list): List of all transpositions

    Examples:
    >>> transpositions([0,1,2])
    [[1, 0, 2], [2, 1, 0], [0, 2, 1]]
    >>> transpositions(np.array([0,1,2]))
    [array([2, 1, 0]), array([2, 1, 0]), array([2, 1, 0])]
    """
    return [swapped(A, i, j) for i, j in list(findsubsets(A, 2))]


def elementary_perm_matrix(size, frm, to):
    """Returns an elementary permutation matrix (to permute from index 'frm' to index
    'to')

    Args:
        size (int): Size of the permutation matrix
        frm (int): _description_
        to (int): _description_

    Returns:
        _type_ (numpy array): Elementary permutation matrix

    Examples:
    >>> elementary_perm_matrix(3,1,2)
    array([[1., 0., 0.],
           [0., 0., 1.],
           [0., 1., 0.]])
    """
    P = np.identity(size)
    P[:, [frm, to]] = P[:, [to, frm]]
    return P


def check_sign_perm(perm_mat):
    """Check if given matrix (perm_mat) is a signed permutation or not.

    Args:
        perm_mat (numpy array): Input (which will be checked to be signed permutation or not)

    Returns:
        _type_ (bool): Returns True if given matrix is signed permutation, and returns false if not.

    Examples:
    >>> A = elementary_perm_matrix(3,1,2)
    >>> print(check_sign_perm(A))
    True
    >>> A2 = A.copy()
    >>> A2[0,1] = 1
    >>> print(check_sign_perm(A2))
    False
    """
    n = perm_mat.shape[0]
    if np.linalg.norm((np.abs(perm_mat) > 1e-6).sum(axis=0) - np.ones((1, n))) < 1e-6:
        return True
    else:
        return False


########### LATEX Style Display Matrix ###############
def display_matrix(array):
    """Display given numpy array with Latex format in Jupyter Notebook.

    Args:
        array (numpy array): Array to be displayed
    """
    data = ""
    for line in array:
        if len(line) == 1:
            data += " %.3f &" % line + r" \\\n"
            continue
        for element in line:
            data += " %.3f &" % element
        data += r" \\" + "\n"
    display(Math("\\begin{bmatrix} \n%s\\end{bmatrix}" % data))


########### Other Utilities ##########################
def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return "{:.2f}{}{}".format(bytes, unit, suffix)
        bytes /= factor
