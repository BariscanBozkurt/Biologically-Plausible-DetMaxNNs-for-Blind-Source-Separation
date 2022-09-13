#######  GENERAL UTILITY FUNCTIONS ####################
import time
from IPython.display import display, Math

class Timer:
    """
    A simple timer class for performance profiling
    Taken from https://github.com/flatironinstitute/online_psp/blob/master/online_psp/online_psp_simulations.py
    
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

def display_matrix(array):
    data = ''
    for line in array:
        if len(line) == 1:
            data += ' %.3f &' % line + r' \\\n'
            continue
        for element in line:
            data += ' %.3f &' % element
        data += r' \\' + '\n'
    display(Math('\\begin{bmatrix} \n%s\end{bmatrix}' % data))
