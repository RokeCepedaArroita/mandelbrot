from cython.parallel import prange

cdef int mandelbrot(complex z, int maxiter):
    cdef complex c = z
    for n in range(maxiter):
        if abs(z) > 2.0:
            return n
        z = z*z + c
    return maxiter

def compute_mandelbrot(int[:, :] output, double xmin, double xmax, double ymin, double ymax, int maxiter):
    cdef int nx = output.shape[1]
    cdef int ny = output.shape[0]
    cdef double dx = (xmax - xmin) / nx
    cdef double dy = (ymax - ymin) / ny

    cdef int i, j

    with nogil:
        for j in prange(ny):
            with gil:
                for i in range(nx):
                    x = xmin + i*dx
                    y = ymin + j*dy
                    output[j, i] = mandelbrot(x + y*1j, maxiter)
