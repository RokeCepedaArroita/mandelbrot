''' First run: python setup.py build_ext --inplace
    For parallel multiprocessing, call this with
    mpiexec -n nthreads python mandelbrot.py, i.e.
    mpiexec -n 4 python mandelbrot.py'''

import numpy as np
import pyximport
pyximport.install()
import mandelbrot
import matplotlib.pyplot as plt

# Maximum number of iterations
maxiter = 3000

# Resolution, centre and pixel size
coordinates = {'resolution': [3840, 2160], # 3840, 2160 for 4K video
               'centre': [-1.111625614959122, 0.23062280776701957],
               'dpix': 1.3e-3} # pixel size, for 4K I recommend a start of 1.3e-3 and limit 5e-17 (i.e. image widths of 1e-13 are near the precision limit)

# Choose a custom color map
from matplotlib import colors
from colour_scales import *
my_colours = ultra_fractal
cmap = colors.LinearSegmentedColormap.from_list('mycmap', my_colours[::-1])

# Create video: for 3 min 4K video at 30 fps need 5400 frames, have 2.6*10^13 orders
# of magnitude to cover, so it will take around 8 hours on 30 threads.
nframes = 5400
start_frame = 0
initial_dpix = np.copy(coordinates['dpix']) # initial pixel size
zoom_factor = 0.994296 # zoom to apply to each frame
folder_name = 'my_animation'
keyframes_only = False # set to True to compute only 10 evenly spaced frames

# Start MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# If folder does not exist then create it
if rank==0:
    import os   # Make source-specific directory
    if not os.path.exists(f'./{folder_name}'):
        print(f'Creating video directory "{folder_name}" since it does not exist.',flush=True)
        os.makedirs(f'./{folder_name}')

# If testing only check a number of evenly spaced keyframes
if keyframes_only:
    N_keyframes = 20
    framelist = list(range(0, nframes+1, nframes//(N_keyframes-1)))[:N_keyframes] if N_keyframes > 0 else [0]
    nframes = np.shape(framelist)[0]
else:
    framelist = range(start_frame, start_frame+nframes)

# Initialize the tqdm progress bar on rank 0
from tqdm import tqdm
if rank == 0:
    progress_bar = tqdm(total=nframes)

# Loop through every frame
for frame in framelist:
    '''
    Assign the right task to each thread. If the node is
    correct loop continues. If it is not correct then the
    rank continues on to the next iteration!
    '''

    if frame%size!=rank: continue

    # Redefine coordinates by applying the zoom factor
    coordinates['dpix'] = initial_dpix*zoom_factor**frame

    # Calculate the extent from the coordinate dictionary
    def calculate_coordinates(coordinates):
        resolution_x, resolution_y = coordinates['resolution']
        centre_x, centre_y = coordinates['centre']
        dpix = coordinates['dpix']
        xmin = centre_x - (resolution_x / 2) * dpix
        xmax = centre_x + (resolution_x / 2) * dpix
        ymin = centre_y - (resolution_y / 2) * dpix
        ymax = centre_y + (resolution_y / 2) * dpix
        return xmin, xmax, ymin, ymax

    extent = np.array(calculate_coordinates(coordinates))

    # Initialise the image
    output = np.zeros((coordinates['resolution'][1], coordinates['resolution'][0]), dtype=np.int32)

    # Compute the set
    mandelbrot.compute_mandelbrot(output, *calculate_coordinates(coordinates), maxiter)

    # Display the Mandelbrot set using imshow
    plt.figure(figsize=(16,9))
    plt.imshow(output, cmap=cmap, vmin=0, vmax=maxiter)
    plt.axis('off')

    # At 4K, images will take roughly 4.3 MB each, so roughly 23 GB set. The dpi
    # and figsize are set so that the original resolution is preserved at 4K.
    plt.savefig(f'./{folder_name}/mandelbrot_f{frame}.png', bbox_inches='tight', pad_inches=0, dpi=311.69)

    # Close figure
    plt.close()

    # Update the tqdm progress bar on rank 0
    if rank == 0:
        progress_bar.update(1*size)
