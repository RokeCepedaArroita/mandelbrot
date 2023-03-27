''' First run: python setup.py build_ext --inplace
    For parallel multiprocessing, call this with mpiexec -n nthreads python mandelbrot.py'''

import numpy as np
import pyximport
pyximport.install()
import mandelbrot
import matplotlib.pyplot as plt

# Maximum number of iterations
maxiter = 3000 # 3000 saturates the last frame but gives better dynamic range midway through

# Resolution, centre and pixel size
coordinates = {'resolution': [3840, 2160], # 3840, 2160
               'centre': [-0.74364388703715, 0.131825904205330],
               'dpix': 1.3e-3} # start 1.3e-3, limit 5e-17

# Create a custom color map
from matplotlib import colors
my_colours = [(0.0, 0.0, 0.0),    # black
              (0.15, 0.15, 0.7),  # dark blue
              (0.25, 0.45, 0.9),  # light blue
              (0.5, 0.7, 0.5),    # light green
              (0.9, 0.6, 0.25),   # light orange
              (0.8, 0.3, 0.2),    # dark orange
              (0.7, 0.0, 0.15),   # deep red
              (1.0, 1.0, 1.0)]    # white
cmap = colors.LinearSegmentedColormap.from_list('mycmap', my_colours)

# Create video: for 3 min 30 fps need 5400 frames, have 2.6*10^13 orders
# of magnitude to cover, so it will take around 45 hours on my laptop.
nframes = 1 # 5400
start_frame = 2438
initial_zoom = 1.3e-3 # 1.3e-3
zoom_factor = 0.994296 # 60 fps = 0.997144
folder_name = 'video'

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

# Initialize the tqdm progress bar on rank 0
from tqdm import tqdm
if rank == 0:
    progress_bar = tqdm(total=nframes)

# Loop through every frame
for frame in range(start_frame, start_frame+nframes):
    '''
    Assign the right task to each thread. If the node is
    correct loop continues. If it is not correct then the
    rank continues on to the next iteration!
    '''

    if frame%size!=rank: continue

    # Redefine coordinates by applying the zoom factor
    coordinates['dpix'] = initial_zoom*zoom_factor**frame

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
    plt.imshow(output, cmap=cmap)
    plt.axis('off')

    # At 4K, images will take roughly 4.3 MB each, so roughly 23 GB set. The dpi
    # and figsize are set so that the original resolution is preserved at 4K.
    plt.savefig(f'./{folder_name}/mandelbrot_f{frame}.png', bbox_inches='tight', pad_inches=0, dpi=311.69)

    # Update the tqdm progress bar on rank 0
    if rank == 0:
        progress_bar.update(1*size)
