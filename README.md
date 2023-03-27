# Mandelbrot Set Demo

This is a demo of the Mandelbrot set implemented using Cython (roughly x40 speedup) and MPI to create an zooming animation. The demo allows you to create animations to explore different regions of the spanning over 13 orders of magnitude and adjust various parameters such as the maximum iteration count, colour scheme, frame resolution, etc.

![Mandelbrot Set Demo](./examples/mandelbrot_f2438.png)

## Equation

The Mandelbrot set is defined by the following equation:

$$ z_{n+1} = z_n^2 + c $$

where $z_n$ is a complex number and $c$ is a constant complex number that varies across the complex plane. The set is defined as the set of complex numbers $c$ for which the sequence $z_0, z_1, z_2, \ldots$ remains bounded as $n$ approaches infinity.

## Features

- Efficient Cython implementation
- MPI parallelization for a further speedup
- Easily set up zooming animations
- Able to zoom over 13 orders of magnitude
- TQDM progress bar
- Adjust the maximum iteration count
- Change the colour scheme
- Start on a specific frame
- Ouput folder parameter

## Installation

To install and run the demo, follow these steps:

1. Clone the repository to your local machine
2. Install the required dependencies (Cython, mpi4py, opencv, tqdm). You might have to run `pip install opencv-python` to be able to use cv2 for creating animations.
3. Compile the `mandelbrot.pyx` file with `python setup.py build_ext --inplace`
4. Run `mandelbrot.py`. For MPI parallelization use `mpiexec -n nthreads python mandelbrot.py` where `nthreads` is the number of cores. For hyperthreading, specify the `mpiexec -n nthreads --use-hwthread-cpus python mandelbrot.py` argument. This will create all the frames.
5. Run `animate.py` to put the frames together for a video.

## Usage

To use the demo:

1. Choose your resolution, maximum number of iterations, central coordinates, and the number of frames for your animation. You can set the number of frames to 1 for this exploration phase.
2. Establish the minimum and maximum zoom levels (i.e. `coordinates['dpix']`), then compute the linear zoom in factor $F_{z}$ as follows:

$$ F_{z} = \left ( \frac{Z_{\mathrm{min}}}{Z_{\mathrm{max}}} \right )^{1/N_\mathrm{frames}} $$

3. Run your animation. The frames will be saved to the folder specified.
4. Turn the frames into a video using `animate.py`. Make sure the resolution matches the resolution of the images.

## Code

The code for this demo is written in Python and uses the Cython and MPI libraries for computational efficiency. The code is organized into several files:

- `mandelbrot.py`: The main script that generates the frames and handles user input.
- `mandelbrot.pyx`: Contains the code for computing the fractal.
- `setup.py`: Contains the setup code for compiling the Cython code.
- `animate.py`: Contains the script to animate the frames and save them as a video.

## Examples

Here is an examples of the output of the demo:

![Mandelbrot Island](./examples/mandelbrot_f0.png)

This coordinate is known as the "Mandelbrot Island", and is located near the center of the set. Zooming in on this point will reveal a fascinating pattern of smaller islands and filaments, each with their own unique shapes and structures.

## Troubleshooting

If you encounter any issues or errors while using the demo, please [open an issue](https://github.com/RokeCepedaArroita/mandelbrot/issues) on GitHub.
