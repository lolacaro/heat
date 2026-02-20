import heat as ht
from perun import monitor
import sys
import numpy as np

device = "cpu"
ht.use_device(device)

#monitored functions

@monitor()
def conv2d_sobel_batch(image, kernel, mode):
    y = ht.convolve2d(image, kernel, mode)
    x = ht.convolve2d(image, kernel.T, mode)
    return ht.abs(y) + ht.abs(x)

@monitor()
def conv2d_gauss_batch(image, kernel, mode):
    return ht.convolve2d(image, kernel, mode)


# Sobel filter
base_sobel_y_7x7 = [[-3/18, -2/13, -1/10, 0, 1/10, 2/13, 3/18],
                    [-3/13, -2/8,  -1/5,  0, 1/5,  2/8,  3/13],
                    [-3/10, -2/5,  -1/2,  0, 1/2,  2/5,  3/10],
                    [-3/9,  -2,    -1,    0, 1,    2,   3/9],
                    [-3/10, -2/5,  -1/2,  0, 1/2,  2/5,  3/10],
                    [-3/13, -2/8,  -1/5,  0, 1/5,  2/8,  3/13],
                    [-3/18, -2/13, -1/10, 0, 1/10, 2/13, 3/18]]


sobel_y_7x7_np = np.array(base_sobel_y_7x7)*20
sobel_y_7x7 = ht.array(sobel_y_7x7_np, split=None)

# Gaussian filter
def gaussian_filter(kernel_size, sigma=1, muu=0):
    # Initializing value of x, y as grid of kernel size in the range of kernel size
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2 + y**2)

    # Normal part of the Gaussian function
    normal = 1 / (2 * np.pi * sigma**2)

    # Calculating Gaussian filter
    gauss = np.exp(-((dst - muu)**2 / (2.0 * sigma**2))) * normal

    return gauss  # Return the calculated Gaussian filter

gauss_9x9 = ht.array(gaussian_filter(9), split=None)
# data

data_path = "/lsdf/kit/scc/projects/himalaya/Data/Data/hdf5/CT_P0001.h5"
set_name = "CT/volume"
if len(sys.argv) > 1:
    subset_min = int(sys.argv[1])
    subset_max = int(sys.argv[2])
    slice_config = slice(subset_min, subset_max)
else:
    slice_config = None

data = ht.load_hdf5(data_path, set_name, split=0, slices=(slice_config, None, None))

# Experiments batch
for i in range(50):
    conv2d_sobel_batch(data, sobel_y_7x7, mode="same")
    conv2d_gauss_batch(data, gauss_9x9, mode="same")
