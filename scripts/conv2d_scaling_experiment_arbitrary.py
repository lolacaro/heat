import heat as ht
import sys
import os
from timeit import default_timer as timer
import csv
import numpy as np

device = os.getenv("HEAT_DEVICE")
# define config
if len(sys.argv) > 1:
    split_image = int(sys.argv[1])
    if split_image < 0:
        split_image = None
    split_kernel = int(sys.argv[2])
    if split_kernel < 0:
        split_kernel = None
    incr_kernel = int(sys.argv[3])
    start_image = int(sys.argv[4])
    stop_image = int(sys.argv[5])
    kernel_size = int(sys.argv[6])
else:
    split_image = 0
    split_kernel = None
    incr_kernel = 0
    start_image = 256
    stop_image = 4097
    kernel_size = 5

world_size = ht.MPI_WORLD.size
rank = ht.MPI_WORLD.rank


# Only image distributed
kernel = ht.random.rand(kernel_size**2, split=split_kernel)
kernel = ht.reshape(kernel, (kernel_size, kernel_size))

# Iterate through image systes
image_sizes = ht.arange(start_image,stop_image,start_image, split=None)
times = {}
for c, current_size in enumerate(image_sizes):
    image = ht.random.rand(current_size**2, split=split_image, device=device)
    image = ht.reshape(image, (current_size, current_size))

    if incr_kernel:
        c_kernel_size = (c+1)*kernel_size
        kernel = ht.random.rand(c_kernel_size**2, split=split_kernel)
        kernel = ht.reshape(kernel, (c_kernel_size, c_kernel_size))

    for i in range(50):
        start = timer()
        convolved_image = ht.convolve2d(image, kernel, mode="full")
        stop = timer()
        if i == 0:
            times[current_size] = []
        times[current_size].append(stop - start)

    print(rank, "Convolved images", current_size, c_kernel_size)

if split_image is not None:
    sig_info = "splitSignal"
else:
    sig_info = "Signal"

if split_kernel is not None:
    kern_info = "splitKernel"
else:
    kern_info = "Kernel"

csv_base = f"Timing_conv2d_{device}_{sig_info}{start_image}-{stop_image}"
if incr_kernel:
    csv_file_name = f"{csv_base}_{kern_info}{kernel_size}-{c_kernel_size}.csv"
else:
    csv_file_name = f"{csv_base}_{kern_info}{kernel_size}.csv"

path = "timing_results"
if not os.path.exists(path):
    os.makedirs(path)
with open(f"{path}/{csv_file_name}", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(times.keys())

    for row in zip(*times.values()):
        w.writerow(row)
