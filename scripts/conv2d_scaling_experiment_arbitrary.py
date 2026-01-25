import heat as ht
import sys
import os
from perun import monitor



if os.environ["HEAT_DEVICE"]:
    device = os.environ["HEAT_DEVICE"]
else:
    device = "cpu"

print(device)
ht.use_device(device)
ht.random.seed(732034)

# define config
if len(sys.argv) > 1:
    split_image = int(sys.argv[1])
    d_split_image = split_image
    if split_image < 0:
        split_image = None
        d_split_image = "nodist"
    split_kernel = int(sys.argv[2])
    d_split_kernel = split_kernel
    if split_kernel < 0:
        split_kernel = None
        d_split_kernel = "nodist"
    incr_kernel = int(sys.argv[3])
    start_image = int(sys.argv[4])
    stop_image = int(sys.argv[5])
    kernel_size = int(sys.argv[6])
else:
    split_image = 0
    d_split_image = 0
    split_kernel = None
    d_split_kernel = "nodist"
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
image_sizes = range(start_image,stop_image+1,start_image)
times = {}
for c, current_size in enumerate(image_sizes):
    image = ht.random.rand(current_size**2, split=split_image)
    image = ht.reshape(image, (current_size, current_size))

    if incr_kernel:
        c_kernel_size = (c+1)*kernel_size
        kernel = ht.random.rand(c_kernel_size**2, split=split_kernel)
        kernel = ht.reshape(kernel, (c_kernel_size, c_kernel_size))
    else:
        c_kernel_size = kernel_size

    print (image.shape, kernel.shape)

    @monitor(f"convolve2d_{device}{world_size}_image-{d_split_image}-{current_size}_kernel-{d_split_kernel}-{c_kernel_size}")
    def call_convolution2d(signal, kernel):
        convolved_image = ht.convolve2d(signal, kernel, mode="full")

    for i in range(50):
        call_convolution2d(image, kernel)

    print(rank, "Convolved images", current_size, c_kernel_size)
