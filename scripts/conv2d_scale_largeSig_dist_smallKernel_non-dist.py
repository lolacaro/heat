import heat as ht
from perun import monitor
import os
from timeit import default_timer as timer
import csv
import numpy as np

device = os.getenv("HEAT_DEVICE")

@monitor()
def conv2d_fixed_kernel_1(image, kernel, mode):
    return ht.convolve2d(image, kernel, mode=mode)
@monitor()
def conv2d_fixed_kernel_2(image, kernel, mode):
    return ht.convolve2d(image, kernel, mode=mode)
@monitor()
def conv2d_fixed_kernel_3(image, kernel, mode):
    return ht.convolve2d(image, kernel, mode=mode)
@monitor()
def conv2d_fixed_kernel_4(image, kernel, mode):
    return ht.convolve2d(image, kernel, mode=mode)
@monitor()
def conv2d_fixed_kernel_5(image, kernel, mode):
    return ht.convolve2d(image, kernel, mode=mode)

# Only image distributed
kernel = ht.random.rand(25, split=None, device=device)
kernel = ht.reshape(kernel, (5, 5))
# Scale 2D image, convolution fixed

current_size = 256
image = ht.random.rand(current_size**2, split=0, device=device)
image = ht.reshape(image, (current_size, current_size))
print("Prepared image", current_size)

times={}
for i in range(50):
    start = timer()
    convolved_image = conv2d_fixed_kernel_1(image, kernel, mode="full")
    stop = timer()
    if i == 0:
        times[current_size] = []
    times[current_size].append(stop - start)

print("Convolved images", current_size)

current_size = 512
image = ht.random.rand(current_size ** 2, split=0, device=device)
image = ht.reshape(image, (current_size, current_size))
print("Prepared image", current_size)

for i in range(50):
    start = timer()
    convolved_image = conv2d_fixed_kernel_2(image, kernel, mode="full")
    stop = timer()
    if i == 0:
        times[current_size] = []
    times[current_size].append(stop - start)

print("Convolved image", current_size)


current_size = 1024
image = ht.random.rand(current_size ** 2, split=0, device=device)
image = ht.reshape(image, (current_size, current_size))
print("Prepared image", current_size)

for i in range(50):
    start = timer()
    convolved_image = conv2d_fixed_kernel_3(image, kernel, mode="full")
    stop = timer()
    if i == 0:
        times[current_size] = []
    times[current_size].append(stop - start)

current_size = 2048
image = ht.random.rand(current_size ** 2, split=0, device=device)
image = ht.reshape(image, (current_size, current_size))
print("Prepared image", current_size)

for i in range(50):
    start = timer()
    convolved_image = conv2d_fixed_kernel_4(image, kernel, mode="full")
    stop = timer()
    if i == 0:
        times[current_size] = []
    times[current_size].append(stop - start)

print("Convolved image", current_size)

current_size = 4096
image = ht.random.rand(current_size ** 2, split=0, device=device)
image = ht.reshape(image, (current_size, current_size))
print("Prepared image", current_size)

for i in range(50):
    start = timer()
    convolved_image = conv2d_fixed_kernel_5(image, kernel, mode="full")
    stop = timer()
    if i == 0:
        times[current_size] = []
    times[current_size].append(stop - start)

print("Convolved image", current_size)


print("Finished Scale 2d image, convolution fixed")

print()
for key in times.keys():
    print(key, np.mean(times[key]), np.std(times[key]))
with open("Timing_inc-dist-signal_fix-nondist-kernels_%s.csv"%device, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(times.keys())

    for row in zip(*times.values()):
        w.writerow(row)
