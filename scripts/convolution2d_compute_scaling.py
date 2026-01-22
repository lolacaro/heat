import heat as ht
from perun import monitor
import os

device = os.getenv("HEAT_DEVICE")

@monitor()
def conv2d_fixed_kernel(image, kernel, mode):
    return ht.convolve2d(image, kernel, mode=mode)

# Only image distributed
# Scale 2D image, convolution fixed
current_size = 100
kernel = ht.random.rand(9, split=None)
kernel = ht.reshape(kernel, (3, 3))
print("Prepared kernel")
for n in range(0,10):
    image = ht.random.rand(current_size**2, split=0, device=device)

    image = ht.reshape(image, (current_size, current_size))
    print("Prepared image", n, current_size)

    convolved_image = conv2d_fixed_kernel(image, kernel, mode="full")
    current_size = int(2*current_size)

print("Finished Scale 2d image, convolution fixed")


# Both distributed
# Scale 2D image + kernel the same way (keep window the same)

# Kernel distributed
# Scaled kernel, keep 2D image the same
