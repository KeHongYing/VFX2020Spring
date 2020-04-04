import numpy as np
from myImgIO import read_image, output_image
import sys

image = read_image(sys.argv[1:], (1500, 1000))

stack1, stack2, stack3 = image[0], image[3], image[6]

for i in range(1, 3):
	stack1 = np.hstack((np.hstack((stack1, np.ones((1000, 1, 3)) * 255)), image[i]))
	stack2 = np.hstack((np.hstack((stack2, np.ones((1000, 1, 3)) * 255)), image[i + 3]))
	stack3 = np.hstack((np.hstack((stack3, np.ones((1000, 1, 3)) * 255)), image[i + 6]))

pad = np.ones((1, 4502, 3)) * 255
result = np.vstack((stack1, pad, stack2, pad, stack3))
output_image(result, "origin_assemble.png")
