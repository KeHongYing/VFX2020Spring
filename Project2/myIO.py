import numpy as np
import cv2

def read_image(path, resize):
	img = cv2.imread(path)
	shape = (int(img.shape[1] * resize), int(img.shape[0] * resize))
	img = cv2.resize(img, shape)

	return img

if __name__ == "__main__":
	pass
