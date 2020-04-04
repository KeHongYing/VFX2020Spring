import numpy as np
import os
from findEg import find_E_and_g
from myImgIO import read_image_rawpy, get_exposure_time
from threading import Thread
from argparse import ArgumentParser

def split_RGB(img):
	from PIL import Image
	R, G, B = [], [], []

	for i in img:
		r, g, b = Image.fromarray(i).split()
		R.append(np.array(r).reshape(-1))
		G.append(np.array(g).reshape(-1))
		B.append(np.array(b).reshape(-1))

	return np.array(R).T, np.array(G).T, np.array(B).T

parser = ArgumentParser()
parser.add_argument("-d", "--directory", dest = "dir", help = "Image directory")
parser.add_argument("-i", "--iteration", dest = "iter", help = "Iteration times", type = int)
parser.add_argument("-s", "--shape", dest = "shape", help = "Image directory", default = (750, 500), type = eval)
args = parser.parse_args()

print(type(args.shape))

img_list = [os.path.join(args.dir, i) for i in os.listdir(args.dir)]
img_list.sort()
shape = args.shape

LDR_image = read_image_rawpy(img_list, shape)
print(LDR_image.shape)
img_width, img_height = LDR_image[0].shape[1], LDR_image[0].shape[0]

iteration = args.iter
R, G, B = split_RGB(LDR_image)
T = get_exposure_time(img_list)

E, g = [None] * 3, [None] * 3
threads = [None] * 3

threads[0] = Thread(target = find_E_and_g, args = (R, T, img_width * img_height, LDR_image.shape[0], None, None, iteration, E, g, 0))
threads[1] = Thread(target = find_E_and_g, args = (G, T, img_width * img_height, LDR_image.shape[0], None, None, iteration, E, g, 1))
threads[2] = Thread(target = find_E_and_g, args = (B, T, img_width * img_height, LDR_image.shape[0], None, None, iteration, E, g, 2))

threads[0].start()
threads[1].start()
threads[2].start()

for i in range(3):
	threads[i].join()

np.save("gR.npy", g[0])
np.save("gG.npy", g[1])
np.save("gB.npy", g[2])
np.save("ER.npy", E[0])
np.save("EG.npy", E[1])
np.save("EB.npy", E[2])
