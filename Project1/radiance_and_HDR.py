import numpy as np
from myImgIO import output_image
from argparse import ArgumentParser

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def plot_radiance(img):
	import matplotlib.pyplot as plt
	
	fig = plt.figure()
	im = plt.imshow(img, cmap = "jet")
	fig.colorbar(im)
	plt.savefig("radiance.png")

def plot_exr(img):
	import pyexr
	pyexr.write("radiance.exr", img)
	

def normalize(img):
	gamma = 1
	delta = 1e-8
	Enorm = (img - np.min(img)) / (np.max(img) - np.min(img))
	Egamma = Enorm ** gamma
	L = rgb2gray(Egamma) + delta
	Lavg = np.exp(np.sum(np.log(L)) / np.prod(L.shape))
	a = 0.18
	T = (a / Lavg) * L
	Ltone = T * (1 + T / np.max(T) ** 2) / (1 + T)
	M = Ltone / L
	
	return Egamma * np.dstack((M, M, M))

def assemble_HDR(R, G, B):
	img = np.dstack((R, G, B))
	plot_radiance(rgb2gray(img))
	plot_exr(img)
	
	img = normalize(img)
	output_image(np.clip(img * 255, 0, 255), "RGB.png")

parser = ArgumentParser()
parser.add_argument("-r", "--red", help = "a .npy file of ER", dest = "ER", default = "./ER.npy")
parser.add_argument("-g", "--green", help = "a .npy file of EG", dest = "EG", default = "./EG.npy")
parser.add_argument("-b", "--blue", help = "a .npy file of EB", dest = "EB", default = "./EB.npy")
parser.add_argument("-s", "--shape", help = "image shape", dest = "shape", default = (500, 750), type = eval)
args = parser.parse_args()

ER = np.load(args.ER)
EG = np.load(args.EG)
EB = np.load(args.EB)
shape = args.shape

assemble_HDR(ER.reshape(shape), EG.reshape(shape), EB.reshape(shape))
