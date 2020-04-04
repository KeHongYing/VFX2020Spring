import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-r", "--red", help = "A .npy file of gR", dest = "gR", default = "./gR.npy")
parser.add_argument("-g", "--green", help = "A .npy file of gG", dest = "gG", default = "./gG.npy")
parser.add_argument("-b", "--blue", help = "A .npy file of gB", dest = "gB", default = "./gB.npy")
args = parser.parse_args()

gR = np.load(args.gR)
gG = np.load(args.gG)
gB = np.load(args.gB)

plt.ylabel("Pixel Value")
plt.xlabel("Log Exposure")

plt.plot(np.log(gR), [i for i in range(256)], label = "R", color = "red")
plt.plot(np.log(gG), [i for i in range(256)], label = "G", color = "green")
plt.plot(np.log(gB), [i for i in range(256)], label = "B", color = "blue")
plt.legend(loc = 0)
plt.savefig("RecoverResponseCurve.png")
