from features import HarrisKeypointDetector, MOPSFeatureDescriptor, SSDFeatureMatcher
from argparse import ArgumentParser
from myIO import read_image
from find_motion import alignPair 
from image_pasting import ImageInfo, blendImages
from cylindricalWarping import cylindricalWarp
import numpy as np
import os
import cv2

def parse():
	parser = ArgumentParser()
	parser.add_argument("-d", "--dir", help = "data directory", dest = "data", default = "./data1")
	parser.add_argument("-o", "--output", help = "output directory", dest = "output", default = "./output1")
	parser.add_argument("-r", "--resize", help = "resize", dest = "resize", default = 1, type = eval)
	parser.add_argument("-n", "--num", help = "num of feature point", dest = "n_fp", default = 250, type = int)
	parser.add_argument("-f", "--focal_length", help = "image focal length", dest = "f", default = 800, type = int)
	
	return parser.parse_args()

def draw_feature_point(image, feature, matcher):
	color = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(20)]
	
	for idx, m in enumerate(match):
		cnt = 0
		for i in m:
			center1, center2 = feature[idx][i.queryIdx].pt, feature[(idx + 1) % n][i.trainIdx].pt
			cv2.circle(image[idx], (int(center1[0]), int(center1[1])), 3, (0, 0, 255), 1)
			cv2.circle(image[(idx + 1) % n], (int(center2[0]), int(center2[1])), 3, (0, 0, 255), 1)
		
			cv2.putText(image[idx], str(cnt), (int(center1[0]), int(center1[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color[idx], 1, cv2.LINE_AA)
			cv2.putText(image[(idx + 1) % n], str(cnt), (int(center2[0]), int(center2[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color[idx], 1, cv2.LINE_AA)
		
			cnt += 1
	
	return image

if __name__ == "__main__":
	args = parse()

	if not os.path.exists(args.output):
		os.mkdir(args.output)

	path = os.listdir(args.data)
	path.sort()
	
	# Read image
	image = [read_image(os.path.join(args.data, img), args.resize) for img in path]
	
	# Cylindrical Warping
	h, w = image[0].shape[: 2]
	K = np.array([[args.f, 0, w / 2], [0, args.f, h / 2], [0, 0, 1]]) # mock intrinsics
	cylindrical_image = [cylindricalWarp(i, K) for i in image]

	# Detect the feature point
	feature = [HarrisKeypointDetector().detectKeypoints(img) for img in cylindrical_image]
	
	n = len(cylindrical_image)
	
	# Feature point description
	descriptor = [MOPSFeatureDescriptor().describeFeatures(cylindrical_image[i], feature[i]) for i in range(n)]
	
	# Matching the feature point
	match = [SSDFeatureMatcher().matchFeatures(descriptor[i], descriptor[(i + 1) % n]) for i in range(n)]

	#feature_point_cylindrical_image = draw_feature_point(image, feature, match)

	# Find the translation matrix
	motion_model = [alignPair(feature[n - 1], feature[0], match[n - 1], 0, 5000, 15)]
	for i in range(1, n):
		motion_model.append(np.dot(alignPair(feature[(n - i - 1) % n], feature[n - i], match[(n - i - 1) % n], 0, 5000, 15), motion_model[i - 1]))
	
	motion_model.reverse()
	for i in range(1, n):
		print(motion_model[i - 1][0, 2] / motion_model[i][0, 2])
	
	# Blending
	ipv = [ImageInfo(f"img_{i}", cylindrical_image[i % n], motion_model[i % n]) for i in range(n)]
	img = blendImages(ipv, 0, True)
	
	cv2.imwrite(os.path.join(args.output, "test.jpg"), img[:int(image[0].shape[0] * 0.95)])
