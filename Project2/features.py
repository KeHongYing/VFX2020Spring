import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial
import pdb

import transformations

## Keypoint detectors ##########################################################

class KeypointDetector(object):
	def detectKeypoints(self, image):
		'''
		Input:
			image -- uint8 BGR image with values between [0, 255]
		Output:
			list of detected keypoints, fill the cv2.KeyPoint objects with the
			coordinates of the detected keypoints, the angle of the gradient
			(in degrees), the detector response (Harris score for Harris detector)
			and set the size to 10.
		'''
		raise NotImplementedError()

class HarrisKeypointDetector(KeypointDetector):
	# Compute harris values of an image.
	def computeHarrisValues(self, srcImage):
		'''
		Input:
			srcImage -- Grayscale input image in a numpy array with
						values in [0, 1]. The dimensions are (rows, cols).
		Output:
			harrisImage -- numpy array containing the Harris score at
						   each pixel.
			orientationImage -- numpy array containing the orientation of the
								gradient at each pixel in degrees.
		'''
		Ix = ndimage.sobel(srcImage, axis=0)
		Iy = ndimage.sobel(srcImage, axis=1)

		Ixx = Ix**2
		Iyy = Iy**2
		Ixy = Ix*Iy

		A = ndimage.gaussian_filter(Ixx, 0.5)
		B = ndimage.gaussian_filter(Ixy, 0.5)
		C = ndimage.gaussian_filter(Iyy, 0.5)

		det = (A * C) - (B**2)
		trace = (A + C)
		alpha = 0.05

		harrisImage = det - alpha*(trace**2)

		degrees = np.arctan2(Ix, Iy)

		orientationImage = np.degrees(degrees)

		#self.saveHarrisImage(harrisImage, srcImage)

		return harrisImage, orientationImage

	def computeLocalMaxima(self, harrisImage):
		'''
		Input:
			harrisImage -- numpy array containing the Harris score at
						   each pixel.
		Output:
			destImage -- numpy array containing True/False at
						 each pixel, depending on whether
						 the pixel value is the local maxima in
						 its 7x7 neighborhood.
		'''

		destImage = harrisImage == ndimage.maximum_filter(harrisImage, 7)

		return destImage

	def NMS(self, features, shape, num = 250):

		def in_bound(pt, shape):
			return (pt[0] > 20 and pt[0] < shape[1] + 20) and (pt[1] > 20 and pt[1] < shape[0] + 20)

		r = [-i.response for i in features]
		pt = [i.pt for i in features]
		index = np.argsort(r)[: num * 30 if num * 30 < len(r) else len(r)]
		f = []

		choosed = [pt[index[0]]]
		
		radius = 10000
		n = 1

		while n < num:
			for idx in index:
				if in_bound(pt[idx], shape) and np.min(np.linalg.norm(np.array(choosed) - np.array(pt[idx]), axis = 1)) > radius:
					choosed.append(np.array(pt[idx]))
					f.append(features[idx])
					n += 1
			radius /= 2
	
		print(f"NMS radius:{radius}")

		return f

	def detectKeypoints(self, image):
		'''
		Input:
			image -- BGR image with values between [0, 255]
		Output:
			list of detected keypoints, fill the cv2.KeyPoint objects with the
			coordinates of the detected keypoints, the angle of the gradient
			(in degrees), the detector response (Harris score for Harris detector)
			and set the size to 10.
		'''
		image = image.astype(np.float32)
		image /= 255.
		height, width = image.shape[:2]
		features = []

		# Create grayscale image used for Harris detection
		grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# computeHarrisValues() computes the harris score at each pixel
		# position, storing the result in harrisImage.
		# You will need to implement this function.
		harrisImage, orientationImage = self.computeHarrisValues(grayImage)

		# Compute local maxima in the Harris image.  You will need to
		# implement this function. Create image to store local maximum harris
		# values as True, other pixels False
		harrisMaxImage = self.computeLocalMaxima(harrisImage)
		#harrisMaxImage = harrisImage

		# Loop through feature points in harrisMaxImage and fill in information
		# needed for descriptor computation for each point.
		# You need to fill x, y, and angle.
		for y in range(height):
			for x in range(width):
				if not harrisMaxImage[y, x]:
					continue

				f = cv2.KeyPoint()
				f.size = 10
				f.pt = (x, y)
				f.angle = orientationImage[y, x]
				f.response = harrisImage[y, x]
				# TODO 3: Fill in feature f with location and orientation
				# data here. Set f.size to 10, f.pt to the (x,y) coordinate,
				# f.angle to the orientation in degrees and f.response to
				# the Harris score

				features.append(f)

		features = self.NMS(features, harrisImage.shape, 400)

		return features

## Feature descriptors #########################################################


class FeatureDescriptor(object):
	# Implement in child classes
	def describeFeatures(self, image, keypoints):
		'''
		Input:
			image -- BGR image with values between [0, 255]
			keypoints -- the detected features, we have to compute the feature
			descriptors at the specified coordinates
		Output:
			Descriptor numpy array, dimensions:
				keypoint number x feature descriptor dimension
		'''
		raise NotImplementedError

class MOPSFeatureDescriptor(FeatureDescriptor):
	# TODO: Implement parts of this function
	def describeFeatures(self, image, keypoints):
		'''
		Input:
			image -- BGR image with values between [0, 255]
			keypoints -- the detected features, we have to compute the feature
			descriptors at the specified coordinates
		Output:
			desc -- K x W^2 numpy array, where K is the number of keypoints
					and W is the window size
		'''
		image = image.astype(np.float32)
		image /= 255.
		# This image represents the window around the feature you need to
		# compute to store as the feature descriptor (row-major)
		windowSize = 8
		desc = np.zeros((len(keypoints), windowSize * windowSize))
		grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		grayImage = ndimage.gaussian_filter(grayImage, 0.5)

		for i, f in enumerate(keypoints):
			# TODO 5: Compute the transform as described by the feature
			# location/orientation. You will need to compute the transform
			# from each pixel in the 40x40 rotated window surrounding
			# the feature to the appropriate pixels in the 8x8 feature
			# descriptor image.

			transMx = np.zeros((2, 3))

			trans1 = transformations.get_trans_mx(np.array([-f.pt[0], -f.pt[1], 0]))
			rotate = transformations.get_rot_mx(0, 0, -np.radians(f.angle))
			scale = transformations.get_scale_mx(0.2, 0.2, 1)
			trans2 = transformations.get_trans_mx(np.array([4, 4, 0]))

			transMx =  np.dot(trans2, np.dot(scale, np.dot(rotate, trans1)))[:2,(0, 1, 3)]


			# Call the warp affine function to do the mapping
			# It expects a 2x3 matrix
			destImage = cv2.warpAffine(grayImage, transMx,
				(windowSize, windowSize), flags=cv2.INTER_LINEAR)

			# TODO 6: Normalize the descriptor to have zero mean and unit
			# variance. If the variance is zero then set the descriptor
			# vector to zero. Lastly, write the vector to desc.

			destImage = destImage.flatten()

			std = np.std(destImage)
			if std < 1e-5:
				continue

			dest_mean = np.mean(destImage)
			std_dest = np.std(destImage)

			desc[i, :] = (destImage - dest_mean) / std_dest

		return desc

## Feature matchers ############################################################

class FeatureMatcher(object):
	def matchFeatures(self, desc1, desc2):
		'''
		Input:
			desc1 -- the feature descriptors of image 1 stored in a numpy array,
				dimensions: rows (number of key points) x
				columns (dimension of the feature descriptor)
			desc2 -- the feature descriptors of image 2 stored in a numpy array,
				dimensions: rows (number of key points) x
				columns (dimension of the feature descriptor)
		Output:
			features matches: a list of cv2.DMatch objects
				How to set attributes:
					queryIdx: The index of the feature in the first image
					trainIdx: The index of the feature in the second image
					distance: The distance between the two features
		'''
		raise NotImplementedError

	# Evaluate a match using a ground truth homography.  This computes the
	# average SSD distance between the matched feature points and
	# the actual transformed positions.
	@staticmethod
	def evaluateMatch(features1, features2, matches, h):
		d = 0
		n = 0

		for m in matches:
			id1 = m.queryIdx
			id2 = m.trainIdx
			ptOld = np.array(features2[id2].pt)
			ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

			# Euclidean distance
			d += np.linalg.norm(ptNew - ptOld)
			n += 1

		return d / n if n != 0 else 0

	# Transform point by homography.
	@staticmethod
	def applyHomography(pt, h):
		x, y = pt
		d = h[6]*x + h[7]*y + h[8]

		return np.array([(h[0]*x + h[1]*y + h[2]) / d, (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
	def matchFeatures(self, desc1, desc2):
		'''
		Input:
			desc1 -- the feature descriptors of image 1 stored in a numpy array,
				dimensions: rows (number of key points) x
				columns (dimension of the feature descriptor)
			desc2 -- the feature descriptors of image 2 stored in a numpy array,
				dimensions: rows (number of key points) x
				columns (dimension of the feature descriptor)
		Output:
			features matches: a list of cv2.DMatch objects
				How to set attributes:
					queryIdx: The index of the feature in the first image
					trainIdx: The index of the feature in the second image
					distance: The distance between the two features
		'''
		matches = []
		# feature count = n
		assert desc1.ndim == 2
		# feature count = m
		assert desc2.ndim == 2
		# the two features should have the type
		assert desc1.shape[1] == desc2.shape[1]

		if desc1.shape[0] == 0 or desc2.shape[0] == 0:
			return []

		# TODO 7: Perform simple feature matching.  This uses the SSD
		# distance between two feature vectors, and matches a feature in
		# the first image with the closest feature in the second image.
		# Note: multiple features from the first image may match the same
		# feature in the second image.
		# TODO-BLOCK-BEGIN
		distance = scipy.spatial.distance.cdist(desc1, desc2, 'euclidean')
		has_choose = np.zeros(np.shape(desc1)[0])
		can_choose = np.ones(np.shape(desc2)[0])
		epsilon = 1e-9

		for confidence in range(5, 10):
			for i, ssd in enumerate(distance):
				if has_choose[i]:
					continue
				index = np.argsort(ssd)
				for cnt in range(index.shape[0] - 1):
					if can_choose[index[cnt]] and ((ssd[index[cnt]] + epsilon) / (ssd[index[cnt + 1]] + epsilon) < (confidence / 10)):
						match = cv2.DMatch()
						match.queryIdx = i
						match.trainIdx = index[cnt]
						match.distance = ssd[index[cnt]]
						matches.append(match)

						can_choose[index[cnt]] = 0
						has_choose[i] = 1
						break

		return matches
