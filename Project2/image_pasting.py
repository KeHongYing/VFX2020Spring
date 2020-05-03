import sys
import math
import numpy as np
import cv2

# 他用來存圖片的class
class ImageInfo:
	def __init__(self, name, img, position):
		self.name = name
		self.img = img
		self.position = position


# 回傳圖片經過transform之後的邊界值(兩個軸最大最小)
# 這邊的 X 是橫軸
# Y 是縱軸
def imageBoundingBox(img, M):
	"""
	   This is a useful helper function that you might choose to implement
	   that takes an image, and a transform, and computes the bounding box
	   of the transformed image.
	   INPUT:
		 img: image to get the bounding box of
		 M: the transformation to apply to the img
	   OUTPUT:
		 minX: int for the minimum X value of a corner
		 minY: int for the minimum Y value of a corner
		 minX: int for the maximum X value of a corner
		 minY: int for the maximum Y value of a corner
	"""

	TL = np.array((0, 0, 1))
	TR = np.array((0, img.shape[0]-1, 1))
	BR = np.array((img.shape[1]-1, img.shape[0]-1, 1))
	BL = np.array((img.shape[1]-1, 0, 1))
	p1 = np.dot(M,TL)/np.dot(M,TL)[2]
	p2 = np.dot(M,TR)/np.dot(M,TR)[2]
	p3 = np.dot(M,BR)/np.dot(M,BR)[2]
	p4 = np.dot(M,BL)/np.dot(M,BL)[2]
	minX = min(p1[0],p2[0],p3[0],p4[0])
	maxX = max(p1[0],p2[0],p3[0],p4[0])
	minY = min(p1[1],p2[1],p3[1],p4[1])
	maxY = max(p1[1],p2[1],p3[1],p4[1])

	#TODO-BLOCK-END
	return int(minX), int(minY), int(maxX), int(maxY)


# 拿到大圖的資訊
def getAccSize(ipv):
	"""
	   This function takes a list of ImageInfo objects consisting of images and
	   corresponding transforms and Returns useful information about the accumulated
	   image.
	   INPUT:
		 ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
	   OUTPUT:
		 accWidth: Width of accumulator image(大圖的寬)
		 accHeight: Height of accumulator image(大圖的高)
		 channels: Number of channels in the accumulator image (應該要是 3 或 4) (4是因為他之後會存權重資訊在第四維)
		 width: Width of each image (小圖的寬度，他假設每一張小圖(照片)的寬度都一樣)
		 translation: 回傳一個使大圖的左上角是(0,0)的矩陣
	"""

	# Compute bounding box for the mosaic
	minX = sys.maxsize
	minY = sys.maxsize
	maxX = 0
	maxY = 0
	channels = -1
	width = -1  # Assumes all images are the same width
	M = np.identity(3)
	for i in ipv:
		M = i.position
		img = i.img
		_, w, c = img.shape
		if channels == -1:
			channels = c
			width = w

		minX2,minY2,maxX2,maxY2 = imageBoundingBox(img,M)
		minX = min(minX,minX2)
		maxX = max(maxX,maxX2)
		minY = min(minY,minY2)
		maxY = max(maxY,maxY2)

	# Create an accumulator image
	accWidth = int(math.ceil(maxX) - math.floor(minX))
	accHeight = int(math.ceil(maxY) - math.floor(minY))
	print ('accWidth, accHeight:', accWidth, accHeight)
	translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

	return accWidth, accHeight, channels, width, translation



# 把每張小圖加進大圖 會做blending
def accumulateBlend(img, acc, M, blendWidth, cnt):
	"""
	   INPUT:
		 img: 小圖
		 acc: 大圖
		 M: 小圖的tranformation matrix
		 blendWidth: width of blending function. horizontal hat function (兩側要分幾次降到0 下面有註解)
	   OUTPUT:
		 modify acc with weighted copy of img added where the first
		 three channels of acc record the weighted sum of the pixel colors
		 and the fourth channel of acc records a sum of the weights
	"""

	x,y,z = img.shape  # 小圖的 高 寬 厚
	acc_x = acc.shape[1]
	acc_y = acc.shape[0]
	clen = 3 #modify!!!

	# 不是直接把小圖加進去，而是作一些操作再加進去
	a_chan = np.ones((x,y))  # 小圖的比重
	pre_warp = np.zeros((x,y,z+1))  #原本的小圖乘以比重之後的小圖
	img_add = np.zeros(acc.shape)  # 等等加進大圖的大圖

	# 這邊是讓要加的小圖 在左右兩側逐漸消失 消失的比例由 blendwidth來決定
	# 例如 blend width = 5 則會依 0.8 0.6 0.4 0.2 0 這樣遞減
	for blendVal in range(blendWidth):
		difference = y-1-blendVal
		newVal = x * [float(blendVal)/blendWidth]
		a_chan[:,difference] = newVal
		a_chan[:,blendVal] = newVal

	# 做某種正規化 雖然我不知道為什麼要這樣.. 可能跟不同圖片會是不同亮暗有關?
	for chan in range(clen):
		img[:,:,chan] = (img[:,:,chan]-np.min(img[:,:,chan]))/float(np.max(img[:,:,chan])-np.min(img[:,:,chan]))*255
		img = np.nan_to_num(img)  # 這行是我自己加的 怕會有nan

	for chan in range(clen):
		# 比重 乘以 原本的圖片
		pre_warp[:,:,chan] = a_chan * img[:,:,chan]

	# 最後一層就是原本圖片的比重 (每個pixel乘以的某個0~1的值)
	pre_warp[:, :, -1] = a_chan

	# 用cv2做tranform
	for chan in range(clen + 1):
		img_add[:, :, chan] = cv2.warpPerspective(pre_warp[:, :, chan], M, flags=cv2.INTER_NEAREST, dsize=(acc_x, acc_y))

	p = np.clip(0.5 + (-1) * ((np.var(acc[:, :, :3], axis = 2) < 200) & (np.sum(acc[:, :, :3], axis = 2) < 50)) + 0.5 * ((np.var(img_add[:, :, :3], axis = 2) < 1000) & (np.sum(img_add[:, :, :3], axis = 2) < 300)), 0, 1)
	p = np.dstack((p, p, p, p))
	acc += acc * (p - 1) + img_add * (1 - p)
	#acc += img_add
	cv2.imwrite(f"test_{cnt}.jpg", acc)

# 這個function要做的是把圖片貼上大圖
def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
	'''
	ipv 是他的class 的list
	tranlation 是 那個使大圖的左上變(0,0)的矩陣
	blendwidth 是 hat function 的 寬 (可以看一下accumulateBlend函式)
	accwidth 是大圖的寬
	accheight 是大圖的高
	channels 就是多少channel
	'''
	acc = np.zeros((accHeight, accWidth, channels + 1))
	# Add in all the images
	M = np.identity(3)
	for cnt, i in enumerate(ipv):
		M = i.position
		img = i.img

		# 這邊做dot是因為圖片做自己的transform之後還要使大圖左上角對齊(0,0)
		M_trans = translation.dot(M)
		
		accumulateBlend(img, acc, M_trans, blendWidth, cnt)

	return acc


# 把所有小圖貼到大圖之後，因為是累加貼上，所以要對權重normalize讓他們變回0-255之類的
def normalizeBlend(acc):
	"""
	   INPUT:
		 acc: input image whose alpha channel (4th channel) contains
		 normalizing weight values
	   OUTPUT:
		 img: image with r,g,b values of acc normalized
	"""
	shape = acc.shape
	img = np.zeros((shape[0],shape[1],4),dtype=np.uint8)

	for x in range(shape[0]):
		for y in range(shape[1]):
			for z in range(3): #modify
				num = float(acc[x,y,z])
				den = float(acc[x,y,3])

				if den != 0: img[x,y,z] = int(num/den)
				else: img[x,y,z] = 0
				

			img[x,y,3] = 1
	return img

# 修正用的參數 第一張小圖跟最後一張小圖的座標資訊
def getDriftParams(ipv, translation, width):
	# Add in all the images
	M = np.identity(3)
	for count, i in enumerate(ipv):
		if count != 0 and count != (len(ipv) - 1):
			continue

		M = i.position

		M_trans = translation.dot(M)

		p = np.array([0.5 * width, 0, 1])
		p = M_trans.dot(p)

		# First image
		if count == 0:
			x_init, y_init = p[:2] / p[2]
		# Last image
		if count == (len(ipv) - 1):
			x_final, y_final = p[:2] / p[2]

	return x_init, y_init, x_final, y_final


# 計算修正用的affine matrix
def computeDrift(x_init, y_init, x_final, y_final, width):
	A = np.identity(3)
	drift = (float)(y_final - y_init)
	# We implicitly multiply by -1 if the order of the images is swapped...
	length = (float)(x_final - x_init)
	A[0, 2] = -0.5 * width
	# Negative because positive y points downwards
	A[1, 0] = -drift / length

	return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
	"""
	   INPUT: 先把每張照片 還有他們對應的motion model 用他的class存起來(在上面) 塞進list
		 ipv: list of input images and their relative positions in the mosaic
		 blendWidth: 是 blending hat function 的 寬 (可以看一下pasteImages 跟 accumulateBlend 函式)
		 跟圖片兩側逐漸消失的比率有關
	   OUTPUT:
		 croppedImage: final mosaic created by blending all images and
		 correcting for any vertical drift  (應該就是最後的圖片了)
	"""

	##### 拼接 #####

	# 先拿到大圖(拼接後的圖)的資訊
	accWidth, accHeight, channels, width, translation = getAccSize(ipv)
	# 然後用pasteImages把小圖一張張貼上
	acc = pasteImages(
		ipv, translation, blendWidth, accWidth, accHeight, channels
	)
	# 貼上去之後數值應該會超出界線 還要押回來
	compImage = normalizeBlend(acc)  # 這個東西可能已經可以用了 後面是在做修正
	cv2.imwrite("normalize.jpg", compImage)
	
	##### 修正 #####
	
	# Determine the final image width
	outputWidth = (accWidth - width) if is360 else accWidth

	# 這邊是做修正 上課講的 因為可能拚一拚會歪掉
	# (這邊我還沒怎麼看)

	x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
	# Compute the affine transform
	A = np.identity(3)

	if is360: A=computeDrift(x_init,y_init,x_final,y_final,width)

	if A_out is not None:
		A_out[:] = A

	# Warp and crop the composite
	croppedImage = cv2.warpPerspective(
		compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
	)

	return croppedImage
