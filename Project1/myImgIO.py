import numpy as np
from PIL import Image

#def read_image(path, resize = False):
#	from rawkit.raw import Raw
#	LDR_image = []
#
#	for p in path:
#		raw_image = Raw(p)
#		buffered_image = np.array(raw_image.to_buffer())
#		img = np.array(Image.frombytes("RGB", (raw_image.metadata.width, raw_image.metadata.height), buffered_image))
#		if resize:
#			img = np.array(Image.fromarray(img).resize(resize))
#
#		LDR_image.append(img)
#	
#	return np.array(LDR_image)

def read_image_rawpy(path, resize = False):
	import rawpy

	LDR_img = []
	for p in path:
		img = rawpy.imread(p).postprocess(gamma = None, no_auto_bright = True, use_camera_wb = True, half_size = False)
	
		if resize:
			img = np.array(Image.fromarray(img).resize(resize))
	
		LDR_img.append(img)

	return np.array(LDR_img)

def output_image(img, filename):
	print(img.shape)
	Image.fromarray(img.astype(np.uint8)).convert("RGB").save(filename)

def get_exposure_time(path):
	import exifread
	T = []
	for p in path:
		f = open(p, "rb")
		T.append(float(str(exifread.process_file(f)["EXIF ExposureTime"])))
	
	print(T)
	return T
