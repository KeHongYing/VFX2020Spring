import numpy as np

Z_min = 0
Z_max = 256
PMAX = 256

def W(z):
	if z <= 0.5 * (Z_min + Z_max):
		return z - Z_min
	else:
		return Z_max - z

def find_E_and_g(Z, exposures, num_pixel, num_photo, E = None, g = None, num_iter = 10, E_for_thread = None, g_for_thread = None, index = None):

	#initialize
	if E == None:
		E = np.ones([num_pixel,]) * 10
	if g == None:
		g = np.ones([PMAX,]) * 10

	for it in range(num_iter): 
		print(f"iteration:{it}")
		# find E
		E = np.array([np.sum(W(Z[ci, cj]) * g[int(Z[ci, cj])] * exposures[cj] for cj in range(num_photo)) / (np.sum([W(Z[ci, cj]) * (exposures[cj] ** 2) for cj in range(num_photo)]) + 1e-8) for ci in range(num_pixel)])

		# find g
		for m in range(PMAX):
			s_em = np.sum(Z == m)
			tot = np.sum([np.sum(E[(Z == m)[:, j]] * exposures[j]) for j in range(num_photo)])
			
			if s_em > 0:
				g[m] = tot / s_em
	
	if E_for_thread is not None and g_for_thread is not None and index is not None:
		E_for_thread[index] = E
		g_for_thread[index] = g

	return E, g


if __name__=="__main__":
	test_images = np.ones([10, 5])
	ee = np.array([1,2,4,6,8])
	E, g = find_E_and_g(test_images, exposures = ee, num_pixel = 10, num_photo = 5, num_iter = 1000)

	print(E)
	print(g)
