import numpy as np
import imageio # used for read the image

######
# METHODS CODE
######
def fourier_transform_2d(img):
	n, m = img.shape
	fourier_spectrum = np.zeros((n,m),dtype=np.complex64)
	for u in range(n):
		for v in range(m):
			for x in range(n):
				for y in range(m):
					f_exp = np.exp(-2j*np.pi*((u*x)/n + (v*y)/m))
					fourier_spectrum[u,v] += img[x,y] * f_exp
			fourier_spectrum[u,v] = fourier_spectrum[u,v]/np.sqrt(n*m)
	return fourier_spectrum

def opt_fourier_transform_2d(img):
	n, m = img.shape

	u, v = np.arange(n), np.arange(m)
	x, y = u.reshape((n,1)), v.reshape((m,1))

	ux, vy = np.multiply(u,x), np.multiply(v,y)
	f_exp_n = np.exp(-2j * np.pi * (ux/n))
	f_exp_m = np.exp(-2j * np.pi * (vy/m))
	fourier_spectrum = f_exp_n.dot(img).dot(f_exp_m)

	return fourier_spectrum/np.sqrt(n*m)

def get_second_peak(fourier_spectrum):
	width, height = fourier_spectrum.shape
	vector = fourier_spectrum.reshape((width*height))
	vector = list(np.unique(np.abs(vector)))
	vector.sort(reverse=True)
	return vector[1]

def spectrum_filter(fourier_spectrum,threshold):
	filtered_coefficients = 0
	n,m = fourier_spectrum.shape
	for u in range(n):
		for v in range(m):
			if np.abs(fourier_spectrum[u,v]) < threshold:
				fourier_spectrum[u,v] = 0
				filtered_coefficients += 1
	return fourier_spectrum, filtered_coefficients

def inverse_fourier_transform_2d(fourier_spectrum):
	n, m = fourier_spectrum.shape
	img = np.zeros((n,m),dtype=np.complex64)
	for x in range(n):
		for y in range(m):
			for u in range(n):
				for v in range(m):
					f_exp = np.exp(2j*np.pi*((u*x)/n + (v*y)/m))
					img[x,y] += fourier_spectrum[u,v] * f_exp
			img[x,y] = img[x,y]/np.sqrt(n*m)
	return img

def opt_inverse_fourier_transform_2d(fourier_spectrum):
	n, m = fourier_spectrum.shape

	x, y = np.arange(n), np.arange(m)
	u, v = x.reshape((n,1)), y.reshape((m,1))

	ux, vy = np.multiply(u,x), np.multiply(v,y)
	f_exp_n = np.exp(2j * np.pi * (ux/n))
	f_exp_m = np.exp(2j * np.pi * (vy/m))
	img = f_exp_n.dot(fourier_spectrum).dot(f_exp_m)

	return img/np.sqrt(n*m)
	
######
# MAIN CODE
######
# 1. Reading the inputs
# a. image
filename = str(input()).rstrip()
input_img = imageio.imread(filename)

# b. percentage (float number)
percentage = float(input())

# 2. Computing the Fourrier Transform of the image
fourier_spectrum = opt_fourier_transform_2d(input_img)

# 3. Getting the second peak
p2 = get_second_peak(fourier_spectrum)

# 4. Removing the coefficients for which the Fourier Spectrum is below
# T% of the second peak, that is, |F| < p2*T
threshold = percentage*p2
fourier_spectrum, filtered_coefficients = spectrum_filter(fourier_spectrum,threshold)

# 5. Computing the Inverse Fourrier Transform
new_img = opt_inverse_fourier_transform_2d(fourier_spectrum)

# 6. Printing the result
original_mean = input_img.mean()
new_mean = np.abs(new_img).mean()
print('Threshold=%.4f' %(threshold))
print('Filtered Coefficients=%d' %(filtered_coefficients))
print('Original Mean=%.2f' %(original_mean))
print('New Mean=%.2f' %(new_mean))