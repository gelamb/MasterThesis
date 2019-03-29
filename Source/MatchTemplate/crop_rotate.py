import cv2
import numpy as np


def resize(img, scale_percent):

	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
	return resized


def rotate_bound(image, angle):
	# grab the dimensions of the image and then determine the center
	(h, w) = image.shape[:2]
	(cX, cY) = (w // 2, h // 2)

	# grab the rotation matrix (applying the negative of the
	# angle to rotate clockwise), then grab the sine and cosine
	# (i.e., the rotation components of the matrix)
	M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])

	# compute the new bounding dimensions of the image
	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))

	# adjust the rotation matrix to take into account translation
	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY

	# perform the actual rotation and return the image
	rotated = cv2.warpAffine(image, M, (nW, nH))
	return rotated


def crop_rotate(image, degree):

	(h, w) = image.shape[:2]
	rotated = rotate_bound(image, degree)
	(h_r, w_r) = rotated.shape[:2]
	# diff_h = h_r - h
	# diff_w = w_r - w
	# cropped = rotated[diff_h:-diff_h, diff_w:-diff_w]
	return rotated


# image = cv2.imread("/Users/leonardotanzi/Desktop/MasterThesis/Templates/Bacino_left.jpg")
# crop_rotate(image, 10)
