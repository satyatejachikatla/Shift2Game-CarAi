import cv2
import numpy as np

from Parameters import roi_vertices , minimum_lenth , maximum_gap , canny_threshold1 , canny_threshold2

def draw_lines(img,lines):
	'''
	Draws white lines
	'''
	for line in lines:
		coords = line[0]
		cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)

def roi(img):
	'''
	Region of intrest
	'''
	#blank mask:
	mask = np.zeros_like(img)
	# fill the mask
	cv2.fillPoly(mask, roi_vertices, 255)
	# now only show the area that is the mask
	masked = cv2.bitwise_and(img, mask)

	return masked


def process_img(original_image):

	# Heavy tinkering required

	# Convert image to Black and White
	processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

	# Draw Canny Edges
	processed_img = cv2.Canny(processed_img, threshold1=canny_threshold1, threshold2=canny_threshold2)

	# Gause Blure for next steps - Detect Edges Thicker
	processed_img = cv2.GaussianBlur(processed_img,(5,5),0)

	# Curtting down to Region of intrest
	processed_img = roi(processed_img)

	# Finding lines in the Cany filter 

	# Lines are given from below algo
	lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180 , np.array([]) , minimum_lenth , maximum_gap)

	# If no lines are found
	if type(lines) == type(None):
		lines = []

	# Draws the above lines
	draw_lines(processed_img,lines)

	return processed_img

