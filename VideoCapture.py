'''
Capture frames from the screen.
'''

import cv2
import numpy as np 
import ModifyCapture
import time

# The screen part to capture
from Parameters import monitor , monitor_reduction_factor

# Done by Frannecklp

import win32gui, win32ui, win32con, win32api

def grab_screen(region=None):
	'''
	# Done by Frannecklp
	# Copied from sentdex website pythonprogramng.net
	'''

	hwin = win32gui.GetDesktopWindow()

	if region:
			left,top,x2,y2 = region 
			width = x2 - left + 1
			height = y2 - top + 1
	else:
		width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
		height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
		left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
		top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

	hwindc = win32gui.GetWindowDC(hwin)
	srcdc = win32ui.CreateDCFromHandle(hwindc)
	memdc = srcdc.CreateCompatibleDC()
	bmp = win32ui.CreateBitmap()
	bmp.CreateCompatibleBitmap(srcdc, width, height)
	memdc.SelectObject(bmp)
	memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
	signedIntsArray = bmp.GetBitmapBits(True)
	img = np.fromstring(signedIntsArray, dtype='uint8')
	img.shape = (height,width,4)

	srcdc.DeleteDC()
	memdc.DeleteDC()
	win32gui.ReleaseDC(hwin, hwindc)
	win32gui.DeleteObject(bmp.GetHandle())

	# This swaps the blue shades with green shade of image. If required conversion uncomment.
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	return img 

def image_capture_video():
	'''
	Use this function only for analysis.
	'''
	last_time = time.time()
	#Video Loop
	while True:

		img = grab_screen((monitor['left'],monitor['top'],monitor['width'],monitor['height']))

		#Comment below when not required.
		#Hogs the Frames

		#Modifications to the captured images

		#img = ModifyCapture.process_img(img)

		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img = cv2.resize(img, (monitor['width']//monitor_reduction_factor,monitor['height']//monitor_reduction_factor))

		# To view the reduced image bigger
		img = cv2.resize(img, (monitor['width'],monitor['height']))
		

		# Frame Rate info
		print('Loop took {} seconds'.format(time.time()-last_time))
		last_time = time.time()

		# CV2 display window and its control

		cv2.imshow('Captured Video', img)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
		

if __name__ == '__main__':
	# use for only analysis
	image_capture_video()


