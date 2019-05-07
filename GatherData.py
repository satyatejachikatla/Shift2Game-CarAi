import cv2
import numpy as np 
from RecordKeys import key_check , VK_BACK as PAUSE_BUTTON, VK_RETURN as PLAY_BUTTON , VK_ESCAPE as STOP_BUTTON, VK_LEFT, VK_UP , VK_RIGHT
from VideoCapture import grab_screen
from Parameters import monitor , save_training_data_file , monitor_reduction_factor , frames_to_save_after
import os
import time

def keys_to_multihot(keys):
	# [ VK_LEFT , VK_UP , VK_RIGHT ]

	output = [ 0 , 0 , 0 ]

	if VK_LEFT in keys:
		output[0] = 1
	if VK_UP in keys:
		output[1] = 1
	if VK_RIGHT in keys:
		output[2] = 1

	return output

if os.path.isfile(save_training_data_file):
	print('File Exists , loading previous data!')
	training_data = list(np.load(save_training_data_file,allow_pickle=True))
else:
	print('File does not exit , starting fresh')
	training_data = []

def record_data():
	'''
	Captures data into save files.
	'''

	print('Prepare to start playing in ...')
	for i in range(10,10,-1):
		print(i)
		time.sleep(1)

	#last_time = time.time()
	print('Started Recording Data')
	time.sleep(1)
	
	# Unkown Reason why key_check is giving initially backspace alwyas. clearning it with the below
	key_check()
	
	pause_flag = False
	#Video Loop
	while True:
		
		# Gathering info from environment
		img = grab_screen((monitor['left'],monitor['top'],monitor['width'],monitor['height']))
		keys = key_check()

		#Pausing Logic
		if PAUSE_BUTTON in keys:
			if pause_flag == False:
				print('WARNING!! Data collection paused')
			pause_flag = True
		elif PLAY_BUTTON in keys:
			if pause_flag == True:
				print('Resuming Data Collection')
			pause_flag = False
		elif STOP_BUTTON in keys:
			print('Ending Gathering Data')
			break

		if pause_flag:
			continue

		# Cleaning up the data
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img = cv2.resize(img, (monitor['width']//monitor_reduction_factor ,monitor['height']//monitor_reduction_factor))

		output = keys_to_multihot(keys)

		# Adding it to the data list

		training_data.append([img,output])

		'''
		# Frame Rate info, uncomment to see
		print('Loop took {} seconds'.format(time.time()-last_time))
		last_time = time.time()
		'''

		if len(training_data) % frames_to_save_after == 0:
			print('Saving .....',len(training_data))
			np.save(save_training_data_file,training_data)

def show_data():

	# CV2 display window and its control
	for img , output in training_data:
		
		# Value of the input in that frame
		print(output)

		# image at that frame
		img = cv2.resize(img,(monitor['width'] ,monitor['height']) )
		
		# Display image
		cv2.imshow('Captured Video', img)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			exit()

if __name__ == '__main__':
	# Command line to make life easier
	import argparse

	class CommandLine:
		def __init__(self):
			parser = argparse.ArgumentParser(description = "Description for my parser")
			parser.add_argument("-r", "--record_data", help = "Records Data", required = False, default = "" , action = 'store_true')
			parser.add_argument("-s", "--show_data", help = "Shows Data", required = False, default = "", action= 'store_true')

			argument = parser.parse_args()

			if argument.record_data:
				record_data()
			if argument.show_data:
				show_data()
	app = CommandLine()
