import cv2
import numpy as np 
import tensorflow as tf
from Model import my_model
from VideoCapture import grab_screen
import Actions
import time

from Parameters import monitor , save_training_data_file , monitor_reduction_factor , training_epochs , lr 

print('Loading training_data ...')
training_data = list(np.load(save_training_data_file,allow_pickle=True))
print('Successfully loaded !!')

WIDTH = monitor['width']//monitor_reduction_factor
HEIGHT = monitor['height']//monitor_reduction_factor
EPOCHS = training_epochs
LR = lr
MODEL_NAME = './Models/shift-car.model'
OUTPUT_CLASSES = 3

# Init model
model = my_model(WIDTH,HEIGHT,OUTPUT_CLASSES)

def show_results():

	# CV2 display window and its control
	for img , output in training_data:
		
		X = np.array([i for i in [img]]).reshape(-1,WIDTH,HEIGHT,1)

		# Value of the input in that frame
		predicted = model.y_.eval({model.x:X})[0]
		print('Actual:',output,'Predicted' , predicted , 'Difference' , np.absolute(np.subtract(output,predicted)))

		# image at that frame
		img = cv2.resize(img,(monitor['width'] ,monitor['height']) )
		
		# Display image
		cv2.imshow('Captured Video', img)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			exit()

def multi_hot_to_action(output):

	Actions.left(output[0])
	Actions.straight(output[1])
	Actions.right(output[2])

def run_game():
	print('Starting in ')
	for i in range(5,0,-1):
		print(i)
		time.sleep(1)

	while True:
		img = grab_screen((monitor['left'],monitor['top'],monitor['width'],monitor['height']))
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img = cv2.resize(img, (monitor['width']//monitor_reduction_factor ,monitor['height']//monitor_reduction_factor))
		X = np.array([i for i in [img]]).reshape(-1,WIDTH,HEIGHT,1)
		predicted = model.y_.eval({model.x:X})[0]
		multi_hot_to_action(predicted)
		print(np.round(predicted))


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
	# Load model
	print('Loaded the previous model ....')
	model.saver.restore(sess,MODEL_NAME)
	print('Successfully loaded !!')

	a = input('Enter option:')
	if a == 'a':
		show_results()
	elif a == 'b':
		run_game()