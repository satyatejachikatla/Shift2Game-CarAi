'''
Paramteres to modify for testing.
'''
import numpy as np


# ----------------------------------------------------------------------------- #

# Frams to save after

frames_to_save_after = 500

# Training Data file names.

#save_training_data_file = 'TrainingData/' + 'training_data.npy'
save_training_data_file = 'TrainingData/' + 'nurburgring_without_green_line.npy'
normalized_training_data_file = save_training_data_file + '_normalized.npy'

# Training Parameters

training_epochs = 10
lr = 1e-5

# ----------------------------------------------------------------------------- #

# WARNING DONT TOUCH THIS BLOCK UNLESS REQUIRED

# The screen part to capture
monitor = {"top": 40, "left": 40, "width": 700, "height": 550}

# Decreases the monitor window size by this factor
monitor_reduction_factor = 10

# ----------------------------------------------------------------------------- #

# Analysis with ModifyCapture

# Region of intrest in the window

roi_box_height_from_bottom = 200

roi_vertices = [
	np.array([
		[0,monitor['height']],
		[0,roi_box_height_from_bottom],
		[monitor['width'],roi_box_height_from_bottom],
		[monitor['width'],monitor['height']]
	], np.int32)]

# Canny thresholds

canny_threshold1 = 200
canny_threshold2 = 300

# Dawn lines parameters

minimum_lenth = 180
maximum_gap = 20

# ----------------------------------------------------------------------------- #

