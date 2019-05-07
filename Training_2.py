import numpy as np
from Model_2 import alexnet
from Parameters import monitor , normalized_training_data_file , training_epochs , lr , monitor_reduction_factor
import tensorflow as tf

WIDTH = monitor['width']//monitor_reduction_factor
HEIGHT = monitor['height']//monitor_reduction_factor
LR = lr
EPOCHS = training_epochs
MODEL_NAME = './Models/shift-car-{}-{}-{}-epochs.model_2'.format(LR, 'alexnetv2',EPOCHS)

# Loading model
model = alexnet(WIDTH,HEIGHT,LR)

# Loading the training data
train_data = np.load(normalized_training_data_file, allow_pickle=True)

# Train / Test Split
train = train_data[:-500]
test = train_data[-500:]

# Reshaping the train data to X Y feed dicts
X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = [i[1] for i in train]

# Reshaping the test data to X Y feed dicts
test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = [i[1] for i in test]

# Training 

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
snapshot_step=1000, show_metric=True, run_id=MODEL_NAME,batch_size=1024)

# tensorboard --logdir=foo:C:/Users/H/Desktop/ai-gaming/log

model.save(MODEL_NAME)