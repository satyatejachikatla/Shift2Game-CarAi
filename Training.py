import numpy as np
from Model import my_model
from Parameters import monitor , normalized_training_data_file , training_epochs , lr , monitor_reduction_factor
import tensorflow as tf

WIDTH = monitor['width']//monitor_reduction_factor
HEIGHT = monitor['height']//monitor_reduction_factor
LR = lr
EPOCHS = training_epochs
MODEL_NAME = './Models/shift-car.model'
BATCH_SIZE = 256
OUTPUT_CLASSES = 3
RC = 0.01

# Loading model
model = my_model(WIDTH,HEIGHT,OUTPUT_CLASSES)

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

# Defining batching scheme  for the data.

def serial_batch_selector(data,batch_size,batch_index):

	data_size = len(data)
	if data_size//batch_size*(batch_index+1) < data_size:
		return data[data_size//batch_size*batch_index:data_size//batch_size*(batch_index+1)]
	else:
		return data[data_size//batch_size*batch_index:]

# Training step

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

	sess.run(tf.global_variables_initializer())

	try:
		print('Loaded the previous session ....')
		model.saver.restore(sess,MODEL_NAME)
	except:
		print('Could not load , continuing fresh start .....')

	for e in range(EPOCHS):
		epoch_loss =0
		for batch_index in range(len(X)//BATCH_SIZE):

			epoch_x , epoch_y = serial_batch_selector(X,BATCH_SIZE,batch_index) , serial_batch_selector(Y,BATCH_SIZE,batch_index)
			_ , l = sess.run([model.optimizer,model.loss], feed_dict = {model.x:epoch_x , model.y:epoch_y , model.learning_rate:LR , model.regularization_constant:RC})
			epoch_loss += l

		'''
		# Need to change this
		acc = tf.equal(tf.argmax(y_predicted,1),tf.argmax(y,1))
		acc = tf.reduce_mean(tf.cast(acc,'float'))
		acc = acc.eval({x:test_x, y:test_y})
		'''

		print('Epoch:' , e+1 , 'Loss:', epoch_loss)#, 'Acc:', acc)

	model.saver.save(sess,MODEL_NAME)

# tensorboard --logdir=foo:C:/Users/H/Desktop/ai-gaming/log
