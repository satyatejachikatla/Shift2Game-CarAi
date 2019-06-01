import tensorflow as tf
import numpy as np 

def conv2d_layer(in_layer,kernel_shape,kernel_count,name,**kwargs):
	"""
	Convolutional Layer with Auto weights initilization.
	"""
	with tf.name_scope(name):
		if len(in_layer.shape.as_list()) != 4 :
			raise Exception('Error in the input layer shape , shape recivedis '+ str(in_layer.shape.as_list()))

		# If input is of shape [None,28,28,3] , 3 will indicate the inputs channel dimensions. This will allow for any [x,y] kernel with 3 input channels
		kernels_shape = kernel_shape + [in_layer.shape.as_list()[-1]] 

		w = tf.Variable(tf.truncated_normal([kernels_shape[0],kernels_shape[1],kernels_shape[2],kernel_count]))

		# Change the initialization to other if required for bias , bias is of the kernel_count shape.

		c = tf.nn.conv2d( in_layer , w , **kwargs )

		# Comment if not required bias for a conv outpput. Some cases it will be helpfull
		b = tf.Variable(tf.constant(0., shape=c.shape.as_list()[1:]))

		h = c + b

		'''
		# debug prints
		print('Input shape:', in_layer.shape.as_list() )
		print('Kernels shape:', kernels_shape )
		print('No. of Kernels:', kernel_count )
		print('conv2d output shape:', c.shape.as_list() )
		print('Biases output shape:', b.shape.as_list() )
		print('Convolution output shape:', h.shape.as_list() )
		'''

		return w , b , h

def fully_connected_layer(in_layer,output_nodes_count,name):
	"""
	Fully Connected Layer with Auto weights initilization.
	"""

	with tf.name_scope(name):
		# No. of nodes in the input layer
		input_nodes_count = np.prod(in_layer.shape.as_list()[1:])

		# Flatetened input layer
		in_layer_flattened = tf.reshape(in_layer,[-1]+[input_nodes_count])
		w = tf.Variable(tf.truncated_normal([input_nodes_count,output_nodes_count]))
		
		# Biases weights initialization
		b = tf.Variable(tf.constant(0., shape=[output_nodes_count]))

		# Intermediate output 

		h_ = tf.matmul(in_layer_flattened , w)

		# Output of the fully cunnected.
		h = h_ + b

		'''
		# debug prints
		print('Input shape:', in_layer.shape.as_list() )
		print('in_layer_flattened shape:', in_layer_flattened.shape.as_list() )
		print('Weights shape:', w.shape.as_list() )
		print('Biases output shape:', b.shape.as_list() )
		print('intermediate output shape:', h_.shape.as_list() )
		print('Output shape:', h.shape.as_list() )
		'''

		return w , b , h

def alexnet(input_shape,output_classes=1):

	# Network Structure

	learning_rate = tf.placeholder(tf.float32)
	rc = tf.placeholder(tf.float32)

	x = tf.placeholder(tf.float32, shape=[None]+input_shape , name = 'input')
	y = tf.placeholder(tf.float32, shape=[None,output_classes] , name = 'expected_output')

	w1 , b1, l1 = conv2d_layer( x , kernel_shape=[11,11] , kernel_count=96 ,  name='conv1' , strides=[1,4,4,1] , padding='SAME' )
	r1 = tf.nn.relu(l1)
	p1 = tf.nn.max_pool(r1, ksize = [1,3,3,1], strides = [1,2,2,1], padding='SAME')
	n1 = tf.nn.local_response_normalization(p1)

	w2 , b2, l2 = conv2d_layer( n1 , kernel_shape=[5,5] , kernel_count=256 , name='conv2' , strides=[1,1,1,1] , padding='SAME')
	r2 = tf.nn.relu(l2)
	p2 = tf.nn.max_pool(r2, ksize = [1,3,3,1], strides = [1,2,2,1], padding='SAME')
	n2 = tf.nn.local_response_normalization(p2)

	w3 , b3, l3 = conv2d_layer( n2 , kernel_shape=[3,3] , kernel_count=384 , name='conv3' , strides=[1,1,1,1] , padding='SAME' )
	r3 = tf.nn.relu(l3)

	w4 , b4, l4 = conv2d_layer( r3 , kernel_shape=[3,3] , kernel_count=384 , name='conv3' , strides=[1,1,1,1] , padding='SAME' )
	r4 = tf.nn.relu(l4)

	w5 , b5, l5 = conv2d_layer( r4 , kernel_shape=[3,3] , kernel_count=384 , name='conv5' , strides=[1,1,1,1] , padding='SAME' )
	r5 = tf.nn.relu(l5)
	p5 = tf.nn.max_pool(r5, ksize = [1,3,3,1], strides = [1,2,2,1], padding='SAME')
	n5 = tf.nn.local_response_normalization(p5)

	w6 , b6, l6 = fully_connected_layer( n5 , 4096 , name = 'fc1')
	h6 = tf.nn.tanh(l6)
	d6 = tf.nn.dropout(h6,0.5)

	w7 , b7, l7 = fully_connected_layer( d6 , 4096 , name = 'fc2')
	h7 = tf.nn.tanh(l7)
	d7 = tf.nn.dropout(h7,0.5)

	w8 , b8, l8 = fully_connected_layer( d7 , output_classes , name = 'fcl' )
	y_ = tf.nn.softmax(l8)

	#loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_))
	loss0 = tf.losses.mean_squared_error(labels=y, predictions=y_)
	#modulo_distance = tf.reduce_mean(tf.abs(y-y_))

	# L2 regularization on weights of FC
	reg_loss = rc*(tf.nn.l2_loss(w6)+tf.nn.l2_loss(w7)+tf.nn.l2_loss(w8))

	loss = loss0 + reg_loss

	# Optimizer

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	#optimizer = tf.train.MomentumOptimizer(learning_rate).minimize(loss0)

	# Saver for the weights and biases
	saver = tf.train.Saver({'w1':w1,'b1':b1,'w2':w2,'b2':b2,'w3':w3,'b3':b3,'w4':w4,'b4':b4,'w5':w5,'b5':b5,'w6':w6,'b6':b6,'w7':w7,'b7':b7,'w8':w8,'b8':b8})


	class model_obj():
		def __init__(self):
			pass

	model = model_obj()

	model.x          = x
	model.y          = y
	model.y_         = y_
	model.saver      = saver
	model.optimizer  = optimizer
	model.loss       = loss
	model.learning_rate = learning_rate
	model.rc         = rc

	'''
	tf.summary.histogram('Loss' , model.x)
	tf.summary.histogram('Loss' , model.y)
	tf.summary.histogram('Loss' , model.y_ )
	tf.summary.histogram('Loss' , model.loss)
	tf.summary.histogram('Loss' , model.loss0)
	tf.summary.histogram('Loss' , model.reg_loss)
	'''

	return model 
