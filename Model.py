import tensorflow as tf, numpy as np

def conv_layer(in_layer,out_chan,size,sigma=0.01,b=0.0,strd=[1,1,1,1],pool=True):
    """
    Convolutional Layer with Max Pooling and Local Response Normalization
    """
    in_chan = in_layer.shape.as_list()[3]
    w = tf.Variable(tf.truncated_normal([size,size,in_chan,out_chan],stddev=sigma))
    b = tf.Variable(tf.constant(b, shape=[out_chan]))
    h_ = tf.nn.conv2d(in_layer, w, strides=strd,padding='VALID')+b
    p = tf.nn.max_pool(h_,ksize = [1,4,4,1], strides = [1,1,1,1], padding='VALID')
    h = tf.nn.relu(p)
    n = tf.nn.local_response_normalization(h, depth_radius=min(4,out_chan-2))
    if pool:
        return w,b,h,n
    h = tf.nn.relu(h_)
    n1 = tf.nn.local_response_normalization(h,depth_radius=min(4,out_chan-2))
    return w,b,h,n1



def conn_layer(in_layer,out_nodes,op_layer=False,sigma=0.01,b=0.0):
    """
    Fully Connected Layer
    """
    i_s = in_layer.shape.as_list()
    #print(i_s)
    in_layer2 = in_layer
    if len(i_s) > 2:
        in_layer2 = tf.reshape(in_layer,[-1,i_s[1]*i_s[2]*i_s[3]])
        w = tf.Variable(tf.truncated_normal([i_s[1]*i_s[2]*i_s[3],out_nodes],stddev=sigma))
    else:
        w = tf.Variable(tf.truncated_normal([i_s[-1],out_nodes],stddev=sigma))
    b = tf.Variable(tf.constant(b, shape=[out_nodes]))
    h = tf.matmul(in_layer2,w)+b
    if not op_layer:
        h = tf.nn.relu(h)
    r = tf.nn.l2_loss(w)
    return w,b,h,r


def my_model(width,height,output_classes):
    
    # Hyper Parameters

    learning_rate = tf.placeholder(tf.float32)
    regularization_constant = tf.placeholder(tf.float32)

    """
    The architecture
    """
    x = tf.placeholder(tf.float32, shape=[None,width,height,1])
    y = tf.placeholder(tf.float32, shape=[None,output_classes])

    '''
    # Need to check the correct numbers
    w1,b1,h1,n1 = conv_layer(x,96,11)
    w2,b2,h2,n2 = conv_layer(n1,384,3)
    w3,b3,h3,n3 = conv_layer(n2,384,3)
    w4,b4,h4,n4 = conv_layer(n3,256,3)

    w5,b5,h5,r5 = conn_layer(n4,4096)
    h5_drop     = tf.nn.dropout(h5,0.5)

    w6,b6,h6,r6 = conn_layer(h5_drop,4096)
    h6_drop = tf.nn.dropout(h6,0.5)

    w7,b7,y_,r7 = conn_layer(h6_drop,output_classes,op_layer=True)
    '''

    w1,b1,h1,n1 = conv_layer(x,64,16)
    w2,b2,h2,n2 = conv_layer(n1,32,8)
    w3,b3,h3,n3 = conv_layer(n2,16,16)
    w4,b4,h4,r4 = conn_layer(n3,1024)
    h4_drop = tf.nn.dropout(h4,0.5)
    w5,b5,h5,r5 = conn_layer(h4_drop,512)
    h5_drop = tf.nn.dropout(h5,0.5)
    w6,b6,y_,r6 = conn_layer(h5_drop,output_classes,op_layer=True)

    # Loss function
    #loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_))
    loss0 = tf.losses.mean_squared_error(labels=y, predictions=y_)
    modulo_distance = tf.reduce_mean(tf.abs(y-y_))
    reg = r4+r5+r6
    loss = loss0 + regularization_constant*reg + modulo_distance

    # Optimizer

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    #optimizer = tf.train.MomentumOptimizer(learning_rate).minimize(loss0)

    # Saver for the weights and biases
    saver = tf.train.Saver({'w1':w1,'b1':b1,'w2':w2,'b2':b2,'w3':w3,'b3':b3,'w4':w4,'b4':b4,'w5':w5,'b5':b5,'w6':w6,'b6':b6})

    class inter_obj():
        def __init__(self):
            pass

    model = inter_obj()

    model.x          = x
    model.y          = y
    model.y_         = y_
    model.saver      = saver
    model.optimizer  = optimizer
    model.loss       = loss
    model.loss0       = loss0
    model.learning_rate = learning_rate
    model.regularization_constant = regularization_constant

    return model 

