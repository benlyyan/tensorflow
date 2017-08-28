from __future__ import division,print_function,unicode_literals 

import numpy as np 
import os 
import numpy.random as rnd 

rnd.seed(42)

import matplotlib 
import matplotlib.pyplot as plt 

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

project_root = '.'
chapter_id = 'deep'

def save_fig(fig_id,tight_layout=True):
    path = os.path.join(project_root,'images',chapter_id,fig_id+'.png')
    print("saving figure",fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path,format='png',dpi=300)



def logit(z):
    return 1/(1+np.exp(-z))

z = np.linspace(-5,5,200)

def plot_sigmoid():
    plt.figure()
    plt.plot([-5,5],[0,0],'k-')
    plt.plot([-5,5],[1,1],'k--')
    plt.plot([0,0],[-0.2,1.2],'k-')
    
    
    plt.plot(z,logit(z),'b-',linewidth=2)
    props = dict(facecolor='red',shrink=0.1)
    plt.annotate('saturating',xytext=(3.5,0.7),xy=(5,1),arrowprops=props,fontsize=14,ha='center')
    plt.annotate('saturating',xytext=(-3.5,0.3),xy=(-5,0),arrowprops=props,fontsize=14,ha='center')
    plt.annotate('linear',xytext=(1,0.2),xy=(0,0.5),arrowprops=props,fontsize=14,ha='center')
    plt.grid(True)
    plt.title('sigmoid activation function',fontsize=14)
    plt.axis([-5,5,-0.2,1.2])
    save_fig('sigmoid_function')
    plt.show()

def leaky_relu(z,alpha=0.01):
    return np.maximum(alpha*z,z)

def plot_relu():
    plt.plot(z,leaky_relu(z,0.05),'b-')
    plt.plot([-5,5],[0,0],'k-')
    plt.plot([0,0],[-0.5,5],'k-')
    plt.grid(True)
    props = dict(facecolor='red',shrink=0.1)
    plt.annotate('leaky',xytext=(-4,0.5),xy=(-5,-0.2),arrowprops=props,fontsize=14,ha='center')
    plt.title('leaky relu activation function',fontsize=14)
    plt.axis([-5,5,-0.5,4])
    save_fig('leaky_relu')
    plt.show()

# plot_relu()

def elu(z,alpha=1):
    return np.where(z<0, alpha*(np.exp(z)-1), z)

def plot_elu():
    plt.plot(z,elu(z),'-b',linewidth=2)
    plt.plot([-5,5],[0,0],'k-')
    plt.plot([0,0],[-2,5],'k-')
    plt.grid(True)
    props = dict(facecolor='red',shrink=1)
    plt.title(r"elu function ($\alpha=1$)",fontsize=14)
    plt.axis([-5,5,-2,5])
    save_fig('elu')
    plt.show()
# plot_elu()

from tensorflow.examples.tutorials.mnist import input_data 

mnist = input_data.read_data_sets('/home/ben/Documents/Pyspace/tensorflow/MNIST_data/')

import tensorflow as tf 

from IPython.display import clear_output,Image,display

def strip_consts(graph_def,max_const_size=32):
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size>max_const_size:
                tensor.tensor_content = b"<stripped %d bytes>"%size 
    return strip_def 

def show_graph(graph_def,max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

def dnn():

    tf.reset_default_graph()
    
    n_inputs = 28*28 
    n_hidden1 = 300 
    n_hidden2 = 100 
    n_outputs = 10 
    learning_rate = 0.01 
    x = tf.placeholder(tf.float32,shape=(None,n_inputs),name='x')
    y = tf.placeholder(tf.int64,shape=(None),name='y')
    
    with tf.name_scope('dnn'):
        hidden1 = tf.layers.dense(x, n_hidden1,activation=tf.nn.relu,name='hidden1')
        hidden2 = tf.layers.dense(hidden1, n_hidden2,activation=tf.nn.relu,name='hidden2')
        logits = tf.layers.dense(hidden2, n_outputs,name='outputs')
    
    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
        loss = tf.reduce_mean(xentropy,name='loss')
    with tf.name_scope('train'):
        opimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = opimizer.minimize(loss)
    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    n_epochs = 20
    batch_size = 100 
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(len(mnist.test.labels)//batch_size):
                x_batch,y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op,feed_dict={x:x_batch,y:y_batch})
            acc_train = accuracy.eval(feed_dict={x:x_batch,y:y_batch})
            acc_test = accuracy.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels})
            print(epoch,"train accuracy:,",acc_train,"test accuracy",acc_test)
    
        save_path = saver.save(sess, 'my_model_final.ckpt')


from functools import partial
 

tf.reset_default_graph()

def batch_norm_dnn():
    n_inputs = 28*28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10 
    learning_rate = 0.01 
    momentum = 0.25
    x = tf.placeholder(tf.float32,shape=(None,n_inputs),name='x')
    y = tf.placeholder(tf.int64,shape=(None),name='y')
    is_training = tf.placeholder(tf.bool,shape=(None),name='training')

    with tf.name_scope('dnn'):
        he_init = tf.contrib.layers.variance_scaling_initializer()
        batch_norm_layer = partial(tf.layers.batch_normalization,training=is_training,momentum=0.9)

        dense_layer = partial(tf.layers.dense,kernel_initializer=he_init)
    hidden1 = dense_layer(x,n_hidden1,name='hidden1')
    bn1 = tf.nn.elu(batch_norm_layer(hidden1))

    hidden2 = dense_layer(bn1,n_hidden2,name='hidden2')
    bn2 = tf.nn.elu(batch_norm_layer(hidden2))

    logits_before_bn = dense_layer(bn2,n_outputs,activation=None,name='outputs')
    logits = batch_norm_layer(logits_before_bn)
    ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
        loss = tf.reduce_mean(xentropy,name='loss')

    with tf.name_scope('train'):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
        training_op = optimizer.minimize(loss)

    # alternatively 
    # with tf.name_scope('train'):
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(ops):
            # training_op = optimizer.minimize(loss)

    with tf.name_scope('eval'):
        corret = tf.nn.in_top_k(logits,y, 1)
        accuracy = tf.reduce_mean(tf.cast(corret,tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 20
    batch_size = 200
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(len(mnist.test.labels)//batch_size):
                x_batch,y_batch = mnist.train.next_batch(batch_size)
                sess.run([training_op,ops],feed_dict={x:x_batch,y:y_batch,is_training:True})
            acc_train = accuracy.eval(feed_dict={is_training:False,x:x_batch,y:y_batch})
            acc_test = accuracy.eval(feed_dict={is_training:False,x:mnist.test.images,y:mnist.test.labels})
            print(epoch,"train acc:",acc_train,"test_acc:",acc_test)

        save_path = saver.save(sess, './models/my_model_final_ckpt')

        # reuse the first two layers weights if need 
        with tf.variable_scope("",default_name="",reuse=True):
            weights1 = tf.get_variable("hidden1/kernel")
            weights2 = tf.get_variable("hidden2/kernel")
            print(np.shape(weights1.eval()),np.shape(weights2.eval()))



# batch_norm_dnn()

def sav():
    v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer())
    v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer())
    
    inc_v1 = v1.assign(v1+1)
    dec_v2 = v2.assign(v2-1)
    
    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    
    # Later, launch the model, initialize the variables, do some work, and save the
    # variables to disk.
    with tf.Session() as sess:
      sess.run(init_op)
      # Do some work with the model.
      inc_v1.op.run()
      dec_v2.op.run()
      # Save the variables to disk.
      save_path = saver.save(sess, "/tmp/model.ckpt")
      print("Model saved in file: %s" % save_path)

def test():
    tf.reset_default_graph()
    
    # Create some variables.
    v1 = tf.get_variable("v1", shape=[3])
    v2 = tf.get_variable("v2", shape=[5])
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    
    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
      # Restore variables from disk.
      saver.restore(sess, "/tmp/model.ckpt")
      print("Model restored.")
      # Check the values of the variables
      print("v1 : %s" % v1.eval())
      print("v2 : %s" % v2.eval())
    
# sav()
# test()


# transfer the first layer weight 

def restore_hidden1():
    tf.reset_default_graph()
    weights = tf.get_variable("kernel",shape=[28*28,300],initializer=tf.zeros_initializer())
    init = tf.global_variables_initializer()
    reuse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="hidden1/kernel")

    # reuse_vars_dict = dict([(var.name,var.name) for var in reuse_vars])
    saver = tf.train.Saver({"hidden1/kernel":weights})
    with tf.Session() as sess:
        sess.run(init)
        print(weights.eval())
        saver.restore(sess, "./models/my_model_final_ckpt")
        res = weights.eval()
        print(res)
        print([var.name for var in tf.global_variables()])

    weights1 = tf.get_variable('kernel1',dtype=tf.float32, initializer=tf.constant(res))
    init =  tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(weights1.eval())

# restore the first hedden layer weights
# restore_hidden1()


def clip_norm_dnn():
    tf.reset_default_graph()
    
    n_inputs = 28*28 
    n_hidden1 = 300 
    n_hidden2 = 100 
    n_outputs = 10 
    learning_rate = 0.01 
    momentum = 0.9 

    x = tf.placeholder(tf.float32,shape=(None,n_inputs),name='x')
    y = tf.placeholder(tf.int64,shape=(None),name='y')
    is_training = tf.placeholder(tf.bool,shape=(),name='is_training')

    def max_norm_regularizer(threshold,axes=1,name='max_norm',collection='max_norm'):
        def max_norm(weights):
            clip_weights = tf.assign(weights,tf.clip_by_norm(weights, clip_norm=threshold,axes=axes),name=name)
            tf.add_to_collection(collection, clip_weights)
        return max_norm 

    with tf.name_scope('dnn'):
        my_dense = partial(tf.layers.dense,activation=tf.nn.relu,kernel_regularizer=max_norm_regularizer)

        hidden1 = my_dense(x,n_hidden1,name='hidden1')
        hidden2 = my_dense(hidden1,n_hidden2,name='hidden2')
        logits = my_dense(hidden2,n_outputs,activation=None,name='outputs')

    clip_all_weights = tf.get_collection('max_norm')

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
        loss = tf.reduce_mean(xentropy,name='loss')

    with tf.name_scope('train'):
        optimizer = tf.train.MomentumOptimizer(learning_rate,momentum)
        threshold = 1.0 
        grads_and_vars = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold),var) for grad,var in grads_and_vars]
        training_op = optimizer.apply_gradients(capped_gvs)

    with tf.name_scope('eval'):
        corret = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(corret, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 20
    batch_size = 50
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(len(mnist.test.labels)//batch_size):
                x_batch,y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op,feed_dict={is_training:True,x:x_batch,y:y_batch})
                sess.run(clip_all_weights)
            acc_train = accuracy.eval(feed_dict={is_training:False,x:x_batch,y:y_batch})
            acc_test = accuracy.eval(feed_dict={is_training:False,x:mnist.test.images,y:mnist.test.labels})
            print(epoch,"train acc:",acc_train,"test acc:",acc_test)
        save_path = saver.save(sess, './models/clip_norm_model.ckpt')

# clip_norm_dnn()

def drop_out_dnn():
    tf.reset_default_graph()
    n_inputs = 28*28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10
    x = tf.placeholder(tf.float32,shape=(None,n_inputs),name='x')
    y = tf.placeholder(tf.int64,shape=(None),name='y')
    is_training = tf.placeholder(tf.bool,shape=(),name='is_training')

    initial_learning_rate = 0.1 
    decay_steps = 10000
    decay_rate = 1/10 
    global_step = tf.Variable(0,trainable=False)
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)
    dropout_rate = 0.5
    momentum = 0.9

    with tf.name_scope('dnn'):
        he_init = tf.contrib.layers.variance_scaling_initializer()
        my_dense = partial(tf.layers.dense,
            activation=tf.nn.relu,kernel_initializer=he_init)

        x_drop = tf.layers.dropout(x,dropout_rate,training=is_training)
        hidden1 = my_dense(x_drop,n_hidden1,name='hidden1')
        hidden1_drop = tf.layers.dropout(hidden1,dropout_rate,training=is_training)

        hidden2 = my_dense(hidden1_drop,n_hidden2,name='hidden2')
        hidden2_drop = tf.layers.dropout(hidden2,dropout_rate,training=is_training)
        logits = my_dense(hidden2_drop,n_outputs,activation=None,name='outputs')
    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
        loss = tf.reduce_mean(xentropy,name='loss')

    with tf.name_scope('train'):
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        training_op = optimizer.minimize(loss,global_step=global_step)
    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    n_epochs = 20
    batch_size = 50
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(len(mnist.test.labels)//batch_size):
                x_batch,y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op,feed_dict={is_training:True,x:x_batch,y:y_batch})
            acc_train = accuracy.eval(feed_dict={is_training:False,x:x_batch,y:y_batch})
            acc_test = accuracy.eval(feed_dict={is_training:False,x:mnist.test.images,y:mnist.test.labels})
            print(epoch,"train accuracy:",acc_train,"test_accuracy:",acc_test)
        save_path = saver.save(sess, './models/droupout.ckpt')


# drop_out_dnn()

# gradient clipping with 5 hidden layers 
# demonstrate how to use pretrained layers

def gradient_clip_5hidden_layers():
    tf.reset_default_graph()
    # define params 
    n_inputs = 28*28
    n_hidden1 = 300
    n_hidden2 = 50
    n_hidden3 = 50
    n_hidden4 = 50
    n_hidden5 = 50
    n_outputs = 10
    learning_rate = 0.01
    momentum = 0.9 

    # define input placeholder 
    x = tf.placeholder(tf.float32,shape=(None,n_inputs),name='x')
    y = tf.placeholder(tf.int64,shape=(None),name='y')

    with tf.name_scope('dnn'):
        hidden1 = tf.layers.dense(x, n_hidden1,activation=tf.nn.relu,name='hidden1')
        hidden2 = tf.layers.dense(hidden1, n_hidden2,activation=tf.nn.relu,name='hidden2')
        hidden3 = tf.layers.dense(hidden2, n_hidden3,activation=tf.nn.relu,name='hidden3')
        hidden4 = tf.layers.dense(hidden3, n_hidden4,activation=tf.nn.relu,name='hidden4')
        hidden5 = tf.layers.dense(hidden4, n_hidden5,activation=tf.nn.relu,name='hidden5')
        logits = tf.layers.dense(hidden5, n_outputs,activation=None,name='outputs')

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
        loss = tf.reduce_mean(xentropy,name='loss')


    threshold = 1.0 
    with tf.name_scope('train'):
        # without clipping gradients
        # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        # training_op = optimizer.minimize(loss)

        # with clipping gradients threshold 1.0 
        # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        grads = [(tf.clip_by_value(grad, -threshold, threshold),var) for grad,var in grads_and_vars]
        training_op = optimizer.apply_gradients(grads)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32),name='accuracy')

    init = tf.global_variables_initializer()
    n_epochs = 20
    batch_size = 50
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(len(mnist.test.labels)//batch_size):
                x_batch,y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op,feed_dict={x:x_batch,y:y_batch})

            acc_train = accuracy.eval(feed_dict={x:x_batch,y:y_batch})
            acc_test = accuracy.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels})
            print(epoch,"acc_train: ",acc_train,"acc_test: ",acc_test)

        saver.save(sess, './models/clip_gradient_5hidden_layers.ckpt')

# gradient_clip_5hidden_layers()


# using pretrained model to continue model training 

def restore_graph():
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph('./models/clip_gradient_5hidden_layers.ckpt.meta')
    # for op in tf.get_default_graph().get_operations():
    #     print(op.name)

    x = tf.get_default_graph().get_tensor_by_name('x:0')
    y = tf.get_default_graph().get_tensor_by_name('y:0')
    accuracy = tf.get_default_graph().get_tensor_by_name('eval/accuracy:0')
    training_op = tf.get_default_graph().get_operation_by_name('train/GradientDescent')

    for op in (x,y,accuracy,training_op):
        tf.add_to_collection('important_ops', op)

    x,y,accuracy,training_op = tf.get_collection('important_ops')

    with tf.Session() as sess:
        n_epochs = 20
        batch_size = 50 
        saver.restore(sess, './models/clip_gradient_5hidden_layers.ckpt')
        for epoch in range(n_epochs):
            for iteration in range(len(mnist.test.labels)//batch_size):
                x_batch,y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op,feed_dict={x:x_batch,y:y_batch})

            acc_train = accuracy.eval(feed_dict={x:x_batch,y:y_batch})
            acc_test = accuracy.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels})
            print(epoch,"using pretrained model:acc_train:",acc_train,"acc_test:",acc_test)

        saver.save(sess, "./models/using_pretrained_clip_gradient_5hidden_layers.ckpt")
        for op in tf.get_default_graph().get_operations():
            print(op.name)

# restore_graph()


# replace a hidden layer and continue to train 
# if you don't want to import meta graph,you can use:
# tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="hidden[123]")
# reuse_vars_dict = dict([(var.op.name,var) for var in reuse_vars])
# restore_saver = tf.train.Saver(reuse_vars_dict)
def replace_a_layer_dnn():
    new_n_hidden4 = 20
    n_outputs = 10 
    learning_rate = 0.01 
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph('./models/clip_gradient_5hidden_layers.ckpt.meta')
    x = tf.get_default_graph().get_tensor_by_name('x:0')
    y = tf.get_default_graph().get_tensor_by_name('y:0')

    hidden3 = tf.get_default_graph().get_tensor_by_name('dnn/hidden4/Relu:0')

    with tf.name_scope('dnn'):
        new_hidden4 = tf.layers.dense(hidden3, new_n_hidden4,activation=tf.nn.relu,name='new_hidden4')
        logits = tf.layers.dense(new_hidden4, n_outputs,activation=None,name='new_outputs')

    with tf.name_scope('new_loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
        loss = tf.reduce_mean(xentropy,name='loss')

    with tf.name_scope('new_train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope('new_eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32),name='accuracy')

    init = tf.global_variables_initializer()

    new_saver = tf.train.Saver()

    n_epochs = 20
    batch_size = 50 
    with tf.Session() as sess:
        saver.restore(sess, './models/clip_gradient_5hidden_layers.ckpt')
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(len(mnist.test.labels)//batch_size):
                x_batch,y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op,feed_dict={x:x_batch,y:y_batch})
            acc_train = accuracy.eval(feed_dict={x:x_batch,y:y_batch})
            acc_test = accuracy.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels})
            print(epoch,"acc_train:",acc_train,"acc_test:",acc_test)

        new_saver.save(sess, './models/clip_gradient_5hidden_layers_with_4th_replace.ckpt')

        for op in tf.get_default_graph().get_operations():
            print(op.name)

        print('global variables','--'*40)
        for v in tf.global_variables():
            print(v.name)
        print('trainable variables','--'*40)
        for v in tf.trainable_variables():
            print(v.name)

# replace_a_layer_dnn()


# freezing the lower layers

def freeze_lower_layer_dnn():
    tf.reset_default_graph()
    n_inputs = 28*28
    n_hidden1 = 300
    n_hidden2 = 50
    n_hidden3 = 50
    n_hidden4 = 20
    n_outputs = 10 
    learning_rate = 0.01 
    x = tf.placeholder(tf.float32,shape=(None,n_inputs),name='x')
    y = tf.placeholder(tf.int64,shape=(None),name='y')

    with tf.name_scope('dnn'):
        hidden1 = tf.layers.dense(x, n_hidden1,activation=tf.nn.relu,name='hidden1')
        hidden2 = tf.layers.dense(hidden1, n_hidden2,activation=tf.nn.relu,name='hidden2')
        hidden3 = tf.layers.dense(hidden2, n_hidden3,activation=tf.nn.relu,name='hidden3')
        hidden4 = tf.layers.dense(hidden3, n_hidden4,activation=tf.nn.relu,name='hidden4')
        logits = tf.layers.dense(hidden4, n_outputs,activation=None,name='outputs')

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
        loss = tf.reduce_mean(xentropy,name='loss')
    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32),name='accuracy')

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        # freeze 1,2 layers,train 3,4 layers,every time training 3,4layers
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="hidden[34]|outputs")
        training_op = optimizer.minimize(loss,var_list=train_vars)

    init = tf.global_variables_initializer()
    new_saver = tf.train.Saver()

    # every time restore 1,2,3 layers
    reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="hidden[123]")
    resue_vars_dict = dict([(var.op.name,var) for var in reuse_vars])
    restore_saver = tf.train.Saver(resue_vars_dict)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    epoch = 20
    batch_size = 50 
    with tf.Session() as sess:
        init.run()
        restore_saver.restore(sess, "./models/clip_gradient_5hidden_layers_with_4th_replace.ckpt")
        for epoch in range(epoch):
            for iteration in range(mnist.train.num_examples//batch_size):
                x_batch,y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op,feed_dict={x:x_batch,y:y_batch})
            accuracy_val = accuracy.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels})
            
            print(epoch,"test accuacy:",accuracy_val)    

        save_path = saver.save(sess,"./models/freezing_model.ckpt")


# freeze_lower_layer_dnn()

# learning rate scheduling
def schedule_learn_rate():
    tf.reset_default_graph()
    n_inputs = 28*28
    n_hidden1 = 300
    n_hidden2 = 50
    n_hidden3 = 10
    n_outputs = 10
    x = tf.placeholder(tf.float32,shape=(None,n_inputs),name='x')
    y = tf.placeholder(tf.int64,shape=(None),name='y')

    with tf.name_scope('dnn'):
        hidden1 = tf.layers.dense(x, n_hidden1,activation=tf.nn.relu,name='hidden1')
        hidden2 = tf.layers.dense(hidden1, n_hidden2,activation=tf.nn.relu,name='hidden2')
        hidden3 = tf.layers.dense(hidden2, n_hidden3,activation=tf.nn.relu,name='hidden3')
        logits = tf.layers.dense(hidden3,n_outputs,activation=None,name='outputs')

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
        loss = tf.reduce_mean(xentropy,name='loss')

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32),name='accuracy')

    with tf.name_scope('train'):
        init_learning_rate = 0.1
        decay_steps = 10000
        decay_rate = 0.1

        global_step = tf.Variable(0,trainable=False,name='global_step')
        learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, decay_steps, decay_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        training_op = optimizer.minimize(loss,global_step=global_step)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 5
    batch_size = 50

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples//batch_size):
                x_batch,y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op,feed_dict={x:x_batch,y:y_batch})
            accuracy_val = accuracy.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels})
            print(epoch,"test accuracy:",accuracy_val)
        save_path = saver.save(sess, './models/schedule_learning_rate_model.ckpt')

# schedule_learn_rate()

# l1 and l2 regularization 
# method1 get weights from all layers if you have a few layers
# method2 use dense function with kernel_regularizer argument
# i.e. kernel_regularizer=tf.contrib.layers.l1_regularizer(scale)
# later reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
# loss = tf.add_n([base_loss] + reg_losses, name="loss")

def l_norm_dnn():
    tf.reset_default_graph()
    n_inputs = 28*28
    n_hidden1 = 300
    n_outputs = 10
    learning_rate = 0.01
    x = tf.placeholder(tf.float32,shape=(None,n_inputs),name='x')
    y = tf.placeholder(tf.int64,shape=(None),name='y')

    with tf.name_scope('dnn'):
        hidden1 = tf.layers.dense(x, n_hidden1,activation=tf.nn.relu,name='hidden1')
        logits = tf.layers.dense(hidden1, n_outputs,activation=None,name='outputs')
    w1 = tf.get_default_graph().get_tensor_by_name('hidden1/kernel:0')
    w2 = tf.get_default_graph().get_tensor_by_name('outputs/kernel:0')
    scale = 0.001

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
        base_loss = tf.reduce_mean(xentropy,name='xentropy')
        reg_loss = tf.reduce_sum(tf.abs(w1)+tf.reduce_sum(tf.abs(w2)))
        loss = tf.add(base_loss, scale*reg_loss,name='loss')

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32),name='accuracy')
    learning_rate = 0.01

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    n_epochs = 20
    batch_size = 200

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples//batch_size):
                x_batch,y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op,feed_dict={x:x_batch,y:y_batch})

            acc_eval = accuracy.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels})
            print(epoch,"acc_test:",acc_eval)
        save_path = saver.save(sess,'./models/l1_norm_model.ckpt')

# l_norm_dnn()

# ex dnn

he_init = tf.contrib.layers.variance_scaling_initializer()
def dnn5(inputs,n_hidden_layers=5,n_neurons=100,name=None,activation=tf.nn.elu,initializer=he_init):
    with tf.variable_scope(name,'dnn'):
        for layer in range(n_hidden_layers):
            inputs = tf.layers.dense(inputs,n_neurons,activation=activation,kernel_initializer=initializer,
                name="hidden%d" % (layer+1))
        return inputs 
def ex_8_1():
    n_inputs = 28*28
    n_outputs = 5
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32,shape=(None,n_inputs),name='x')
    y = tf.placeholder(tf.int64,shape=(None),name='y')
    dnn_outputs = dnn5(x)
    logits = tf.layers.dense(dnn_outputs, n_outputs,kernel_initializer=he_init,name='logits')
    y_proba = tf.nn.softmax(logits,name='y_proba')

    learning_rate = 0.01
    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
        loss = tf.reduce_mean(xentropy,name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss,name='training_op')

    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32),name='accuracy')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    x_train1 = mnist.train.images[mnist.train.labels<5]
    y_train1 = mnist.train.labels[mnist.train.labels<5]
    x_val1 = mnist.validation.images[mnist.validation.labels<5]
    y_val1 = mnist.validation.labels[mnist.validation.labels<5]
    x_test1 = mnist.test.images[mnist.test.labels<5]
    y_test1 = mnist.test.labels[mnist.test.labels<5]

    n_epochs = 100
    batch_size = 20

    max_checks_without_progress = 20
    checks_without_progress = 0
    best_loss = np.infty

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            rnd_index = np.random.permutation(len(x_train1))
            for rnd_ind in np.array_split(rnd_index, len(x_train1)//batch_size):
                x_batch,y_batch = x_train1[rnd_ind],y_train1[rnd_ind]
                sess.run(training_op,feed_dict={x:x_batch,y:y_batch})
            loss_val,acc_val = sess.run([loss,accuracy],feed_dict={x:x_val1,y:y_val1})
            if loss_val < best_loss:
                save_path = saver.save(sess, './models/ex_8_1_models.ckpt')
                best_loss = loss_val
                checks_without_progress = 0
            else:
                checks_without_progress +=1
                if checks_without_progress > max_checks_without_progress:
                    print('early stopping')
                    break
            print(epoch,'val_loss:',loss_val,'best loss:',best_loss,'acc:',acc_val)
    with tf.Session() as sess:
        saver.restore(sess, './models/ex_8_1_models.ckpt')
        acc_test = accuracy.eval(feed_dict={x:x_test1,y:y_test1})
        print('final test accuracy:{:.2f}%'.format(acc_test*100))

ex_8_1()
