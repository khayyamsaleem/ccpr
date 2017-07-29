import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from sklearn.model_selection import train_test_split as split_data
import matplotlib.pyplot as plt

class CC_Learn:
    def __init__(self, train_test_split=70, filename="data.csv", norm=True):
        self.hp = {
                "num_filt_1" : 8, #Number of filters in first conv layer
                "num_filt_2" : 4,
                "num_filt_3" : 2,
                "num_fc_1" : 12, #number of neurons in fully connected layer
                "max_iterations" : 500,
                "batch_size" : 16,
                "dropout" : 0.70,
                "learning_rate" : 2e-5,
                "input_norm" : False
                }
        self.__filename = "data.csv"
        self.__ttsplit = train_test_split*0.01
        self.norm = norm


        """loading data"""
        self.data = np.loadtxt(self.__filename, delimiter=',')
        self.data_train, self.data_test_val = split_data(self.data, test_size=self.__ttsplit)

        self.data_test, self.data_val = np.array_split(self.data_test_val,2)

        self.X_train = self.data_train[:,1:]
        self.X_val = self.data_val[:,1:]
        self.X_test = self.data_test[:,1:]
        self.__N = self.X_train.shape[0]
        print("N: " + str(self.__N))
        self.__D = self.X_train.shape[1]
        self.y_train = self.data_train[:,0]
        self.y_val = self.data_val[:,0]
        self.y_test = self.data_test[:,0]
        print("We have %s observations with %s dimensions"%(self.__N,self.__D))

        self.num_classes = len(np.unique(self.y_train))
        base = np.min(self.y_train)
        if base != 0: #checks if labels are zero based
            self.y_train -= base
            self.y_val -= base
            self.y_test -=base

        if self.norm:
            self.input_normalize()

    def input_normalize(self):
        mean = np.mean(self.X_train,axis=0)
        variance = np.var(self.X_train, axis=0)
        self.X_train = (self.X_train - mean) / np.sqrt(variance)+1e-9
        self.X_val = (self.X_val - mean) / np.sqrt(variance)+1e-9
        self.X_test = (self.X_test - mean) / np.sqrt(variance)+1e-9

    def plot(self, plot_row=5):
        f, axarr = plt.subplots(plot_row, self.num_classes)
        for c in np.unique(self.y_train):
            ind = np.where(self.y_train == c)
            ind_plot = np.random.choice(ind[0], size=plot_row)
            for n in range(plot_row):
                c = int(c)
                axarr[n,c].plot(self.X_train[ind_plot[n],:])
                if not n == plot_row-1:
                    plt.setp([axarr[n,c].get_xticklabels()], visible=False)
                if not c == 0:
                    plt.setp([axarr[n,c].get_yticklabels()], visible=False)
        f.subplots_adjust(hspace=0)
        f.subplots_adjust(wspace=0)
        plt.show()

    def train(self):
        epochs = int(self.hp["batch_size"]*self.hp["max_iterations"] / self.__N)
        print("Train with approximately %d epochs" % (epochs))

        x = tf.placeholder("float", shape=[None, self.__D], name="Input_data")
        y_ = tf.placeholder(tf.int64, shape=[None], name="Ground_truth")
        keep_prob = tf.placeholder("float")
        bn_train = tf.placeholder(tf.bool)

        bias_variable = lambda s,n: tf.Variable(tf.constant(0.1, shape=s), name=n)
        conv2d = lambda x,W: tf.nn.conv2d(x,W, strides=[1,1,1,1], padding="SAME")
        max_pool = lambda x: tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

        with tf.name_scope("Reshaping_data") as scope:
            x_image = tf.reshape(x, [-1,self.__D,1,1])

        i = tf.contrib.layers.xavier_initializer()

        with tf.name_scope("Conv1") as scope:
            W_conv1 =tf.get_variable("Conv_Layer_1",shape=[5,1,1,self.hp["num_filt_1"]],initializer=i)
            b_conv1 = bias_variable([self.hp["num_filt_1"]], 'bias_for_Conv_Layer_1')
            a_conv1 = conv2d(x_image, W_conv1) + b_conv1

        with tf.name_scope("Batch_norm_conv1") as scope:
            a_conv1 = tf.contrib.layers.batch_norm(a_conv1,is_training=bn_train,updates_collections=None)
            h_conv1=tf.nn.relu(a_conv1)

        with tf.name_scope("Conv2") as scope:
            W_conv2 = tf.get_variable("Conv_Layer_2", shape=[4,1,self.hp["num_filt_1"], self.hp["num_filt_2"]], initializer=i)
            b_conv2 = bias_variable([self.hp["num_filt_2"]], "bias_for_Conv_Layer_2")
            a_conv2 = conv2d(h_conv1, W_conv2) + b_conv2

        with tf.name_scope("Batch_norm_conv2") as scope:
            a_conv2 = tf.contrib.layers.batch_norm(a_conv2, is_training=bn_train, updates_collections=None)
            h_conv2 = tf.nn.relu(a_conv2)

        with tf.name_scope("Fully_Connected1") as scope:
            W_fc1 = tf.get_variable("Fully_Connected_layer_1", shape=[self.__D*self.hp["num_filt_2"], self.hp["num_fc_1"]], initializer=i)
            b_fc1 = bias_variable([self.hp["num_fc_1"]], "bias_for_Fully_Connected_Layer_1")
            h_conv3_flat = tf.reshape(h_conv2, [-1, self.__D*self.hp["num_filt_2"]])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        with tf.name_scope("Fully_Connected2") as scope:
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
            W_fc2 = tf.get_variable("W_fc2", shape=[self.hp["num_fc_1"], self.num_classes], initializer=i)
            b_fc2 = tf.Variable(tf.constant(0.1, shape=[self.num_classes]),name="b_fc2")
            h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        with tf.name_scope("SoftMax") as scope:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h_fc2, labels=y_)
            cost = tf.reduce_sum(loss) / self.hp["batch_size"]
            loss_summ = tf.summary.scalar("cross entropy_loss", cost)

        with tf.name_scope("train") as scope:
            tvars = tf.trainable_variables()
            grads = tf.gradients(cost, tvars)
            optimizer = tf.train.AdamOptimizer(self.hp["learning_rate"])
            gradients = list(zip(grads, tvars))
            train_step = optimizer.apply_gradients(gradients)
            numel = tf.constant([[0]])
            for gradient, variable in gradients:
                if isinstance(gradient, ops.IndexedSlices):
                    grad_values = gradient.values
                else:
                    grad_values = gradient
                numel += tf.reduce_sum(tf.size(variable))

                h1 = tf.summary.histogram(variable.name, variable)
                h2 = tf.summary.histogram(variable.name + "/gradients", grad_values)
                h3 = tf.summary.histogram(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))
        with tf.name_scope("Evaluating_accuracy") as scope:
            correct_prediction = tf.equal(tf.argmax(h_fc2, 1), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            accuracy_summary = tf.summary.scalar("accuracy", accuracy)

        merged = tf.summary.merge_all()

        tvars = tf.trainable_variables()
        for v in tvars:
            print(v.name)

        perf_collect = np.zeros((3, int(np.floor(self.hp["max_iterations"] / 10))))
        cost_ma = 0.0
        acc_ma = 0.0

        with tf.Session() as sess:
            writer = tf.summary.FileWriter("./log_tb", sess.graph)
            sess.run(tf.global_variables_initializer())
            step = 0
            for k in range(self.hp["max_iterations"]):
                batch_ind = np.random.choice(self.__N, self.hp["batch_size"],replace=False)
                if k == 0:
                    result = sess.run(accuracy, feed_dict = {x : self.X_test, y_: self.y_test, keep_prob: 1.0, bn_train : False})
                    acc_test_before = result
                if k%50 == 0:
                    result = sess.run([cost, accuracy], feed_dict = {x : self.X_train, y_ : self.y_train, keep_prob: 1.0, bn_train : False})
                    perf_collect[1, step] = acc_train = result[1]
                    cost_train = result[0]

                    result = sess.run([accuracy,cost,merged], feed_dict={x:self.X_val, y_:self.y_val, keep_prob: 1.0, bn_train: False})
                    perf_collect[0, step] = acc_val = result[0]
                    cost_val = result[1]
                    if k == 0: cost_ma = cost_train
                    if k == 0: acc_ma = acc_train
                    cost_ma = 0.7*cost_ma+0.3*cost_train
                    acc_ma = 0.7*acc_ma + 0.3*acc_train

                    writer.add_summary(result[2], k)
                    writer.flush()
                    print("At %5.0f/%5.0f Cost: train%5.3f val%5.3f(%5.3f) Acc: train%5.3f val%5.3f(%5.3f) " % (k, self.hp["max_iterations"], cost_train, cost_val, cost_ma, acc_train, acc_val, acc_ma))
                    step +=1
                sess.run(train_step, feed_dict={x:self.X_train[batch_ind], y_: self.y_train[batch_ind], keep_prob: self.hp["dropout"], bn_train: True})

            result = sess.run([accuracy, numel], feed_dict={x:self.X_test, y_ : self.y_test, keep_prob:1.0, bn_train: False})
            acc_test = result[0]
            print("The network has %s trainable parameters"%(result[1]))

            print("The accuracy on the test data is %.3f, before training was %.3f" % (acc_test, acc_test_before))
            plt.figure()
            plt.plot(perf_collect[0], label="Valid accuracy")
            plt.plot(perf_collect[1], label="Train accuracy")
            plt.axis([0, step, 0, np.max(perf_collect)])
            plt.legend()
            plt.show()
