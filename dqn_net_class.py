#!/usr/bin/env python
import tensorflow as tf
import random
import numpy as np
from math import sqrt
from batch_norm import batch_norm
from util import plot_conv_weights
from termcolor import colored
from net import Network

class DqnNetClass(Network):
    """ DQN Network Model for Classification """

    def __init__(
        self, sess, height, width, phi_length, n_actions, name,
        optimizer='RMS', learning_rate=0.00025, epsilon=0.01, decay=0.95, momentum=0.,
        slow=False, tau=0.001, verbose=False, path='', folder='_networks'):
        """ Initialize network """
        super(DqnNetClass, self).__init__(sess, name=name)
        self.slow = slow
        self.tau = tau
        self.name = name
        self.sess = sess
        self.path = path
        self.folder = folder

        self.observation = tf.placeholder(tf.float32, [None, height, width, phi_length], name=self.name + '_observation')
        self.actions = tf.placeholder(tf.float32, shape=[None, n_actions], name=self.name + "_actions")

        # q network model:
        with tf.name_scope("Conv1") as scope:
            kernel_shape = [8, 8, phi_length, 32]
            self.W_conv1 = self.weight_variable(kernel_shape, 'conv1')
            self.b_conv1 = self.bias_variable(kernel_shape, 'conv1')
            self.h_conv1 = tf.nn.relu(tf.add(self.conv2d(self.observation, self.W_conv1, 4), self.b_conv1), name=self.name + '_conv1_activations')
            tf.add_to_collection('conv_weights', self.W_conv1)
            tf.add_to_collection('conv_output', self.h_conv1)
            tf.add_to_collection('transfer_params', self.W_conv1)
            tf.add_to_collection('transfer_params', self.b_conv1)

        with tf.name_scope("Conv2") as scope:
            kernel_shape = [4, 4, 32, 64]
            self.W_conv2 = self.weight_variable(kernel_shape, 'conv2')
            self.b_conv2 = self.bias_variable(kernel_shape, 'conv2')
            self.h_conv2 = tf.nn.relu(tf.add(self.conv2d(self.h_conv1, self.W_conv2, 2), self.b_conv2), name=self.name + '_conv2_activations')
            tf.add_to_collection('conv_weights', self.W_conv2)
            tf.add_to_collection('conv_output', self.h_conv2)
            tf.add_to_collection('transfer_params', self.W_conv2)
            tf.add_to_collection('transfer_params', self.b_conv2)

        with tf.name_scope("Conv3") as scope:
            kernel_shape = [3, 3, 64, 64]
            self.W_conv3 = self.weight_variable(kernel_shape, 'conv3')
            self.b_conv3 = self.bias_variable(kernel_shape, 'conv3')
            self.h_conv3 = tf.nn.relu(tf.add(self.conv2d(self.h_conv2, self.W_conv3, 1), self.b_conv3), name=self.name + '_conv3_activations')
            tf.add_to_collection('conv_weights', self.W_conv3)
            tf.add_to_collection('conv_output', self.h_conv3)
            tf.add_to_collection('transfer_params', self.W_conv3)
            tf.add_to_collection('transfer_params', self.b_conv3)

        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 3136])

        with tf.name_scope("FullyConnected1") as scope:
            kernel_shape = [3136, 512]
            self.W_fc1 = self.weight_variable(kernel_shape, 'fc1')
            self.b_fc1 = self.bias_variable(kernel_shape, 'fc1')
            self.h_fc1 = tf.nn.relu(tf.add(tf.matmul(self.h_conv3_flat, self.W_fc1), self.b_fc1), name=self.name + '_fc1_activations')
            tf.add_to_collection('transfer_params', self.W_fc1)
            tf.add_to_collection('transfer_params', self.b_fc1)

        with tf.name_scope("FullyConnected2") as scope:
            kernel_shape = [512, n_actions]
            self.W_fc2 = self.weight_variable_last_layer(kernel_shape, 'fc2')
            self.b_fc2 = self.bias_variable_last_layer(kernel_shape, 'fc2')
            self.action_output = tf.add(tf.matmul(self.h_fc1, self.W_fc2), self.b_fc2, name=self.name + '_fc1_outputs')
            tf.add_to_collection('transfer_params', self.W_fc2)
            tf.add_to_collection('transfer_params', self.b_fc2)

        if verbose:
            self.init_verbosity()

        # cost of q network
        with tf.name_scope("Entropy") as scope:
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.action_output, self.actions))
            ce_summ = tf.summary.scalar("cross_entropy", self.cross_entropy)
        self.parameters = [
            self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_conv3, self.b_conv3,
            self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2,
        ]
        self.gradient = tf.gradients(self.cross_entropy, self.parameters)
        with tf.name_scope("Train") as scope:
            if optimizer == "Adam":
                self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
            else:
                self.opt = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum, epsilon=epsilon)
            self.train_step = self.opt.apply_gradients(zip(self.gradient, self.parameters))
        with tf.name_scope("Evaluating") as scope:
            correct_prediction = tf.equal(tf.argmax(self.action_output,1), tf.argmax(self.actions,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_summary = tf.summary.scalar("accuracy", self.accuracy)
        # initialize all tensor variable parameters
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.path + self.folder + '/log_tb', self.sess.graph)

    def evaluate(self, s_j_batch, a_batch):
        return self.sess.run(
            [self.accuracy, self.merged],
            feed_dict={
                self.actions: a_batch,
                self.observation : s_j_batch,
            }
        )

    def train(self, s_j_batch, a_batch):
        t_ops = [
            self.merged, self.train_step, self.cross_entropy, self.accuracy
        ]
        return self.sess.run(
            t_ops,
            feed_dict={
                self.actions : a_batch,
                self.observation : s_j_batch,
            }
        )

    def add_summary(self, summary, step):
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def load(self):
        has_checkpoint = False
        # saving and loading networks
        checkpoint = tf.train.get_checkpoint_state(self.folder)

        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
            has_checkpoint = True
        return has_checkpoint

    def save(self, step=-1):
        print colored('Saving checkpoint...', 'blue')
        if step < 0:
            self.saver.save(self.sess, self.folder + '/' + self.name + '-dqn')
        else:
            self.saver.save(self.sess, self.folder + '/' + self.name + '-dqn', global_step=step)

        transfer_params = tf.get_collection("transfer_params")
        transfer_saver = tf.train.Saver(transfer_params)
        transfer_saver.save(self.sess, self.folder + '/transfer_model/' + self.name + '-dqn')

        W1_val = self.W_conv1.eval()
        np.savetxt(self.folder + '/conv1_weights.csv', W1_val.flatten())
        b1_val = self.b_conv1.eval()
        np.savetxt(self.folder + '/conv1_biases.csv', b1_val.flatten())

        W2_val = self.W_conv2.eval()
        np.savetxt(self.folder + '/conv2_weights.csv', W2_val.flatten())
        b2_val = self.b_conv2.eval()
        np.savetxt(self.folder + '/conv2_biases.csv', b2_val.flatten())

        W3_val = self.W_conv3.eval()
        np.savetxt(self.folder + '/conv3_weights.csv', W3_val.flatten())
        b3_val = self.b_conv3.eval()
        np.savetxt(self.folder + '/conv3_biases.csv', b3_val.flatten())

        Wfc1_val = self.W_fc1.eval()
        np.savetxt(self.folder + '/fc1_weights.csv', Wfc1_val.flatten())
        bfc1_val = self.b_fc1.eval()
        np.savetxt(self.folder + '/fc1_biases.csv', bfc1_val.flatten())

        Wfc2_val = self.W_fc2.eval()
        np.savetxt(self.folder + '/fc2_weights.csv', Wfc2_val.flatten())
        bfc2_val = self.b_fc2.eval()
        np.savetxt(self.folder + '/fc2_biases.csv', bfc2_val.flatten())

        conv_weights = self.sess.run([tf.get_collection('conv_weights')])
        for i, c in enumerate(conv_weights[0]):
            plot_conv_weights(c, 'conv{}'.format(i+1), folder=self.folder)
        print colored('Successfully saved checkpoint!', 'green')

    def init_verbosity(self):
        with tf.name_scope("Summary_Conv1") as scope:
            self.variable_summaries(self.W_conv1, 'weights')
            self.variable_summaries(self.b_conv1, 'biases')
            tf.summary.histogram('activations', self.h_conv1)
        with tf.name_scope("Summary_Conv2") as scope:
            self.variable_summaries(self.W_conv2, 'weights')
            self.variable_summaries(self.b_conv2, 'biases')
            tf.summary.histogram('activations', self.h_conv2)
        with tf.name_scope("Summary_Conv3") as scope:
            self.variable_summaries(self.W_conv3, 'weights')
            self.variable_summaries(self.b_conv3, 'biases')
            tf.summary.histogram('/activations', self.h_conv3)
        with tf.name_scope("Summary_Flatten") as scope:
            tf.summary.histogram('activations', self.h_conv3_flat)
        with tf.name_scope("Summary_FullyConnected1") as scope:
            self.variable_summaries(self.W_fc1, 'weights')
            self.variable_summaries(self.b_fc1, 'biases')
            tf.summary.histogram('activations', self.h_fc1)
        with tf.name_scope("Summary_FullyConnected2") as scope:
            self.variable_summaries(self.W_fc2, 'weights')
            self.variable_summaries(self.b_fc2, 'biases')
            tf.summary.histogram('activations', self.action_output)
