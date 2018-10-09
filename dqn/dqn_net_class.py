#!/usr/bin/env python
import tensorflow as tf
import os
import random
import numpy as np
from math import sqrt
from batch_norm import batch_norm
from termcolor import colored

from net import Network
from common.util import plot_conv_weights

try:
    import cPickle as pickle
except ImportError:
    import pickle

class DqnNetClass(Network):
    """ DQN Network Model for Classification """

    def __init__(
        self, height, width, phi_length, n_actions, name,
        optimizer='RMS', learning_rate=0.00025, epsilon=0.01, decay=0.95, momentum=0.,
        slow=False, tau=0.001, verbose=False, path='', folder='_networks', l2_decay=0.0001):
        """ Initialize network """
        super(DqnNetClass, self).__init__(None, name=name)
        self.graph = tf.Graph()
        self.slow = slow
        self.tau = tau
        self.name = name
        self.path = path
        self.folder = folder

        with self.graph.as_default():
            self.observation = tf.placeholder(tf.float32, [None, height, width, phi_length], name=self.name + '_observation')
            self.actions = tf.placeholder(tf.float32, shape=[None, n_actions], name=self.name + "_actions")

            self.observation_n = tf.div(self.observation, 255.)

            # q network model:
            with tf.name_scope("Conv1") as scope:
                self.W_conv1, self.b_conv1 = self.conv_variable([8, 8, phi_length, 32], 'conv1')
                self.h_conv1 = tf.nn.relu(tf.add(self.conv2d(self.observation_n, self.W_conv1, 4), self.b_conv1), name=self.name + '_conv1_activations')
                tf.add_to_collection('conv_weights', self.W_conv1)
                tf.add_to_collection('conv_output', self.h_conv1)
                tf.add_to_collection('transfer_params', self.W_conv1)
                tf.add_to_collection('transfer_params', self.b_conv1)

            with tf.name_scope("Conv2") as scope:
                self.W_conv2, self.b_conv2 = self.conv_variable([4, 4, 32, 64], 'conv2')
                self.h_conv2 = tf.nn.relu(tf.add(self.conv2d(self.h_conv1, self.W_conv2, 2), self.b_conv2), name=self.name + '_conv2_activations')
                tf.add_to_collection('conv_weights', self.W_conv2)
                tf.add_to_collection('conv_output', self.h_conv2)
                tf.add_to_collection('transfer_params', self.W_conv2)
                tf.add_to_collection('transfer_params', self.b_conv2)

            with tf.name_scope("Conv3") as scope:
                self.W_conv3, self.b_conv3 = self.conv_variable([3, 3, 64, 64], 'conv3')
                self.h_conv3 = tf.nn.relu(tf.add(self.conv2d(self.h_conv2, self.W_conv3, 1), self.b_conv3), name=self.name + '_conv3_activations')
                tf.add_to_collection('conv_weights', self.W_conv3)
                tf.add_to_collection('conv_output', self.h_conv3)
                tf.add_to_collection('transfer_params', self.W_conv3)
                tf.add_to_collection('transfer_params', self.b_conv3)

            self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 3136])

            with tf.name_scope("FullyConnected1") as scope:
                self.W_fc1, self.b_fc1 = self.fc_variable([3136, 512], 'fc1')
                self.h_fc1 = tf.nn.relu(tf.add(tf.matmul(self.h_conv3_flat, self.W_fc1), self.b_fc1), name=self.name + '_fc1_activations')
                tf.add_to_collection('transfer_params', self.W_fc1)
                tf.add_to_collection('transfer_params', self.b_fc1)

            with tf.name_scope("FullyConnected2") as scope:
                self.W_fc2, self.b_fc2 = self.fc_variable([512, n_actions], 'fc2')
                self.action_output = tf.add(tf.matmul(self.h_fc1, self.W_fc2), self.b_fc2, name=self.name + '_fc1_outputs')
                tf.add_to_collection('transfer_params', self.W_fc2)
                tf.add_to_collection('transfer_params', self.b_fc2)

            if verbose:
                self.init_verbosity()

            self.max_value = tf.reduce_max(self.action_output, axis=None)
            self.action = tf.nn.softmax(self.action_output)

            # cost of q network
            with tf.name_scope("Entropy") as scope:
                # l2_regularizer = l2_decay * (
                #     tf.nn.l2_loss(self.W_conv1) +
                #     tf.nn.l2_loss(self.W_conv2) +
                #     tf.nn.l2_loss(self.W_conv3) +
                #     tf.nn.l2_loss(self.W_fc1) +
                #     tf.nn.l2_loss(self.W_fc2)
                # )
                self.cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        _sentinel=None,
                        labels=self.actions,
                        logits=self.action_output)) # + l2_regularizer

            with tf.name_scope("Train") as scope:
                if optimizer == "Adam":
                    self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
                else:
                    self.opt = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum, epsilon=epsilon)
                self.grads_vars = self.opt.compute_gradients(self.cross_entropy)
                grads = []
                params = []
                for p in self.grads_vars:
                    if p[0] == None:
                        continue
                    grads.append(p[0])
                    params.append(p[1])

                #grads = tf.clip_by_global_norm(grads, 1)[0]
                self.grads_vars_updates = zip(grads, params)
                self.train_step = self.opt.apply_gradients(self.grads_vars_updates)
                # for grad, var in self.grads_vars:
                #     if grad == None:
                #         continue
                #     tf.summary.histogram(var.op.name + '/gradients', grad)

            with tf.name_scope("Evaluating") as scope:
                correct_prediction = tf.equal(tf.argmax(self.action_output,1), tf.argmax(self.actions,1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            self.saver = tf.train.Saver()
            self.merged = tf.summary.merge_all()

        if not os.path.exists(self.folder + '/transfer_model'):
            os.makedirs(self.folder + '/transfer_model')
        if not os.path.exists(self.folder + '/final/transfer_model'):
            os.makedirs(self.folder + '/final/transfer_model')

    def initializer(self, sess):
        # initialize all tensor variable parameters
        self.sess = sess
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

        self.writer = tf.summary.FileWriter(self.path + self.folder + '/log_tb', self.sess.graph)

    def evaluate(self, state):
        return  self.sess.run(
            self.action,
            feed_dict={
                self.observation: state
            }
        )

    def evaluate_batch(self, s_j_batch, a_batch):
        return self.sess.run(
            [self.cross_entropy, self.accuracy, self.action_output, self.max_value],
            feed_dict={
                self.actions: a_batch,
                self.observation: s_j_batch
            }
        )

    def train(self, s_j_batch, a_batch):
        t_ops = [
            self.train_step, self.cross_entropy, self.accuracy, self.action_output, self.max_value
        ]
        return self.sess.run(
            t_ops,
            feed_dict={
                self.actions : a_batch,
                self.observation : s_j_batch
            }
        )

    def add_summary(self, summary, step):
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def add_accuracy(self, accuracy, entropy, step, stage='Training'):
        summary = tf.Summary()
        summary.value.add(tag='{}/Accuracy'.format(stage), simple_value=float(accuracy))
        summary.value.add(tag='{}/CrossEntropy'.format(stage), simple_value=float(entropy))
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def load(self):
        self.saver.restore(self.sess, self.folder + '/' + self.name + '-dqn')
        print ("Successfully loaded:", self.folder + '/' + self.name + '-dqn')

    def save(self, step=-1, model_max_output_val=0., relative='/'):
        print (colored('Saving model and data...', 'blue'))
        self.saver.save(self.sess, self.folder + '{}'.format(relative) + self.name + '-dqn')
        if step >= 0:
            with open(self.folder + '{}step'.format(relative), 'w') as f_step:
                f_step.write(str(step))

        with self.graph.as_default():
            transfer_params = tf.get_collection("transfer_params")
            transfer_saver = tf.train.Saver(transfer_params)
            transfer_saver.save(self.sess, self.folder + '{}transfer_model/'.format(relative) + self.name + '-dqn')

        with self.sess.as_default():
            W1_val = self.W_conv1.eval()
            np.savetxt(self.folder + '{}conv1_weights.csv'.format(relative), W1_val.flatten())
            b1_val = self.b_conv1.eval()
            np.savetxt(self.folder + '{}conv1_biases.csv'.format(relative), b1_val.flatten())

            W2_val = self.W_conv2.eval()
            np.savetxt(self.folder + '{}conv2_weights.csv'.format(relative), W2_val.flatten())
            b2_val = self.b_conv2.eval()
            np.savetxt(self.folder + '{}conv2_biases.csv'.format(relative), b2_val.flatten())

            W3_val = self.W_conv3.eval()
            np.savetxt(self.folder + '{}conv3_weights.csv'.format(relative), W3_val.flatten())
            b3_val = self.b_conv3.eval()
            np.savetxt(self.folder + '{}conv3_biases.csv'.format(relative), b3_val.flatten())

            Wfc1_val = self.W_fc1.eval()
            np.savetxt(self.folder + '{}fc1_weights.csv'.format(relative), Wfc1_val.flatten())
            bfc1_val = self.b_fc1.eval()
            np.savetxt(self.folder + '{}fc1_biases.csv'.format(relative), bfc1_val.flatten())

            Wfc2_val = self.W_fc2.eval()
            np.savetxt(self.folder + '{}fc2_weights.csv'.format(relative), Wfc2_val.flatten())
            bfc2_val = self.b_fc2.eval()
            np.savetxt(self.folder + '{}fc2_biases.csv'.format(relative), bfc2_val.flatten())

        # with self.graph.as_default():
        #     conv_weights = self.sess.run([tf.get_collection('conv_weights')])
        #     for i, c in enumerate(conv_weights[0]):
        #         plot_conv_weights(c, 'conv{}'.format(i+1), folder=self.folder)

        self.save_max_value(model_max_output_val, relative=relative)
        print (colored('Successfully saved model and data!', 'green'))

    def save_max_value(self, model_max_output_val, relative='/'):
        with open(self.folder + '{}transfer_model/max_output_value'.format(relative), 'w') as f_max_value:
            f_max_value.write(str(model_max_output_val))

    def init_verbosity(self):
        # with tf.name_scope("Summary_Conv1") as scope:
        #     self.variable_summaries(self.W_conv1, 'weights')
        #     self.variable_summaries(self.b_conv1, 'biases')
        #     tf.summary.histogram('activations', self.h_conv1)
        # with tf.name_scope("Summary_Conv2") as scope:
        #     self.variable_summaries(self.W_conv2, 'weights')
        #     self.variable_summaries(self.b_conv2, 'biases')
        #     tf.summary.histogram('activations', self.h_conv2)
        # with tf.name_scope("Summary_Conv3") as scope:
        #     self.variable_summaries(self.W_conv3, 'weights')
        #     self.variable_summaries(self.b_conv3, 'biases')
        #     tf.summary.histogram('/activations', self.h_conv3)
        # with tf.name_scope("Summary_Flatten") as scope:
        #     tf.summary.histogram('activations', self.h_conv3_flat)
        # with tf.name_scope("Summary_FullyConnected1") as scope:
        #     self.variable_summaries(self.W_fc1, 'weights')
        #     self.variable_summaries(self.b_fc1, 'biases')
        #     tf.summary.histogram('activations', self.h_fc1)
        with tf.name_scope("Summary_FullyConnected2") as scope:
            self.variable_summaries(self.W_fc2, 'weights')
            self.variable_summaries(self.b_fc2, 'biases')
            tf.summary.histogram('activations', self.action_output)
