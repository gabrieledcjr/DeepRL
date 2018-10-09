#!/usr/bin/env python
import tensorflow as tf
import random
import numpy as np
from math import sqrt
from batch_norm import batch_norm
from termcolor import colored

from net import Network
from common.util import plot_conv_weights

class DqnNetClass(Network):
    """ DQN Network Model for Classification """

    def __init__(
        self, sess, height, width, phi_length, n_actions, name,
        optimizer='RMS', learning_rate=0.00025, epsilon=0.01, decay=0.95, momentum=0.,
        slow=False, tau=0.001, verbose=False, path='', folder='_networks', l2_decay=0.001):
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

        self.is_training = tf.placeholder(tf.bool, [])

        with tf.name_scope("Conv1") as scope:
            kernel_shape = [8, 8, phi_length, 32]
            self.W_conv1 = self.weight_variable(kernel_shape, 'conv1')
            #self.b_conv1 = self.bias_variable(kernel_shape, 'conv1')
            self.h_conv1_bn = batch_norm(self.conv2d(self.observation, self.W_conv1, 4), 32, self.is_training, self.sess, slow=self.slow, tau=self.tau)
            self.h_conv1 = tf.nn.relu(self.h_conv1_bn.bnorm, name=self.name + '_conv1_activations')
            tf.add_to_collection('conv_weights', self.W_conv1)
            tf.add_to_collection('conv_output', self.h_conv1)
            tf.add_to_collection('transfer_params', self.W_conv1)
            tf.add_to_collection('transfer_params', self.h_conv1_bn.scale)
            tf.add_to_collection('transfer_params', self.h_conv1_bn.beta)
            tf.add_to_collection('transfer_params', self.h_conv1_bn.pop_mean)
            tf.add_to_collection('transfer_params', self.h_conv1_bn.pop_var)

        with tf.name_scope("Conv2") as scope:
            kernel_shape = [4, 4, 32, 64]
            self.W_conv2 = self.weight_variable(kernel_shape, 'conv2')
            #self.b_conv2 = self.bias_variable(kernel_shape, 'conv2')
            self.h_conv2_bn = batch_norm(self.conv2d(self.h_conv1, self.W_conv2, 2), 64, self.is_training, self.sess, slow=self.slow, tau=self.tau)
            self.h_conv2 = tf.nn.relu(self.h_conv2_bn.bnorm, name=self.name + '_conv2_activations')
            tf.add_to_collection('conv_weights', self.W_conv2)
            tf.add_to_collection('conv_output', self.h_conv2)
            tf.add_to_collection('transfer_params', self.W_conv2)
            tf.add_to_collection('transfer_params', self.h_conv2_bn.scale)
            tf.add_to_collection('transfer_params', self.h_conv2_bn.beta)
            tf.add_to_collection('transfer_params', self.h_conv2_bn.pop_mean)
            tf.add_to_collection('transfer_params', self.h_conv2_bn.pop_var)

        with tf.name_scope("Conv3") as scope:
            kernel_shape = [3, 3, 64, 64]
            self.W_conv3 = self.weight_variable(kernel_shape, 'conv3')
            #self.b_conv3 = self.bias_variable(kernel_shape, 'conv3')
            self.h_conv3_bn = batch_norm(self.conv2d(self.h_conv2, self.W_conv3, 1), 64, self.is_training, self.sess, slow=self.slow, tau=self.tau)
            self.h_conv3 = tf.nn.relu(self.h_conv3_bn.bnorm, name=self.name + '_conv3_activations')
            tf.add_to_collection('conv_weights', self.W_conv3)
            tf.add_to_collection('conv_output', self.h_conv3)
            tf.add_to_collection('transfer_params', self.W_conv3)
            tf.add_to_collection('transfer_params', self.h_conv3_bn.scale)
            tf.add_to_collection('transfer_params', self.h_conv3_bn.beta)
            tf.add_to_collection('transfer_params', self.h_conv3_bn.pop_mean)
            tf.add_to_collection('transfer_params', self.h_conv3_bn.pop_var)

        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 3136])

        with tf.name_scope("FullyConnected1") as scope:
            kernel_shape = [3136, 512]
            self.W_fc1 = self.weight_variable(kernel_shape, 'fc1')
            #self.b_fc1 = self.bias_variable(kernel_shape, 'fc1')
            self.h_fc1_bn = batch_norm(tf.matmul(self.h_conv3_flat, self.W_fc1), 512, self.is_training, self.sess, slow=self.slow, tau=self.tau, linear=True)
            self.h_fc1 = tf.nn.relu(self.h_fc1_bn.bnorm, name=self.name + '_fc1_activations')
            tf.add_to_collection('transfer_params', self.W_fc1)
            tf.add_to_collection('transfer_params', self.h_fc1_bn.scale)
            tf.add_to_collection('transfer_params', self.h_fc1_bn.beta)
            tf.add_to_collection('transfer_params', self.h_fc1_bn.pop_mean)
            tf.add_to_collection('transfer_params', self.h_fc1_bn.pop_var)

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
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    _sentinel=None,
                    labels=self.actions,
                    logits=self.action_output)) #+ \
                    #l2_decay*tf.nn.l2_loss(self.W_fc2) + l2_decay*tf.nn.l2_loss(self.b_fc2))
            ce_summ = tf.summary.scalar("cross_entropy", self.cross_entropy)
        # self.parameters = [
        #     self.W_conv1, self.h_conv1_bn.scale, self.h_conv1_bn.beta,
        #     self.W_conv2, self.h_conv2_bn.scale, self.h_conv2_bn.beta,
        #     self.W_conv3, self.h_conv3_bn.scale, self.h_conv3_bn.beta,
        #     self.W_fc1, self.h_fc1_bn.scale, self.h_fc1_bn.beta,
        #     self.W_fc2, self.b_fc2,
        # ]
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

            grads = tf.clip_by_global_norm(grads, 1)[0]
            self.grads_vars_updates = zip(grads, params)
            self.train_step = self.opt.apply_gradients(self.grads_vars_updates)
            # for grad, var in self.grads_vars:
            #     if grad == None:
            #         continue
            #     tf.summary.histogram(var.op.name + '/gradients', grad)
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
                self.is_training: False
            }
        )

    def train(self, s_j_batch, a_batch):
        t_ops = [
            self.merged, self.train_step, self.cross_entropy, self.accuracy,
            self.h_conv1_bn.train, self.h_conv2_bn.train,
            self.h_conv3_bn.train, self.h_fc1_bn.train
        ]
        return self.sess.run(
            t_ops,
            feed_dict={
                self.actions : a_batch,
                self.observation : s_j_batch,
                self.is_training: True
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
            print ("Successfully loaded:", checkpoint.model_checkpoint_path)
            has_checkpoint = True
        return has_checkpoint

    def save(self, step=-1):
        print (colored('Saving checkpoint...', 'blue'))
        if step < 0:
            self.saver.save(self.sess, self.folder + '/' + self.name + '-dqn')
        else:
            self.saver.save(self.sess, self.folder + '/' + self.name + '-dqn', global_step=step)

        transfer_params = tf.get_collection("transfer_params")
        transfer_saver = tf.train.Saver(transfer_params)
        transfer_saver.save(self.sess, self.folder + '/transfer_model/' + self.name + '-dqn')

        W1_val = self.W_conv1.eval()
        np.savetxt(self.folder + '/conv1_weights.csv', W1_val.flatten())
        scale1_val = self.h_conv1_bn.scale.eval()
        np.savetxt(self.folder + '/conv1_scale.csv', scale1_val.flatten())
        beta1_val = self.h_conv1_bn.beta.eval()
        np.savetxt(self.folder + '/conv1_beta.csv', beta1_val.flatten())

        W2_val = self.W_conv2.eval()
        np.savetxt(self.folder + '/conv2_weights.csv', W2_val.flatten())
        scale2_val = self.h_conv2_bn.scale.eval()
        np.savetxt(self.folder + '/conv2_scale.csv', scale2_val.flatten())
        beta2_val = self.h_conv2_bn.beta.eval()
        np.savetxt(self.folder + '/conv2_beta.csv', beta2_val.flatten())

        W3_val = self.W_conv3.eval()
        np.savetxt(self.folder + '/conv3_weights.csv', W3_val.flatten())
        scale3_val = self.h_conv3_bn.scale.eval()
        np.savetxt(self.folder + '/conv3_scale.csv', scale3_val.flatten())
        beta3_val = self.h_conv3_bn.beta.eval()
        np.savetxt(self.folder + '/conv3_beta.csv', beta3_val.flatten())

        Wfc1_val = self.W_fc1.eval()
        np.savetxt(self.folder + '/fc1_weights.csv', Wfc1_val.flatten())
        fc_scale1_val = self.h_fc1_bn.scale.eval()
        np.savetxt(self.folder + '/fc1_scale.csv', fc_scale1_val.flatten())
        fc_beta1_val = self.h_fc1_bn.beta.eval()
        np.savetxt(self.folder + '/fc1_beta.csv', fc_beta1_val.flatten())

        Wfc2_val = self.W_fc2.eval()
        np.savetxt(self.folder + '/fc2_weights.csv', Wfc2_val.flatten())
        bfc2_val = self.b_fc2.eval()
        np.savetxt(self.folder + '/fc2_biases.csv', bfc2_val.flatten())

        conv_weights = self.sess.run([tf.get_collection('conv_weights')])
        for i, c in enumerate(conv_weights[0]):
            plot_conv_weights(c, 'conv{}'.format(i+1), folder=self.folder)
        print (colored('Successfully saved checkpoint!', 'green'))

    def init_verbosity(self):
        with tf.name_scope("Summary_Conv1") as scope:
            self.variable_summaries(self.W_conv1, 'weights')
            self.variable_summaries(self.h_conv1_bn.scale, 'scale')
            self.variable_summaries(self.h_conv1_bn.beta, 'beta')
            tf.summary.histogram('activations', self.h_conv1)
            tf.summary.histogram('BN/activations', self.h_conv1_bn.bnorm)
        with tf.name_scope("Summary_Conv2") as scope:
            self.variable_summaries(self.W_conv2, 'weights')
            self.variable_summaries(self.h_conv2_bn.scale, 'scale')
            self.variable_summaries(self.h_conv2_bn.beta, 'beta')
            tf.summary.histogram('activations', self.h_conv2)
            tf.summary.histogram('BN/activations', self.h_conv2_bn.bnorm)
        with tf.name_scope("Summary_Conv3") as scope:
            self.variable_summaries(self.W_conv3, 'weights')
            self.variable_summaries(self.h_conv3_bn.scale, 'scale')
            self.variable_summaries(self.h_conv3_bn.beta, 'beta')
            tf.summary.histogram('/activations', self.h_conv3)
            tf.summary.histogram('BN/activations', self.h_conv3_bn.bnorm)
        with tf.name_scope("Summary_Flatten") as scope:
            tf.summary.histogram('activations', self.h_conv3_flat)
        with tf.name_scope("Summary_FullyConnected1") as scope:
            self.variable_summaries(self.W_fc1, 'weights')
            self.variable_summaries(self.h_fc1_bn.scale, 'scale')
            self.variable_summaries(self.h_fc1_bn.beta, 'beta')
            tf.summary.histogram('activations', self.h_fc1)
            tf.summary.histogram('BN/activations', self.h_fc1_bn.bnorm)
        with tf.name_scope("Summary_FullyConnected2") as scope:
            self.variable_summaries(self.W_fc2, 'weights')
            self.variable_summaries(self.b_fc2, 'biases')
            tf.summary.histogram('activations', self.action_output)
