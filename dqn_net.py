#!/usr/bin/env python
import tensorflow as tf
import random
import numpy as np
from time import sleep
from util import plot_conv_weights
from termcolor import colored
from net import Network

try:
    import cPickle as pickle
except ImportError:
    import pickle


class DqnNet(Network):
    """ DQN Network Model of DQN Algorithm """

    def __init__(
        self, sess, height, width, phi_length, n_actions, name, gamma=0.99, copy_interval=4,
        optimizer='RMS', learning_rate=0.00025, epsilon=0.01, decay=0.95, momentum=0., l2_decay=0.0001, error_clip=1.0,
        slow=False, tau=0.01, verbose=False, path='', folder='_networks', transfer=False, transfer_folder=''):
        """ Initialize network """
        Network.__init__(self, sess, name=name)
        self.gamma = gamma
        self.slow = slow
        self.tau = tau
        self.name = name
        self.sess = sess
        self.path = path
        self.folder = folder
        self.copy_interval = copy_interval
        self.update_counter = 0

        self.observation = tf.placeholder(tf.float32, [None, height, width, phi_length], name=self.name + '_observation')
        self.actions = tf.placeholder(tf.float32, shape=[None, n_actions], name=self.name + "_actions") # one-hot matrix
        self.next_observation = tf.placeholder(tf.float32, [None, height, width, phi_length], name=self.name + '_t_next_observation')
        self.rewards = tf.placeholder(tf.float32, shape=[None], name=self.name + "_rewards")
        self.terminals = tf.placeholder(tf.float32, shape=[None], name=self.name + "_terminals")

        self.slow_learnrate_vars = []
        self.fast_learnrate_vars = []

        # q network model:
        with tf.name_scope("Conv1") as scope:
            kernel_shape = [8, 8, phi_length, 32]
            self.W_conv1 = self.weight_variable(kernel_shape, 'conv1')
            self.b_conv1 = self.bias_variable(kernel_shape, 'conv1')
            self.h_conv1 = tf.nn.relu(tf.add(self.conv2d(self.observation, self.W_conv1, 4), self.b_conv1), name=self.name + '_conv1_activations')
            tf.add_to_collection('conv_weights', self.W_conv1)
            tf.add_to_collection('conv_output', self.h_conv1)
            if transfer:
                self.slow_learnrate_vars.append(self.W_conv1)
                self.slow_learnrate_vars.append(self.b_conv1)

        with tf.name_scope("Conv2") as scope:
            kernel_shape = [4, 4, 32, 64]
            self.W_conv2 = self.weight_variable(kernel_shape, 'conv2')
            self.b_conv2 = self.bias_variable(kernel_shape, 'conv2')
            self.h_conv2 = tf.nn.relu(tf.add(self.conv2d(self.h_conv1, self.W_conv2, 2), self.b_conv2), name=self.name + '_conv2_activations')
            tf.add_to_collection('conv_weights', self.W_conv2)
            tf.add_to_collection('conv_output', self.h_conv2)
            if transfer:
                self.slow_learnrate_vars.append(self.W_conv2)
                self.slow_learnrate_vars.append(self.b_conv2)

        with tf.name_scope("Conv3") as scope:
            kernel_shape = [3, 3, 64, 64]
            self.W_conv3 = self.weight_variable(kernel_shape, 'conv3')
            self.b_conv3 = self.bias_variable(kernel_shape, 'conv3')
            self.h_conv3 = tf.nn.relu(tf.add(self.conv2d(self.h_conv2, self.W_conv3, 1), self.b_conv3), name=self.name + '_conv3_activations')
            tf.add_to_collection('conv_weights', self.W_conv3)
            tf.add_to_collection('conv_output', self.h_conv3)
            if transfer:
                self.slow_learnrate_vars.append(self.W_conv3)
                self.slow_learnrate_vars.append(self.b_conv3)

        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 3136])

        with tf.name_scope("FullyConnected1") as scope:
            kernel_shape = [3136, 512]
            self.W_fc1 = self.weight_variable(kernel_shape, 'fc1')
            self.b_fc1 = self.bias_variable(kernel_shape, 'fc1')
            self.h_fc1 = tf.nn.relu(tf.add(tf.matmul(self.h_conv3_flat, self.W_fc1), self.b_fc1), name=self.name + '_fc1_activations')
            if transfer:
                self.fast_learnrate_vars.append(self.W_fc1)
                self.fast_learnrate_vars.append(self.b_fc1)

        with tf.name_scope("FullyConnected2") as scope:
            kernel_shape = [512, n_actions]
            self.W_fc2 = self.weight_variable_last_layer(kernel_shape, 'fc2')
            self.b_fc2 = self.bias_variable_last_layer(kernel_shape, 'fc2')
            self.q_value = tf.add(tf.matmul(self.h_fc1, self.W_fc2), self.b_fc2, name=self.name + '_fc1_outputs')
            if transfer:
                self.fast_learnrate_vars.append(self.W_fc2)
                self.fast_learnrate_vars.append(self.b_fc2)

        if transfer:
            self.load_transfer_model(folder=transfer_folder)
            # Scale down the last layer
            W_fc2_scaled = tf.scalar_mul(0.01, self.W_fc2)
            b_fc2_scaled = tf.scalar_mul(0.01, self.b_fc2)
            self.sess.run([
               self.W_fc2.assign(W_fc2_scaled), self.b_fc2.assign(b_fc2_scaled)
            ])

        if verbose:
            self.init_verbosity()

        # target q network model:
        with tf.name_scope("TConv1") as scope:
            kernel_shape = [8, 8, phi_length, 32]
            self.t_W_conv1 = self.weight_variable(kernel_shape, 't_conv1')
            self.t_b_conv1 = self.bias_variable(kernel_shape, 't_conv1')
            self.t_h_conv1 = tf.nn.relu(tf.add(self.conv2d(self.next_observation, self.t_W_conv1, 4), self.t_b_conv1), name=self.name + '_t_conv1_activations')

        with tf.name_scope("TConv2") as scope:
            kernel_shape = [4, 4, 32, 64]
            self.t_W_conv2 = self.weight_variable(kernel_shape, 't_conv2')
            self.t_b_conv2 = self.bias_variable(kernel_shape, 't_conv2')
            self.t_h_conv2 = tf.nn.relu(tf.add(self.conv2d(self.t_h_conv1, self.t_W_conv2, 2), self.t_b_conv2), name=self.name + '_t_conv2_activations')

        with tf.name_scope("TConv3") as scope:
            kernel_shape = [3, 3, 64, 64]
            self.t_W_conv3 = self.weight_variable(kernel_shape, 't_conv3')
            self.t_b_conv3 = self.bias_variable(kernel_shape, 't_conv3')
            self.t_h_conv3 = tf.nn.relu(tf.add(self.conv2d(self.t_h_conv2, self.t_W_conv3, 1), self.t_b_conv3), name=self.name + '_t_conv3_activations')

        self.t_h_conv3_flat = tf.reshape(self.t_h_conv3, [-1, 3136])

        with tf.name_scope("TFullyConnected1") as scope:
            kernel_shape = [3136, 512]
            self.t_W_fc1 = self.weight_variable(kernel_shape, 't_fc1')
            self.t_b_fc1 = self.bias_variable(kernel_shape, 't_fc1')
            self.t_h_fc1 = tf.nn.relu(tf.add(tf.matmul(self.t_h_conv3_flat, self.t_W_fc1), self.t_b_fc1), name=self.name + '_t_fc1_activations')

        with tf.name_scope("TFullyConnected2") as scope:
            kernel_shape = [512, n_actions]
            self.t_W_fc2 = self.weight_variable_last_layer(kernel_shape, 't_fc2')
            self.t_b_fc2 = self.bias_variable_last_layer(kernel_shape, 't_fc2')
            self.t_q_value = tf.add(tf.matmul(self.t_h_fc1, self.t_W_fc2), self.t_b_fc2, name=self.name + '_t_fc1_outputs')

        if transfer:
            # only intialize tensor variables that are not loaded from the transfer model
            #self.sess.run(tf.variables_initializer(fast_learnrate_vars))
            self._global_vars_temp = set(tf.global_variables())

        # cost of q network
        #self.l2_regularizer_loss = l2_decay * (tf.reduce_sum(tf.pow(self.W_conv1, 2)) + tf.reduce_sum(tf.pow(self.W_conv2, 2)) + tf.reduce_sum(tf.pow(self.W_conv3, 2))  + tf.reduce_sum(tf.pow(self.W_fc1, 2)) + tf.reduce_sum(tf.pow(self.W_fc2, 2)))
        self.cost = self.build_loss(error_clip, n_actions) #+ self.l2_regularizer_loss
        self.parameters = [
            self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_conv3, self.b_conv3,
            self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2,
        ]
        self.gradient = tf.gradients(self.cost, self.parameters)
        with tf.name_scope("Train") as scope:
            if optimizer == "Adam":
                #self.train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon).minimize(self.cost)
                self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
            else:
                #self.train_step = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=0.0, epsilon=epsilon).minimize(self.cost)
                self.opt = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum, epsilon=epsilon)
            self.train_step = self.opt.apply_gradients(zip(self.gradient, self.parameters))

        with tf.name_scope("Evaluating") as scope:
            self.accuracy = tf.placeholder(tf.float32, shape=(), name="accuracy")
            accuracy_summary = tf.summary.scalar("accuracy", self.accuracy)

        if transfer:
            vars_diff = set(tf.global_variables()) - self._global_vars_temp
            self.sess.run(tf.initialize_variables(vars_diff))
        else:
            # initialize all tensor variable parameters
            self.sess.run(tf.global_variables_initializer())

        # Make sure q and target model have same initial parameters copy the parameters
        self.sess.run([
            self.t_W_conv1.assign(self.W_conv1), self.t_b_conv1.assign(self.b_conv1),
            self.t_W_conv2.assign(self.W_conv2), self.t_b_conv2.assign(self.b_conv2),
            self.t_W_conv3.assign(self.W_conv3), self.t_b_conv3.assign(self.b_conv3),
            self.t_W_fc1.assign(self.W_fc1), self.t_b_fc1.assign(self.b_fc1),
            self.t_W_fc2.assign(self.W_fc2), self.t_b_fc2.assign(self.b_fc2)
        ])

        if self.slow:
            self.update_target_op = [
                self.t_W_conv1.assign(self.tau*self.W_conv1 + (1-self.tau)*self.t_W_conv1),
                self.t_b_conv1.assign(self.tau*self.b_conv1 + (1-self.tau)*self.t_b_conv1),
                self.t_W_conv2.assign(self.tau*self.W_conv2 + (1-self.tau)*self.t_W_conv2),
                self.t_b_conv2.assign(self.tau*self.b_conv2 + (1-self.tau)*self.t_b_conv2),
                self.t_W_conv3.assign(self.tau*self.W_conv3 + (1-self.tau)*self.t_W_conv3),
                self.t_b_conv3.assign(self.tau*self.b_conv3 + (1-self.tau)*self.t_b_conv3),
                self.t_W_fc1.assign(self.tau*self.W_fc1 + (1-self.tau)*self.t_W_fc1),
                self.t_b_fc1.assign(self.tau*self.b_fc1 + (1-self.tau)*self.t_b_fc1),
                self.t_W_fc2.assign(self.tau*self.W_fc2 + (1-self.tau)*self.t_W_fc2),
                self.t_b_fc2.assign(self.tau*self.b_fc2 + (1-self.tau)*self.t_b_fc2),
            ]
        else:
            self.update_target_op = [
                self.t_W_conv1.assign(self.W_conv1), self.t_b_conv1.assign(self.b_conv1),
                self.t_W_conv2.assign(self.W_conv2), self.t_b_conv2.assign(self.b_conv2),
                self.t_W_conv3.assign(self.W_conv3), self.t_b_conv3.assign(self.b_conv3),
                self.t_W_fc1.assign(self.W_fc1), self.t_b_fc1.assign(self.b_fc1),
                self.t_W_fc2.assign(self.W_fc2), self.t_b_fc2.assign(self.b_fc2),
            ]

        self.saver = tf.train.Saver()
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.path + self.folder + '/log_tb', self.sess.graph)

    def evaluate(self, state):
        return self.sess.run(self.q_value, feed_dict={self.observation: state})

    def evaluate_target(self, state):
        return self.sess.run(self.t_q_value, feed_dict={self.next_observation: state})

    def build_loss(self, error_clip, n_actions):
        with tf.name_scope("Cost") as scope:
            predictions = tf.reduce_sum(tf.mul(self.q_value, self.actions), reduction_indices=1)
            #delta = self.q_values_t - predictions
            #clipped_error = tf.select(
            #    tf.abs(delta) < 1.0,
            #    0.5 * tf.square(delta),
            #    tf.abs(delta) - 0.5, name='clipped_error'
            #)
            max_action_values = tf.reduce_max(self.t_q_value, 1)
            targets = tf.stop_gradient(self.rewards + (self.gamma * max_action_values * (1 - self.terminals)))
            difference = tf.abs(targets - predictions)
            if error_clip >= 0:
                quadratic_part = tf.clip_by_value(difference, 0, error_clip)
                linear_part = difference - quadratic_part
                errors = (0.5 * tf.square(quadratic_part)) + (error_clip * linear_part)
            else:
                errors = (0.5 * tf.square(difference))
            #self.cost = tf.reduce_mean(clipped_error, name='loss')
            cost = tf.reduce_sum(errors, name='loss')
            cost_summ = tf.summary.scalar("cost", cost)
            return cost

    def train(self, s_j_batch, a_batch, r_batch, s_j1_batch, terminal, total_reward):
        t_ops = [self.merged, self.train_step, self.cost, self.accuracy]
        summary = self.sess.run(
            t_ops,
            feed_dict={
                self.observation: s_j_batch,
                self.actions: a_batch,
                self.next_observation: s_j1_batch,
                self.rewards: r_batch,
                self.terminals: terminal,
                self.accuracy: total_reward
            }
        )
        if self.update_counter % self.copy_interval == 0:
            if not self.slow:
                print colored('Update target network', 'green')
            self.update_target_network()
        self.update_counter += 1
        return summary[0]

    def add_summary(self, summary, step):
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def update_target_network(self):
        self.sess.run(self.update_target_op)

    def load(self, folder='_networks'):
        has_checkpoint = False
        # saving and loading networks
        checkpoint = tf.train.get_checkpoint_state(self.folder)

        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print colored('Successfully loaded:{}'.format(checkpoint.model_checkpoint_path), 'green')
            sleep(.2)
            has_checkpoint = True
            data = pickle.load(open(self.folder + '/' + self.name + '-net-variables.pkl', 'rb'))
            self.update_counter = data['update_counter']

        return has_checkpoint

    def save(self, step=-1):
        print colored('Saving checkpoint...', 'blue')
        if step < 0:
            self.saver.save(self.sess, self.folder + '/' + self.name + '-dqn')
        else:
            self.saver.save(self.sess, self.folder + '/' + self.name + '-dqn', global_step=step)
            data = {'update_counter': self.update_counter}
            pickle.dump(data, open(self.folder + '/' + self.name + '-net-variables.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        print colored('Successfully saved checkpoint!', 'green')

        print colored('Saving parameters as csv files...', 'blue')
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
        print colored('Successfully saved parameters!', 'green')

        print colored('Saving convolutional weights as images...', 'blue')
        conv_weights = self.sess.run([tf.get_collection('conv_weights')])
        for i, c in enumerate(conv_weights[0]):
            plot_conv_weights(c, 'conv{}'.format(i+1), folder=self.folder)
        print colored('Successfully saved convolutional weights!', 'green')

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

    def load_transfer_model(self, folder=''):
        assert folder != ''
        saver_transfer_from = tf.train.Saver()
        checkpoint_transfer_from = tf.train.get_checkpoint_state(folder)
        if checkpoint_transfer_from and checkpoint_transfer_from.model_checkpoint_path:
            saver_transfer_from.restore(self.sess, checkpoint_transfer_from.model_checkpoint_path)
            print colored("Successfully loaded: {}".format(checkpoint_transfer_from.model_checkpoint_path), "green")

            for v in tf.global_variables():
                self.sess.run(v)
                print colored("{}: LOADED".format(v.op.name), "green")
                sleep(.2)
