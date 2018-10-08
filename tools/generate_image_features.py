import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import cv2
import numpy as np

from game_state import GameState
from data_set import DataSet
from termcolor import colored

from common.util import get_activations, montage, get_compressed_images

def view_features(args):
    try:
        import cPickle as pickle
    except ImportError:
        import pickle

    assert args.demo_env != ''
    folder = "demo_samples/{}".format(args.demo_env.replace('-', '_'))
    D = DataSet()
    data = pickle.load(open(folder + '/001/' + args.demo_env + '-dqn.pkl', 'rb'))
    D.width = data['D.width']
    D.height = data['D.height']
    D.max_steps = data['D.max_steps']
    D.phi_length = data['D.phi_length']
    D.num_actions = data['D.num_actions']
    D.actions = data['D.actions']
    D.rewards = data['D.rewards']
    D.terminal = data['D.terminal']
    D.size = data['D.size']
    D.imgs = get_compressed_images(folder + '/001/' + args.demo_env + '-dqn-images.h5' + '.gz')

    D.normalize_images()
    state, action, _, _ = D[20]
    # print ("action: {}".format(action))
    # print (np.shape(state[:,:,0]))
    # cv2.imshow("0", state[:,:,0])
    # cv2.imshow("1", state[:,:,1])
    # cv2.waitKey(0)

    from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
    game_state = GameState(env_id=args.env)
    action_size = game_state.env.action_space.n
    game_state.env.close()
    del game_state.env
    del game_state

    config = tf.ConfigProto(
        gpu_options=None,
        log_device_placement=False,
        allow_soft_placement=True)

    GameACLSTMNetwork.use_mnih_2015 = args.use_mnih_2015
    global_network = GameACLSTMNetwork(action_size, -1, "/cpu:0")

    # prepare session
    sess = tf.Session(config=config)

    def initialize_uninitialized(sess):
        global_vars = tf.global_variables()
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

        #print [str(i.name) for i in not_initialized_vars] # only for testing
        if len(not_initialized_vars):
            sess.run(tf.variables_initializer(not_initialized_vars))

    W_init = None
    if args.use_transfer:
        transfer_var_list = [
            global_network.W_conv1, global_network.b_conv1,
            global_network.W_conv2, global_network.b_conv2,
            global_network.W_fc1, global_network.b_fc1
        ]
        if args.use_mnih_2015:
            transfer_var_list += [
                global_network.W_conv3, global_network.b_conv3
            ]
        transfer_folder = args.transfer_folder
        global_network.load_transfer_model(
            sess, folder=transfer_folder,
            not_transfer_fc2=True,
            not_transfer_fc1=False,
            not_transfer_conv3=False,
            not_transfer_conv2=False,
            var_list=transfer_var_list
        )
        initialize_uninitialized(sess)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    if args.view_conv1:
        weights = global_network.W_conv1
    elif args.view_conv2:
        weights = global_network.W_conv2
    elif args.view_conv3:
        weights = global_network.W_conv3
    else:
        weights = global_network.W_fc1
    W_init = sess.run(weights)

    if args.folder != '':
        print (args.folder)
        # init or load checkpoint with saver
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(args.folder)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print(colored("checkpoint loaded:{}".format(checkpoint.model_checkpoint_path), "green"))

    import matplotlib.pyplot as plt
    #get_activations(sess, global_network.h_conv3, state * (1.0/255.0), global_network.s, global_network.keep_prob)

    visual = True
    if args.view_conv1:
        print ("Visualizing weights for convolution layer 1")
        weights = global_network.W_conv1
    elif args.view_conv2:
        print ("Visualizing weights for convolution layer 2")
        weights = global_network.W_conv2
    elif args.view_conv3:
        print ("Visualizing weights for convolution layer 3")
        weights = global_network.W_conv3
    else:
        visual = False
        weights = global_network.W_fc1
    W = sess.run(weights)
    #W = W/(np.max(W))
    if visual:
        plt.imshow(montage(W/(np.max(W))), cmap='coolwarm')
        plt.colorbar()
        plt.savefig('weights.eps', format='eps', dpi=1000)
        plt.show(block=True)


    print (np.shape(W))
    if W_init is not None:
        from scipy import spatial
        from common.util import Similarity
        measures = Similarity()
        W_flatten = W.flatten()
        W_init_flatten = W_init.flatten()
        W_flatten.dtype=np.float64
        W_init_flatten.dtype=np.float64
        print(np.shape(W_flatten), np.shape(W_init_flatten))
        # result_euclidean_distance = measures.euclidean_distance(W_flatten, W_init_flatten)
        # print ('euclidean_distance:{}'.format(result_euclidean_distance))
        # print ('mse:{}'.format(mse(W_flatten, W_init_flatten)))
        # print ('cosine_scipy:{}'.format(1-spatial.distance.cosine(W_flatten, W_init_flatten)))
        # print ('correlation_scipy:{}'.format(1-spatial.distance.correlation(W_flatten, W_init_flatten)))

        W_flat = np.reshape(W, [W.shape[0], W.shape[1], 1, W.shape[2] * W.shape[3]])
        n_plots = int(np.ceil(np.sqrt(W_flat.shape[-1])))
        print (W_flat.shape)
        for i in range(n_plots):
            for j in range(n_plots):
                this_filter = i * n_plots + j
                if this_filter < W_flat.shape[-1]:
                    #print (this_filter, np.squeeze(W_flat[:, :, :, this_filter]).shape)
                    pass
        # invcovar = np.linalg.inv(np.cov(np.vstack([W_flatten,W_init_flatten]).T)).T
        # print (invcovar)
        # print (spatial.distance.mahalanobis(W_flatten, W_init_flatten, invcovar))


from skimage.measure import structural_similarity as ssim
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--demo-env', type=str, default='')
    parser.add_argument('--folder', type=str, default='')
    parser.add_argument('--transfer-folder', type=str, default='')
    parser.add_argument('--use-transfer', action='store_true')
    parser.set_defaults(use_transfer=False)
    parser.add_argument('--use-mnih-2015', action='store_true', help='use Mnih et al [2015] network architecture (3 conv layers)')

    parser.add_argument('--view-conv1', action='store_true')
    parser.set_defaults(view_conv1=False)
    parser.add_argument('--view-conv2', action='store_true')
    parser.set_defaults(view_conv2=False)
    parser.add_argument('--view-conv3', action='store_true')
    parser.set_defaults(view_conv3=False)

    parser.set_defaults(use_mnih_2015=False)
    args = parser.parse_args()

    view_features(args)
