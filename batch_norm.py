import tensorflow as tf

class batch_norm:
    def __init__(
        self, inputs, size, is_training,sess, parForTarget=None,
        decay=0.9, epsilon=1e-4, slow=False, tau=0.01, linear=False):
        """ Initialization of batch_norm class """

        self.slow = slow
        self.sess = sess
        self.scale = tf.Variable(tf.random_uniform([size],0.9,1.1), trainable=True, name='scale')
        #self.scale = tf.Variable(tf.constant(1.0, shape=[size]), name='scale', trainable=True)
        self.beta = tf.Variable(tf.random_uniform([size],-0.03,0.03), trainable=True, name='beta')
        #self.beta = tf.Variable(tf.constant(0.0, shape=[size]), name='beta', trainable=True)
        self.pop_mean = tf.Variable(tf.random_uniform([size],-0.03,0.03), trainable=False, name='mean')
        #self.pop_mean = tf.Variable(tf.constant(0.0, shape=[size]),trainable=False, name='mean')
        self.pop_var = tf.Variable(tf.random_uniform([size],0.9,1.1), trainable=False, name='variance')
        #self.pop_var = tf.Variable(tf.constant(1.0, shape=[size]),trainable=False, name='variance')
        if linear:
            self.batch_mean, self.batch_var = tf.nn.moments(inputs,[0])
        else:
            self.batch_mean, self.batch_var = tf.nn.moments(inputs,[0,1,2])
        self.train_mean = tf.assign(self.pop_mean,self.pop_mean * decay + self.batch_mean * (1 - decay))
        self.train_var = tf.assign(self.pop_var,self.pop_var * decay + self.batch_var * (1 - decay))
        self.train = tf.group(self.train_mean, self.train_var)

        def training():
            return tf.nn.batch_normalization(inputs,
                self.batch_mean, self.batch_var, self.beta, self.scale, epsilon)

        def testing():
            return tf.nn.batch_normalization(inputs,
            self.pop_mean, self.pop_var, self.beta, self.scale, epsilon)

        if parForTarget!=None:
            self.parForTarget = parForTarget
            if self.slow:
                self.updateScale = self.scale.assign(self.scale*(1-tau)+self.parForTarget.scale*tau)
                self.updateBeta = self.beta.assign(self.beta*(1-tau)+self.parForTarget.beta*tau)
            else:
                self.updateScale = self.scale.assign(self.parForTarget.scale)
                self.updateBeta = self.beta.assign(self.parForTarget.beta)
            self.updateTarget = tf.group(self.updateScale, self.updateBeta)

        self.bnorm = tf.cond(is_training,training,testing)
