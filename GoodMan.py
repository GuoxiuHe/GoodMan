#! /user/bin/evn python3
# -*- coding:utf8 -*-

'''
This is the code of GoodMan, which is proposed in
*Creating a Children-Friendly Reading Environment via Joint Learning of Content and Human Attention*.
This paper is accepted as a full paper in SIGIR 2020.
'''

import os, sys
import numpy as np
import tensorflow as tf
tf.set_random_seed(1234) # the random seed of the tensorflow

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
rootdir = '/'.join(curdir.split('/')[:4])
prodir = '../..'
sys.path.insert(0, '..')
sys.path.insert(0, prodir)

class GoodMan(object):

    def __init__(self, maxlen=150, nb_classes=2, nb_words=238083,
                 embedding_dim=64, dense_dim=64, rnn_dim=10, cnn_filters=64,
                 dropout_rate=0.9, learning_rate=0.001,
                 weight_decay=0.0, optim_type='adam',
                 gpu='0', memory=0, **kwargs):

        # experiment settings

        # data settings
        self.nb_classes = nb_classes
        self.nb_words = nb_words
        self.maxlen = maxlen
        self.sentmaxlen = 40

        # model settings
        self.embedding_dim = embedding_dim
        self.dense_dim = dense_dim
        self.rnn_dim = rnn_dim
        self.cnn_filters = cnn_filters
        self.hidden_size = dense_dim

        # training settings
        self.initializer = tf.truncated_normal_initializer(stddev=0.02)
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optim_type = optim_type

        # the model name
        self.model_name = 'GoodMan'

        # GUP settings
        self.gpu = gpu
        self.memory = memory
        if self.memory > 0:
            # fixed gpu memory
            num_threads = os.environ.get('OMP_NUM_THREADS')
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=float(self.memory))
            config = tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads)
            self.sess = tf.Session(config=config)
        else:
            # flexible gpu memory
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

    # set vocabulary size outside the code
    def set_nb_words(self, nb_words):
        self.nb_words = nb_words

    def set_data_name(self, data_name):
        self.data_name = data_name

    def set_name(self, model_name):
        self.model_name = model_name

    # set the model settings outside the code
    def set_from_model_config(self, model_config):
        self.embedding_dim = model_config['embedding_dim']
        self.rnn_dim = model_config['rnn_dim']
        self.dense_dim = model_config['dense_dim']
        self.cnn_filters = model_config['cnn_filters']
        self.dropout_rate = model_config['dropout_rate']
        self.optim_type = model_config['optimizer']
        self.learning_rate = model_config['learning_rate']
        self.weight_decay = model_config['weight_decay']

    # set the data settings outside the code
    def set_from_data_config(self, data_config):
        self.nb_classes = data_config['nb_classes']

    # optimizer settings
    def _create_train_op(self):
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.train_op = self.optimizer.minimize(self.loss)

    # random initialize word embeddings or use a pre-trained embedding matrix
    def _embed(self, embedding_matrix=np.array([None])):
        with tf.variable_scope('word_embedding'):
            if embedding_matrix.any() == None:
                self.word_embeddings = tf.get_variable(
                    'word_embeddings',
                    shape=(self.nb_words, self.embedding_dim),
                    initializer=self.initializer,
                    trainable=True
                )
            else:
                self.word_embeddings = tf.get_variable(
                    'word_embeddings',
                    shape=(self.nb_words, self.embedding_dim),
                    initializer=tf.constant_initializer(embedding_matrix),
                    trainable=True
                )

    # the data placeholders for end-to-end training
    def _setup_placeholders(self):
        self.input_x = tf.placeholder(tf.int32, [None, None, None], name="input_x") # content
        self.input_b = tf.placeholder(tf.float32, [None, None], name='input_b') # behavior
        self.input_y = tf.placeholder(tf.int32, [None, self.nb_classes], name="input_y") # label
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob") # dropout

    def _inference(self):
        # the shape is (batch size, number of pages, number of words per page)
        (self.batch_size, self.sentmaxlen, self.maxlen) = tf.unstack(tf.shape(self.input_x))
        # get the word level embedding matrix
        self.input_x_reshape = tf.reshape(self.input_x, shape=[-1, self.maxlen])
        self.embedded = tf.nn.embedding_lookup(self.word_embeddings, self.input_x_reshape)

        self.encoder_inputs = self.embedded

        # Hierarchical Encoder
        # first level encoding
        with tf.variable_scope('word'):
            with tf.variable_scope('near_neighbor'):
                self.word_nn_output = tf.keras.layers.Conv1D(filters=self.hidden_size,
                                                             kernel_size=3,
                                                             padding='same',
                                                             activation='relu',
                                                             name='word_nn'
                                                             )(self.encoder_inputs)
            self.word_level_output = tf.keras.layers.GlobalAveragePooling1D()(self.word_nn_output)

        self.sentence_inputs = tf.reshape(self.word_level_output,
                                          [-1, self.sentmaxlen, self.hidden_size])

        # second level encoding
        with tf.variable_scope('sentence'):

            with tf.variable_scope('near_neighbor'):
                self.sentence_nn_output = tf.keras.layers.Conv1D(filters=self.hidden_size,
                                                                 kernel_size=3,
                                                                 padding='same',
                                                                 activation='relu',
                                                                 name='sentence_nn'
                                                                 )(self.sentence_inputs)
            self.sentence_level_encoding = self.sentence_nn_output

            # Human Attention Component
            with tf.variable_scope('attention'):
                self.input_b_atten = self.input_b
                self.input_b_atten_expanded = tf.expand_dims(self.input_b_atten, axis=-1)
                self.mask = tf.cast(self.input_b_atten_expanded > 0, tf.float32)
                self.smooth_encoding = tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(self.rnn_dim,
                                        return_sequences=True))(self.input_b_atten_expanded)
                self.smooth_b_tmp = tf.keras.layers.Dense(1, activation='relu')(self.smooth_encoding)
                self.smooth_b = tf.nn.softmax(self.smooth_b_tmp, axis=1)
                self.smooth_b = tf.multiply(self.smooth_b, self.mask)

                self.multiply_sent_atten = tf.multiply(self.sentence_level_encoding, self.smooth_b)
                self.sentence_level_output = tf.reduce_sum(self.multiply_sent_atten, axis=1)

            # Content Attention Component
            with tf.variable_scope('attention2'):
                self.content_atten_encoding = tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(self.rnn_dim,
                                        return_sequences=True))(self.sentence_level_encoding)

                self.content_atten_tmp = tf.keras.layers.Dense(1, activation='relu')(self.content_atten_encoding)

                self.content_atten = tf.nn.softmax(self.content_atten_tmp, 1)
                self.content_atten = tf.multiply(self.content_atten, self.mask)

                self.sentence_level_output_tmp = tf.multiply(self.sentence_level_encoding, self.content_atten)
                self.sentence_level_output_1 = tf.reduce_sum(self.sentence_level_output_tmp, axis=1)

            # Joint Attention Component
            with tf.variable_scope('joint'):
                self.joint_atten_tmp = tf.concat([self.content_atten_encoding, self.smooth_encoding], axis=-1)
                self.joint_atten_encoding = tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(self.rnn_dim,
                                        return_sequences=True))(self.joint_atten_tmp)
                self.joint_atten_encoding_tmp = tf.keras.layers.Dense(1, activation='relu')(self.joint_atten_encoding)
                self.joint_atten = tf.nn.softmax(self.joint_atten_encoding_tmp, axis=1)
                self.joint_atten = tf.multiply(self.joint_atten, self.mask)
                
                self.joint_encoding = tf.multiply(self.sentence_level_encoding, self.joint_atten)
                self.sentence_level_output_2 = tf.reduce_sum(self.joint_encoding, axis=1)

        # the other two combinations of these representations mentioned above
        self.add_feature = tf.add(self.sentence_level_output, self.sentence_level_output_1) / 2
        self.add_feature2 = tf.add(tf.add(self.sentence_level_output, 
                                          self.sentence_level_output_1), 
                                   self.sentence_level_output_2) / 3
        # the final representation
        self.final_feature = tf.concat([self.add_feature,
                                        self.add_feature2, 
                                        self.sentence_level_output, 
                                        self.sentence_level_output_1, 
                                        self.sentence_level_output_2], 
                                       axis=-1)
        self.final_feature = tf.keras.layers.Dense(self.hidden_size, activation='relu')(self.final_feature)

        if self.dropout_rate:
            self.final_feature = tf.nn.dropout(self.final_feature, keep_prob=self.dropout_keep_prob)

        with tf.variable_scope("output"):

            self.logits = tf.keras.layers.Dense(self.nb_classes)(self.final_feature)

            # human behavior based output
            self.logits2 = tf.keras.layers.Dense(self.nb_classes)(self.sentence_level_output)
            # content based output
            self.logits3 = tf.keras.layers.Dense(self.nb_classes)(self.sentence_level_output_1)

            self.output = tf.argmax(self.logits, axis=-1)
            self.proba = tf.nn.softmax(self.logits, axis=-1)

        return self.logits

    # Joint Learning Component
    def _compute_loss(self):
        # the final representation based loss
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y,
                                                       logits=self.logits))

        # the human attention and content attention can benefit from the joint learning
        # human behavior based loss
        self.loss2 = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y,
                                                       logits=self.logits2))
        # content based loss
        self.loss3 = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y,
                                                       logits=self.logits3))

        self.all_params = tf.trainable_variables()
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss

        # total loss
        self.loss = self.loss + self.loss2 + self.loss3

    # to build the graph
    def build_graph(self):
        self._setup_placeholders()
        self._embed()
        self._inference()
        self._compute_loss()
        self._create_train_op()

        # please set your own path of model saving
        self.save_dir = prodir + '/Networks/weights/' + self.data_name + '/' + self.model_name + '/'
        if not os.path.exists(prodir + '/Networks/weights/'):
            os.makedirs(prodir + '/Networks/weights/')
        if not os.path.exists(prodir + '/Networks/weights/' + self.data_name):
            os.makedirs(prodir + '/Networks/weights/' + self.data_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.saver = tf.train.Saver()
        # Initialize the model
        self.sess.run(tf.global_variables_initializer())


if __name__ == '__main__':
    network = GoodMan()
    network.build_graph()
