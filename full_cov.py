import config as cfg
import threading
import time
import tensorflow as tf
import layer_maker_tradition
import numpy as np
import time
import input_data_force3
from math import ceil
import logging

class diguinet(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(diguinet, self).__init__(*args, **kwargs)
        self.__flag = threading.Event()
        self.__flag.set()
        self.__running = threading.Event()
        self.__running.set()
        self.__save = threading.Event()




    def run(self):
        rootdir = '../../ADE20K_2016_07_26/images/validation/'
        if cfg.TEST_MODE == 'on':
            cfg.BATCH_SIZE = 3
            print('TEST_MODE ON')
            rootdir = './ADETEST'
        ade20k = input_data_force3.ADE20K(rootdir, './fcs.xlsx')

        lm = layer_maker_tradition.layermaker(cfg.TRUNCATES_MEAN, cfg.TRUNCATES_STTDEV, cfg.BIAS, 'the_first')
        images512_ = tf.placeholder('float', shape=[None, 512, 512, 3])
        images512_ = images512_ / 128.0 - 1.0
        labels = tf.placeholder('float', [None, 128, 128, 3])
        masks_ = tf.placeholder('float', [None, 128, 128])

        images256_ = lm.aver_pool_layer(images512_)
        images128_ = lm.aver_pool_layer(images256_)
        images64_ = lm.aver_pool_layer(images128_)
        images32_ = lm.aver_pool_layer(images64_)
        images16_ = lm.aver_pool_layer(images32_)

        images128 = lm.pool_layer(lm.conv_layer(lm.pool_layer((lm.conv_layer(images512_,3,3,32))),3,32,32))
        images64 = lm.pool_layer(lm.conv_layer(lm.pool_layer((lm.conv_layer(images256_,3,3,32))),3,32,32))
        images32 = lm.pool_layer(lm.conv_layer(lm.pool_layer((lm.conv_layer(images128_,3,3,32))),3,32,32))
        images16 = lm.pool_layer(lm.conv_layer(lm.pool_layer((lm.conv_layer(images64_,3,3,32))),3,32,32))
        images8 = lm.pool_layer(lm.conv_layer(lm.pool_layer((lm.conv_layer(images32_,3,3,32))),3,32,32))
        images4 = lm.pool_layer(lm.conv_layer(lm.pool_layer((lm.conv_layer(images16_,3,3,32))),3,32,32))


        chanel = 64
        tensor = lm.conv_layer(images4, 3, 32, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor = lm.bilinear_upsample_layer(tensor, chanel, 8)
        tensor = tf.concat([tensor, images8], axis=3)

        chanel = 128
        tensor = lm.conv_layer(tensor, 3, 96, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor = lm.bilinear_upsample_layer(tensor, chanel, 16)
        tensor = tf.concat([tensor, images16], axis=3)

        chanel = 256
        tensor = lm.conv_layer(tensor, 3, 160, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor = lm.bilinear_upsample_layer(tensor, chanel, 32)
        tensor = tf.concat([tensor, images32], axis=3)

        chanel = 512
        tensor = lm.conv_layer(tensor, 3, 288, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor = lm.bilinear_upsample_layer(tensor, chanel, 64)
        tensor = tf.concat([tensor, images64], axis=3)

        chanel = 512
        tensor = lm.conv_layer(tensor, 3, 544, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor = lm.bilinear_upsample_layer(tensor, chanel, 128)
        tensor = tf.concat([tensor, images128], axis=3)

        chanel = 256
        tensor = lm.conv_layer(tensor, 3, 544, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor1 = lm.conv_layer(tensor, 3, chanel, 2)

        #tensor2 = lm.conv_layer(tensor, 3, chanel, 1, linear=True)
        #tensor2 = lm.bilinear_upsample_layer(tensor2, 1, 512)

        #soliedornot = tf.reshape(tf.maximum(tf.minimum(tensor2, 1), -1), [-1, 512, 512])

        labels_elaster = (tf.log(labels[:, :, :, 0]) / tf.log(5.0))
        labels_poisson = tf.maximum(2 * labels[:, :, :, 1], 1.0 / 3 * labels[:, :, :, 1])
        loss_elaster = tf.square(labels_elaster - tensor1[:, :, :, 0])
        loss_poisson = tf.square(labels_poisson - tensor1[:, :, :, 1])
        #loss_soliedornot = tf.square(labels[:, :, :, 2] - soliedornot)
        #temp_loss = (loss_elaster + loss_poisson) * masks_ + loss_soliedornot
        loss_el = tf.reduce_sum(loss_elaster * masks_)
        loss_po = tf.reduce_sum(loss_poisson*masks_)

        train_step_el = tf.train.AdamOptimizer(cfg.TRAIN_STEP).minimize(loss_el)
        train_step_po = tf.train.AdamOptimizer(cfg.TRAIN_STEP).minimize(loss_po)
        sess = tf.Session()
        saver = tf.train.Saver(max_to_keep=5)
        if cfg.IS_RESTORE:
            saver.restore(sess, './weight/lixue.cktp-' + str(cfg.SAVED_WEIGHT_STEPS))
            print('restore '+'./weight/lixue.cktp-' + str(cfg.SAVED_WEIGHT_STEPS))
        else:
            sess.run(tf.global_variables_initializer())

        print('Network has been build!')

        time_n = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
        log_file_name = './log/'+time_n+'.txt'
        self.log_file = open(log_file_name,'w')


        batch = ade20k.next_batch()
        for i in range(1, cfg.EPOCHS):

            if self.__running.isSet() == False:
                self.log_file.close()
                print('log saved')
                break
            self.__flag.wait()

            if (i-1)%cfg.LOSS_PRINT_STEPS == 0:
                sunshi = sess.run([loss_el,loss_po],feed_dict={images512_: batch[0], labels: batch[1][:, :, :, 0:3],masks_: batch[1][:, :, :, 3]})
                loss_info = 'EPOCHS' + str(i) + ':' + str(sunshi)
                self.log_file.write(loss_info+'\n')
                print(loss_info)

     

            sess.run([train_step_el,train_step_po],
                     feed_dict={images512_: batch[0], labels: batch[1][:, :, :, 0:3], masks_: batch[1][:, :, :, 3]})

            if i%cfg.SAVE_STEPS == 0 or self.__save.isSet():
                
                saver.save(sess, './weight/lixue.cktp', global_step=i)
                save_info = 'Saved:./weight/lixue.cktp' + str(i)
                self.log_file.write(save_info+'\n')
                self.log_file.close()
                self.log_file = open(log_file_name, 'a')
                print(save_info)
                self.__save.clear()
                cfg.SAVED_WEIGHT_STEPS = i

                time.sleep(300)

            if cfg.TEST_MODE == 'on':
                pass
            else:
                batch = ade20k.next_batch()

    def pause(self):
        self.__flag.clear()

    def resume(self):
        self.__flag.set()

    def save(self):
        self.__save.set()

    def stop(self):
        self.__flag.set()
        self.__running.clear()

    def _activation_summary(x):
        """Helper to create summaries for activations.

        Creates a summary that provides a histogram of activations.
        Creates a summary that measure the sparsity of activations.

        Args:
          x: Tensor
        Returns:
          nothing
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        tensor_name = x.op.name
        # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def build(self, bgr, train=False, num_classes=20, random_init_fc8=False,
              debug=False, use_dilated=False):
        """
        Build the VGG model using loaded weights
        Parameters
        ----------
        rgb: image batch tensor
            Image in rgb shap. Scaled to Intervall [0, 255]
        train: bool
            Whether to build train or inference graph
        num_classes: int
            How many classes should be predicted (by fc8)
        random_init_fc8 : bool
            Whether to initialize fc8 layer randomly.
            Finetuning is required in this case.
        debug: bool
            Whether to print additional Debug Information.
        """
        # Convert RGB to BGR


        self.conv1_1 = self._conv_layer(bgr, "conv1_1")
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self._max_pool(self.conv1_2, 'pool1', debug)

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self._max_pool(self.conv2_2, 'pool2', debug)

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self._max_pool(self.conv3_3, 'pool3', debug)

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")

        if use_dilated:
            pad = [[0, 0], [0, 0]]
            self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME', name='pool4')
            self.pool4 = tf.space_to_batch(self.pool4,
                                           paddings=pad, block_size=2)
        else:
            self.pool4 = self._max_pool(self.conv4_3, 'pool4', debug)

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
        if use_dilated:
            pad = [[0, 0], [0, 0]]
            self.pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME', name='pool5')
            self.pool5 = tf.space_to_batch(self.pool5,
                                           paddings=pad, block_size=2)
        else:
            self.pool5 = self._max_pool(self.conv5_3, 'pool5', debug)

        self.fc6 = self._fc_layer(self.pool5, "fc6")

        if train:
            self.fc6 = tf.nn.dropout(self.fc6, 0.5)

        self.fc7 = self._fc_layer(self.fc6, "fc7")
        if train:
            self.fc7 = tf.nn.dropout(self.fc7, 0.5)

        if use_dilated:
            self.pool5 = tf.batch_to_space(self.pool5, crops=pad, block_size=2)
            self.pool5 = tf.batch_to_space(self.pool5, crops=pad, block_size=2)
            self.fc7 = tf.batch_to_space(self.fc7, crops=pad, block_size=2)
            self.fc7 = tf.batch_to_space(self.fc7, crops=pad, block_size=2)
            return

        if random_init_fc8:
            self.score_fr = self._score_layer(self.fc7, "score_fr",
                                              num_classes)
        else:
            self.score_fr = self._fc_layer(self.fc7, "score_fr",
                                           num_classes=num_classes,
                                           relu=False)

        self.upscore2 = self._upscore_layer(self.score_fr,
                                            shape=tf.shape(self.pool4),
                                            num_classes=num_classes,
                                            debug=debug, name='upscore2',
                                            ksize=4, stride=2)
        self.score_pool4 = self._score_layer(self.pool4, "score_pool4",
                                             num_classes=num_classes)
        self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)

        self.upscore4 = self._upscore_layer(self.fuse_pool4,
                                            shape=tf.shape(self.pool3),
                                            num_classes=num_classes,
                                            debug=debug, name='upscore4',
                                            ksize=4, stride=2)
        self.score_pool3 = self._score_layer(self.pool3, "score_pool3",
                                             num_classes=num_classes)
        self.fuse_pool3 = tf.add(self.upscore4, self.score_pool3)

        self.upscore32 = self._upscore_layer(self.fuse_pool3,
                                             shape=tf.shape(bgr),
                                             num_classes=num_classes,
                                             debug=debug, name='upscore32',
                                             ksize=16, stride=8)

    def _max_pool(self, bottom, name, debug):
            pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                  padding='SAME', name=name)

            if debug:
                pool = tf.Print(pool, [tf.shape(pool)],
                                message='Shape of %s' % name,
                                summarize=4, first_n=1)
            return pool

    def _conv_layer(self, bottom, name):
            with tf.variable_scope(name) as scope:
                filt = self.get_conv_filter(name)
                conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

                conv_biases = self.get_bias(name)
                bias = tf.nn.bias_add(conv, conv_biases)

                relu = tf.nn.relu(bias)
                # Add summary to Tensorboard
                self._activation_summary(relu)
                return relu

    def _fc_layer(self, bottom, name, num_classes=None,
                      relu=True, debug=False):
            with tf.variable_scope(name) as scope:
                shape = bottom.get_shape().as_list()

                if name == 'fc6':
                    filt = self.get_fc_weight_reshape(name, [7, 7, 512, 4096])
                elif name == 'score_fr':
                    name = 'fc8'  # Name of score_fr layer in VGG Model
                    filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 1000],
                                                      num_classes=num_classes)
                else:
                    filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096])
                conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
                conv_biases = self.get_bias(name, num_classes=num_classes)
                bias = tf.nn.bias_add(conv, conv_biases)

                if relu:
                    bias = tf.nn.relu(bias)
                self._activation_summary(bias)

                if debug:
                    bias = tf.Print(bias, [tf.shape(bias)],
                                    message='Shape of %s' % name,
                                    summarize=4, first_n=1)
                return bias

    def _score_layer(self, bottom, name, num_classes):
            with tf.variable_scope(name) as scope:
                # get number of input channels
                in_features = bottom.get_shape()[3].value
                shape = [1, 1, in_features, num_classes]
                # He initialization Sheme
                num_input = in_features
                stddev = (2 / num_input) ** 0.5
                # Apply convolution
                w_decay = self.wd
                weights = self._variable_with_weight_decay(shape, stddev, w_decay)
                conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
                # Apply bias
                conv_biases = self._bias_variable([num_classes], constant=0.0)
                bias = tf.nn.bias_add(conv, conv_biases)

                self._activation_summary(bias)

                return bias

    def _upscore_layer(self, bottom, shape,
                           num_classes, name, debug,
                           ksize=4, stride=2):
            strides = [1, stride, stride, 1]
            with tf.variable_scope(name):
                in_features = bottom.get_shape()[3].value

                if shape is None:
                    # Compute shape out of Bottom
                    in_shape = tf.shape(bottom)

                    h = ((in_shape[1] - 1) * stride) + 1
                    w = ((in_shape[2] - 1) * stride) + 1
                    new_shape = [in_shape[0], h, w, num_classes]
                else:
                    new_shape = [shape[0], shape[1], shape[2], num_classes]
                output_shape = tf.stack(new_shape)

                logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
                f_shape = [ksize, ksize, num_classes, in_features]

                # create
                num_input = ksize * ksize * in_features / stride
                stddev = (2 / num_input) ** 0.5

                weights = self.get_deconv_filter(f_shape)
                deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                                strides=strides, padding='SAME')

                if debug:
                    deconv = tf.Print(deconv, [tf.shape(deconv)],
                                      message='Shape of %s' % name,
                                      summarize=4, first_n=1)

            self._activation_summary(deconv)
            return deconv

    def get_deconv_filter(self, f_shape):
            width = f_shape[0]
            height = f_shape[1]
            f = ceil(width / 2.0)
            c = (2 * f - 1 - f % 2) / (2.0 * f)
            bilinear = np.zeros([f_shape[0], f_shape[1]])
            for x in range(width):
                for y in range(height):
                    value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                    bilinear[x, y] = value
            weights = np.zeros(f_shape)
            for i in range(f_shape[2]):
                weights[:, :, i, i] = bilinear

            init = tf.constant_initializer(value=weights,
                                           dtype=tf.float32)
            return tf.get_variable(name="up_filter", initializer=init,
                                   shape=weights.shape)

    def get_conv_filter(self, name):
            init = tf.constant_initializer(value=self.data_dict[name][0],
                                           dtype=tf.float32)
            shape = self.data_dict[name][0].shape
            print('Layer name: %s' % name)
            print('Layer shape: %s' % str(shape))
            var = tf.get_variable(name="filter", initializer=init, shape=shape)
            if not tf.get_variable_scope().reuse:
                weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                           name='weight_loss')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                     weight_decay)
            return var

    def get_bias(self, name, num_classes=None):
            bias_wights = self.data_dict[name][1]
            shape = self.data_dict[name][1].shape
            if name == 'fc8':
                bias_wights = self._bias_reshape(bias_wights, shape[0],
                                                 num_classes)
                shape = [num_classes]
            init = tf.constant_initializer(value=bias_wights,
                                           dtype=tf.float32)
            return tf.get_variable(name="biases", initializer=init, shape=shape)

    def get_fc_weight(self, name):
            init = tf.constant_initializer(value=self.data_dict[name][0],
                                           dtype=tf.float32)
            shape = self.data_dict[name][0].shape
            var = tf.get_variable(name="weights", initializer=init, shape=shape)
            if not tf.get_variable_scope().reuse:
                weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                           name='weight_loss')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                     weight_decay)
            return var

    def _bias_reshape(self, bweight, num_orig, num_new):
            """ Build bias weights for filter produces with `_summary_reshape`

            """
            n_averaged_elements = num_orig // num_new
            avg_bweight = np.zeros(num_new)
            for i in range(0, num_orig, n_averaged_elements):
                start_idx = i
                end_idx = start_idx + n_averaged_elements
                avg_idx = start_idx // n_averaged_elements
                if avg_idx == num_new:
                    break
                avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
            return avg_bweight

    def _summary_reshape(self, fweight, shape, num_new):
            """ Produce weights for a reduced fully-connected layer.

            FC8 of VGG produces 1000 classes. Most semantic segmentation
            task require much less classes. This reshapes the original weights
            to be used in a fully-convolutional layer which produces num_new
            classes. To archive this the average (mean) of n adjanced classes is
            taken.

            Consider reordering fweight, to perserve semantic meaning of the
            weights.

            Args:
              fweight: original weights
              shape: shape of the desired fully-convolutional layer
              num_new: number of new classes


            Returns:
              Filter weights for `num_new` classes.
            """
            num_orig = shape[3]
            shape[3] = num_new
            assert (num_new < num_orig)
            n_averaged_elements = num_orig // num_new
            avg_fweight = np.zeros(shape)
            for i in range(0, num_orig, n_averaged_elements):
                start_idx = i
                end_idx = start_idx + n_averaged_elements
                avg_idx = start_idx // n_averaged_elements
                if avg_idx == num_new:
                    break
                avg_fweight[:, :, :, avg_idx] = np.mean(
                    fweight[:, :, :, start_idx:end_idx], axis=3)
            return avg_fweight

    def _variable_with_weight_decay(self, shape, stddev, wd):
            """Helper to create an initialized Variable with weight decay.

            Note that the Variable is initialized with a truncated normal
            distribution.
            A weight decay is added only if one is specified.

            Args:
              name: name of the variable
              shape: list of ints
              stddev: standard deviation of a truncated Gaussian
              wd: add L2Loss weight decay multiplied by this float. If None, weight
                  decay is not added for this Variable.

            Returns:
              Variable Tensor
            """

            initializer = tf.truncated_normal_initializer(stddev=stddev)
            var = tf.get_variable('weights', shape=shape,
                                  initializer=initializer)

            if wd and (not tf.get_variable_scope().reuse):
                weight_decay = tf.multiply(
                    tf.nn.l2_loss(var), wd, name='weight_loss')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                     weight_decay)
            return var

    def _bias_variable(self, shape, constant=0.0):
            initializer = tf.constant_initializer(constant)
            return tf.get_variable(name='biases', shape=shape,
                                   initializer=initializer)

    def get_fc_weight_reshape(self, name, shape, num_classes=None):
            print('Layer name: %s' % name)
            print('Layer shape: %s' % shape)
            weights = self.data_dict[name][0]
            weights = weights.reshape(shape)
            if num_classes is not None:
                weights = self._summary_reshape(weights, shape,
                                                num_new=num_classes)
            init = tf.constant_initializer(value=weights,
                                           dtype=tf.float32)
            return tf.get_variable(name="weights", initializer=init, shape=shape)



