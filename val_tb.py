import config as cfg
import threading
import time
import tensorflow as tf
import layer_maker_tradition
import numpy as np
import time
import input_data_force3
import cv2
class diguinet(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(diguinet, self).__init__(*args, **kwargs)
        cfg.BATCH_SIZE = 1
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

        with tf.name_scope('Input') as scope:
            images512_ = tf.placeholder('float', shape=[None, 512, 512, 3])
            images512_ = images512_ / 128.0 - 1.0

        with tf.name_scope('Info-Distributor') as scope:
            images256_ = lm.aver_pool_layer(images512_)
            images128_ = lm.aver_pool_layer(images256_)
            images64_ = lm.aver_pool_layer(images128_)
            images32_ = lm.aver_pool_layer(images64_)
            images16_ = lm.aver_pool_layer(images32_)

        with tf.name_scope('Neural_Network_Unit1') as scope:

            images4 = lm.pool_layer(lm.conv_layer(lm.pool_layer((lm.conv_layer(images16_, 3, 3, 32))), 3, 32, 32))
            chanel = 64
            tensor = lm.conv_layer(images4, 3, 32, chanel)
            tensor = lm.conv_layer(tensor, 3, chanel, chanel)
            tensor = lm.conv_layer(tensor, 3, chanel, chanel)
            tensor = lm.bilinear_upsample_layer(tensor, chanel, 8)

        with tf.name_scope('Neural_Network_Unit2') as scope:

            images8 = lm.pool_layer(lm.conv_layer(lm.pool_layer((lm.conv_layer(images32_, 3, 3, 32))), 3, 32, 32))
            tensor = tf.concat(3,[tensor, images8])
            chanel = 128
            tensor = lm.conv_layer(tensor, 3, 96, chanel)
            tensor = lm.conv_layer(tensor, 3, chanel, chanel)
            tensor = lm.conv_layer(tensor, 3, chanel, chanel)
            tensor = lm.bilinear_upsample_layer(tensor, chanel, 16)

        with tf.name_scope('Neural_Network_Unit3') as scope:

            images16 = lm.pool_layer(lm.conv_layer(lm.pool_layer((lm.conv_layer(images64_, 3, 3, 32))), 3, 32, 32))
            tensor = tf.concat([tensor, images16], axis=3)
            chanel = 256
            tensor = lm.conv_layer(tensor, 3, 160, chanel)
            tensor = lm.conv_layer(tensor, 3, chanel, chanel)
            tensor = lm.conv_layer(tensor, 3, chanel, chanel)
            tensor = lm.bilinear_upsample_layer(tensor, chanel, 32)

        with tf.name_scope('Neural_Network_Unit4') as scope:

            images32 = lm.pool_layer(lm.conv_layer(lm.pool_layer((lm.conv_layer(images128_, 3, 3, 32))), 3, 32, 32))
            tensor = tf.concat([tensor, images32], axis=3)
            chanel = 512
            tensor = lm.conv_layer(tensor, 3, 288, chanel)
            tensor = lm.conv_layer(tensor, 3, chanel, chanel)
            tensor = lm.conv_layer(tensor, 3, chanel, chanel)
            tensor = lm.bilinear_upsample_layer(tensor, chanel, 64)

        with tf.name_scope('Neural_Network_Unit5') as scope:

            images64 = lm.pool_layer(lm.conv_layer(lm.pool_layer((lm.conv_layer(images256_, 3, 3, 32))), 3, 32, 32))
            tensor = tf.concat([tensor, images64], axis=3)
            chanel = 512
            tensor = lm.conv_layer(tensor, 3, 544, chanel)
            tensor = lm.conv_layer(tensor, 3, chanel, chanel)
            tensor = lm.conv_layer(tensor, 3, chanel, chanel)
            tensor = lm.bilinear_upsample_layer(tensor, chanel, 128)

        with tf.name_scope('Neural_Network_Unit6') as scope:

            images128 = lm.pool_layer(lm.conv_layer(lm.pool_layer((lm.conv_layer(images512_, 3, 3, 32))), 3, 32, 32))
            tensor = tf.concat([tensor, images128], axis=3)
            chanel = 256
            tensor = lm.conv_layer(tensor, 3, 544, chanel)
            tensor = lm.conv_layer(tensor, 3, chanel, chanel)
            tensor1 = lm.conv_layer(tensor, 3, chanel, 2)

        with tf.name_scope('Output') as scope:

            labels = tf.placeholder('float', [None, 128, 128, 3])
            masks_ = tf.placeholder('float', [None, 128, 128])
            labels_elaster = (tf.log(labels[:, :, :, 0]) / tf.log(5.0))
            labels_poisson = tf.maximum(2 * labels[:, :, :, 1], 1.0 / 3 * labels[:, :, :, 1])
            loss_elaster = tf.square(labels_elaster - tensor1[:, :, :, 0])
            loss_poisson = tf.square(labels_poisson - tensor1[:, :, :, 1])
            loss_el = tf.reduce_sum(loss_elaster * masks_)
            loss_po = tf.reduce_sum(loss_poisson * masks_)



        sess = tf.Session()
        writer = tf.summary.FileWriter("logs/", sess.graph)
        saver = tf.train.Saver(max_to_keep=5)
        if cfg.IS_RESTORE:
            saver.restore(sess, './weight/lixue.cktp-' + str(cfg.SAVED_WEIGHT_STEPS))
            print('restore '+'./weight/lixue.cktp-' + str(cfg.SAVED_WEIGHT_STEPS))
        else:
            sess.run(tf.global_variables_initializer())

        print('Network has been build!')

        paper = np.zeros([100, 120, 3], dtype=np.uint8)
        paper[:, :, 1:3] = 255
        for j in range(120):
            paper[:, j, 0] = j
        paper = cv2.cvtColor(paper, cv2.COLOR_HSV2BGR)
        cv2.imshow('duisetiao', paper)
        for i in range(100):
            batch = ade20k.next_batch()
            lq_nt = batch[1][:, :, :, 2].reshape([128, 128])
            lq_nt[lq_nt == -1] = 0

            vp = np.ones([128, 128], np.uint8)
            vp = vp * 255
            sp = np.ones([128, 128], np.uint8)
            sp = sp * 255
            imgll = cv2.resize(batch[0].reshape([512, 512, 3]), (256, 256))
            cv2.imshow('img',imgll)
            result = sess.run(tensor1, feed_dict={images512_: batch[0]})
            sunshi = sess.run([loss_el, loss_po], feed_dict={images512_: batch[0], labels: batch[1][:, :, :, 0:3],
                                                             masks_: batch[1][:, :, :, 3]})
            loss_info = 'EPOCHS' + str(i) + ':' + str(sunshi)
            print(loss_info)

            elaster = result[:, :, :, 0].reshape([128, 128])
            poisson = np.minimum(1.0 / 2 * result[:, :, :, 1], 3 * result[:, :, :, 1]).reshape([128, 128])
            poisson = np.abs(poisson) * lq_nt

            '''
           print(elaster.max())
            t0 = np.zeros(cfg.IMG_SIZE, dtype=np.uint8)
            t1 = np.floor(elaster).astype(np.uint8)
            t2 = ((elaster * 100) % 100).astype(np.uint8)
            cv2.imshow('ei', np.dstack((t0, t1, t2))) 
          '''

            p5 = np.zeros([128, 128])
            p5 = 5
            elaster = np.power(p5, elaster)
            print(elaster.max())
            e_s = 120/22*(-1/22*(elaster-22)**2+22)
            e_s[e_s < 0] = 0
            e_s[e_s > 120] = 120
            print(type(vp), vp.shape, vp.dtype, vp.max(), vp.min())
            e_s = np.dstack((e_s.astype(np.uint8), sp, (vp * lq_nt).astype(np.uint8)))
            e_s = cv2.cvtColor(e_s, cv2.COLOR_HSV2BGR)
            e_s = cv2.resize(e_s,(256,256))
            esmix = cv2.addWeighted(e_s,0.8,imgll,0.2,0)
            cv2.imshow('ei',esmix)

            el = batch[1][:, :, :, 0].reshape([128, 128])
            # el = np.log(el)/np.log(p5)
            print(el.max())
            e_s = el * 120.0 / 22
            e_s[e_s < 0] = 0
            # e_s[e_s > 120] = 120
            print(type(vp), vp.shape, vp.dtype, vp.max(), vp.min())
            e_s = np.dstack((e_s.astype(np.uint8), sp, (vp * lq_nt).astype(np.uint8)))
            e_s = cv2.cvtColor(e_s, cv2.COLOR_HSV2BGR)
            cv2.imshow('el', e_s)

            maxv = poisson.max()
            cv2.imshow('pi', cv2.resize((poisson * 255 / maxv).astype(np.uint8),(255,255)))


            cv2.waitKey(0)


    def pause(self):
        self.__flag.clear()

    def resume(self):
        self.__flag.set()

    def save(self):
        self.__save.set()

    def stop(self):
        self.__flag.set()
        self.__running.clear()
