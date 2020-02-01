import config as cfg
import threading
import time
import tensorflow as tf
import layer_maker_tradition
import numpy as np
import time
import input_data_force
import cv2
class diguinet(threading.Thread):
    def __init__(self, *args, **kwargs):
        cfg.BATCH_SIZE = 1
        super(diguinet, self).__init__(*args, **kwargs)
        self.__flag = threading.Event()
        self.__flag.set()
        self.__running = threading.Event()
        self.__running.set()
        self.__save = threading.Event()

    def run(self):
        rootdir = '../../ADE20K_2016_07_26/images/validation/'
        if cfg.TEST_MODE == 'on':
            print('TEST_MODE ON')
            rootdir = './ADETEST'
        ade20k = input_data_force.ADE20K(rootdir, './fcs.xlsx')

        lm = layer_maker_tradition.layermaker(cfg.TRUNCATES_MEAN, cfg.TRUNCATES_STTDEV, cfg.BIAS, 'the_first')
        images128_ = tf.placeholder('float', shape=[None, 128, 128, 3])
        images128 = images128_ / 128.0 - 1.0
        labels = tf.placeholder('float', [None, 128, 128, 3])
        masks_ = tf.placeholder('float', [None, 128, 128])

        images64 = lm.aver_pool_layer(images128)
        images32 = lm.aver_pool_layer(images64)
        images16 = lm.aver_pool_layer(images32)
        images8 = lm.aver_pool_layer(images16)
        images4 = lm.aver_pool_layer(images8)

        chanel = 64
        tensor = lm.conv_layer(images4, 3, 3, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor = lm.bilinear_upsample_layer(tensor, chanel, 8)
        tensor = tf.concat([tensor, images8], axis=3)

        chanel = 128
        tensor = lm.conv_layer(tensor, 3, 67, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor = lm.bilinear_upsample_layer(tensor, chanel, 16)
        tensor = tf.concat([tensor, images16], axis=3)

        chanel = 256
        tensor = lm.conv_layer(tensor, 3, 131, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor = lm.bilinear_upsample_layer(tensor, chanel, 32)
        tensor = tf.concat([tensor, images32], axis=3)

        chanel = 512
        tensor = lm.conv_layer(tensor, 3, 259, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor = lm.bilinear_upsample_layer(tensor, chanel, 64)
        tensor = tf.concat([tensor, images64], axis=3)

        chanel = 512
        tensor = lm.conv_layer(tensor, 3, 515, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor = lm.bilinear_upsample_layer(tensor, chanel, 128)
        tensor = tf.concat([tensor, images128], axis=3)

        chanel = 256
        tensor = lm.conv_layer(tensor, 3, 515, chanel)
        tensor = lm.conv_layer(tensor, 3, chanel, chanel)
        tensor1 = lm.conv_layer(tensor, 3, chanel, 2)
        labels_elaster = (tf.log(labels[:, :, :, 0]) / tf.log(5.0))
        labels_poisson = tf.maximum(2 * labels[:, :, :, 1], 1.0 / 3 * labels[:, :, :, 1])
        loss_elaster = tf.square(labels_elaster - tensor1[:, :, :, 0])
        loss_poisson = tf.square(labels_poisson - tensor1[:, :, :, 1])
        # loss_soliedornot = tf.square(labels[:, :, :, 2] - soliedornot)
        # temp_loss = (loss_elaster + loss_poisson) * masks_ + loss_soliedornot
        loss_el = tf.reduce_sum(loss_elaster * masks_)
        loss_po = tf.reduce_sum(loss_poisson * masks_)

        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, './weight/lixue.cktp-' + str(cfg.SAVED_WEIGHT_STEPS))
        paper = np.zeros([100,120,3],dtype=np.uint8)
        paper[:,:,1:3] = 255
        for j in range(120):
            paper[:,j,0] = j
        paper = cv2.cvtColor(paper, cv2.COLOR_HSV2BGR)
        cv2.imshow('duisetiao',paper)
        for i in range(100):
            batch = ade20k.next_batch()
            lq_nt = batch[1][:, :, :, 2].reshape([128, 128])
            lq_nt[lq_nt == -1] = 0

            vp = np.ones([128,128],np.uint8)
            vp = vp*255
            sp = np.ones([128, 128],np.uint8)
            sp = sp * 255

            cv2.imshow('img',cv2.resize(batch[0].reshape([128,128,3]),(256,256)))
            result = sess.run(tensor1,feed_dict={images128_: batch[0]})
            sunshi = sess.run([loss_el, loss_po], feed_dict={images128_: batch[0], labels: batch[1][:, :, :, 0:3],
                                                             masks_: batch[1][:, :, :, 3]})
            loss_info = 'EPOCHS' + str(i) + ':' + str(sunshi)
            print(loss_info)

            elaster = result[:,:,:,0].reshape([128,128])
            poisson = np.minimum(1.0/2*result[:,:,:,1],3*result[:,:,:,1]).reshape([128,128])
            poisson = np.abs(poisson)*lq_nt

            '''
           print(elaster.max())
            t0 = np.zeros(cfg.IMG_SIZE, dtype=np.uint8)
            t1 = np.floor(elaster).astype(np.uint8)
            t2 = ((elaster * 100) % 100).astype(np.uint8)
            cv2.imshow('ei', np.dstack((t0, t1, t2))) 
          '''

            p5 = np.zeros([128,128])
            p5 = 5
            elaster = np.power(p5,elaster)
            print(elaster.max())
            e_s = elaster * 120.0 / 22
            e_s[e_s < 0] = 0
            e_s[e_s > 120] = 120
            print(type(vp),vp.shape,vp.dtype,vp.max(),vp.min())
            e_s = np.dstack((e_s.astype(np.uint8),sp,(vp*lq_nt).astype(np.uint8)))
            e_s = cv2.cvtColor(e_s, cv2.COLOR_HSV2BGR)
            cv2.imshow('ei', e_s)

            el = batch[1][:, :, :, 0].reshape([128, 128])
            #el = np.log(el)/np.log(p5)
            print(el.max())
            e_s = el * 120.0 / 22
            e_s[e_s < 0] = 0
            # e_s[e_s > 120] = 120
            print(type(vp), vp.shape, vp.dtype, vp.max(), vp.min())
            e_s = np.dstack((e_s.astype(np.uint8), sp, (vp * lq_nt).astype(np.uint8)))
            e_s = cv2.cvtColor(e_s, cv2.COLOR_HSV2BGR)
            cv2.imshow('el', e_s)

            maxv = poisson.max()
            cv2.imshow('pi',(poisson*255/maxv).astype(np.uint8))



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
