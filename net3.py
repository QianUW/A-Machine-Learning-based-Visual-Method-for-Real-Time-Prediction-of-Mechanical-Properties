import config as cfg
import threading
import time
import tensorflow as tf
import layer_maker_tradition
import numpy as np
import time
import input_data_force3
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
