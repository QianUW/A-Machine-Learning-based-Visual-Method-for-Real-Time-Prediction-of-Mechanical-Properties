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
        self.__flag = threading.Event()
        self.__flag.set()
        self.__running = threading.Event()
        self.__running.set()
        self.__save = threading.Event()

    def run(self):
        rootdir = 'F:\\ADE20K_2016_07_26\\validation'
        if cfg.TEST_MODE == 'on':
            print('TEST_MODE ON')
            rootdir = './ADETEST'
        ade20k = input_data_force3.ADE20K(rootdir, './fcs.xlsx')




        paper = np.zeros([100, 30, 3], dtype=np.uint8)
        paper[:, :, 1:3] = 255
        for j in range(30):
            paper[:, j, 0] = 120 / 22 * (-1 / 22 * (j - 22) ** 2 + 22)
        paper = cv2.cvtColor(paper, cv2.COLOR_HSV2BGR)
        cv2.imshow('duisetiao', paper)



        for i in range(1000):
            batch = ade20k.next_batch()

            lq_nt = batch[1][:, :, :, 2].reshape([256, 256])
            lq_nt[lq_nt == -1] = 0

            vp = np.ones([256, 256], np.uint8)
            vp = vp * 255
            sp = np.ones([256, 256], np.uint8)
            sp = sp * 255

            imgll = cv2.resize(batch[0].reshape([512, 512, 3]), (256, 256))
            cv2.imshow('img', imgll)
            #固体非固体
            lg = batch[1][:, :, :,2].reshape([256, 256])
            lg[lg == 1] = 255
            lg = lg.astype(np.uint8)
            cv2.imshow('lg', lg)
            #掩模矩阵
            mask = batch[1][:, :, :, 3].reshape([256,256])
            mask[mask == 1] = 255
            mask = mask.astype(np.uint8)
            cv2.imshow('mask',mask)
            # 弹性模量标签云图
            el = batch[1][:, :, :, 0].reshape([256, 256])
            elmask = batch[1][:, :, :, 3].reshape(256, 256) == 0
            # el = np.log(el)/np.log(p5)
            e_s = el * 120.0 / 25
            e_s[e_s < 0] = 0
            e_s[e_s > 120] = 120
            e_s = np.dstack((e_s.astype(np.uint8), sp, vp))
            e_s = cv2.cvtColor(e_s, cv2.COLOR_HSV2BGR)
            cv2.imshow('el', e_s)


            # 泊松比标签云图
            poo = batch[1][:, :, :, 1].reshape([256, 256])
            poissonl = poo.reshape([256, 256])
            poissonl[poissonl > 0.5] = 0.5
            poissonl[poissonl < -3] = -3
            hpaperpl = (np.maximum(poissonl * 15, poissonl * 180) + 90).astype(np.uint8)
            paperpl = np.dstack((hpaperpl, sp, vp))
            paperpl = cv2.cvtColor(paperpl, cv2.COLOR_HSV2BGR)
            cv2.imshow('pl', paperpl)
            key = cv2.waitKey(0)
            if  key == 115:
                cv2.imwrite('./'+str(i)+'.jpg',imgll)
                cv2.imwrite('./' + str(i) + 'lg.jpg', lg)
                cv2.imwrite('./' + str(i) + 'mask.jpg', mask)
                cv2.imwrite('./' + str(i) + 'el.jpg', e_s)
                cv2.imwrite('./' + str(i) + 'pl.jpg', paperpl)
                print('writedown')
            else:
                pass

    def pause(self):
        self.__flag.clear()

    def resume(self):
        self.__flag.set()

    def save(self):
        self.__save.set()

    def stop(self):
        self.__flag.set()
        self.__running.clear()
