# -*- coding: utf-8 -*-
import numpy as np
import cv2
import xlrd
import config as cfg
import os
import random as rd


class ADE20K():
    txt_names = []
    def findfile(self,rootdir):
        for dir in os.listdir(rootdir):
            nrootdir = os.path.join(rootdir, dir)
            if os.path.isdir(nrootdir):
                self.findfile(nrootdir)
            else:
                if os.path.splitext(dir)[1] == '.txt':
                    self.txt_names.append(nrootdir)

    def __init__(self,rootdir,excel_file):
        self.findfile(rootdir)
        rd.shuffle(self.txt_names)

        self.force_data = xlrd.open_workbook(excel_file)
        self.force_table = self.force_data.sheets()[0]
        self.names = self.force_table.col_values(0)
        self.names_indexs_dict = {}
        self.force_list = []
        for i in range(0, self.force_table.nrows):
            self.force_list.append(self.force_table.row_values(i))

        for i in range(1, len(self.names)):
            self.names_indexs_dict[self.names[i]] = i + 1

        self.index = 0
        self.num_images = len(self.txt_names)


    def get_labelmask(self, txt_name):
        txt_file = open(txt_name)
        print(txt_name)
        lines = txt_file.readlines()
        lei_dict = {}
        lei_dict[' 0 '] = []
        lei_dict[' 1 '] = []
        lei_dict[' 2 '] = []

        for line in lines:
            try:
                line_s = line.split('#')
                lei_dict[line_s[1]].append(line_s[3])
            except:
                pass

        lei_dict[' 0 '] = list(set(lei_dict[' 0 ']))
        lei_dict[' 1 '] = list(set(lei_dict[' 1 ']))
        lei_dict[' 2 '] = list(set(lei_dict[' 2 ']))

        lei_dict_all = {}

        for i in lei_dict:
            tempdict = {}
            for name in lei_dict[i]:
                index = self.names_indexs_dict[name.strip()]
                tempdict[index] = self.force_list[index - 1]
            lei_dict[i] = tempdict
            lei_dict_all.update(tempdict)




        def hecheng(seg_index, part_index, mask, level):
            temp_maskhc_bl = np.zeros(seg_index.shape, dtype=np.int)
            temp_maskhc_xc = np.ones(seg_index.shape, dtype=np.int)
            level_lei_dict = lei_dict[level]
            for i in level_lei_dict:
                # i为类别id
                if level_lei_dict[i][10] == 0:
                    temp_maskhc_xc[part_index == i] = 0
                    temp_maskhc_bl[temp_maskhc_xc == 0] = 1
                    seg_index = seg_index * temp_maskhc_xc + part_index * temp_maskhc_bl
                else:
                    lei_dict_all.pop(i)
                    if level_lei_dict[i][8] == 1:
                         mask[part_index == i] = 0

            return seg_index, mask

        seg_png = cv2.resize(cv2.imread(txt_name[:-7] + 'seg.png', 1), cfg.LABEL_SIZE, interpolation=cv2.INTER_NEAREST)
        seg_index = seg_png[:, :, 2] / 10 * 256 + seg_png[:, :, 1]
        mask = np.ones(cfg.LABEL_SIZE, dtype=np.int)

        if lei_dict[' 2 '] != {}:
            part2_name = txt_name[:-7] + 'parts_2.png'
            part2_png = cv2.resize(cv2.imread(part2_name, 1), cfg.LABEL_SIZE)
            part2_index = part2_png[:, :, 2] / 10 * 256 + part2_png[:, :, 1]
            seg_index, mask = hecheng(seg_index, part2_index, mask, ' 2 ')

        if lei_dict[' 1 '] != {}:
            part1_name = txt_name[:-7] + 'parts_1.png'
            part1_png = cv2.resize(cv2.imread(part1_name, 1), cfg.LABEL_SIZE)
            part1_index = part1_png[:, :, 2] / 10 * 256 + part1_png[:, :, 1]
            seg_index, mask = hecheng(seg_index, part1_index, mask, ' 1 ')

        # 查看seg图是否正常


        paper_elaster = np.ones(cfg.LABEL_SIZE, dtype=np.float32)
        paper_parken = np.zeros(cfg.LABEL_SIZE, dtype=np.float32)
        paper_slg = np.ones(cfg.LABEL_SIZE, dtype=np.float32)
        # 固体1,流体-1

        for i in lei_dict_all:
            temp_b = seg_index == i
            if lei_dict_all[i][10] == 0:
                temp_bf = (temp_b).astype(np.float32)
                paper_elaster = paper_elaster + temp_bf * lei_dict_all[i][6]
                paper_parken = paper_parken + temp_bf * lei_dict_all[i][7]


            elif lei_dict_all[i][11] == 0:
                paper_slg[temp_b] = -1
        mask[paper_parken == 0] = 0
        mask[paper_slg == -1] = 0

        return np.dstack((paper_elaster, paper_parken, paper_slg, mask))


    def next_batch(self):
        if self.index+cfg.BATCH_SIZE>self.num_images:
            self.index = 0
            rd.shuffle(self.txt_names)
        imgs = []
        labels = []
        loop_times = cfg.BATCH_SIZE
        i = 0

        def handlee():
            try:
                txt_name = self.txt_names[rd.randint(0,self.num_images-1)]
                img = cv2.resize(cv2.imread(txt_name[:-8] + '.jpg'), cfg.IMG_SIZE)
                label_mask = self.get_labelmask(txt_name)
                imgs.append(img)
                labels.append(label_mask)
            except:
               handlee()

        while i < loop_times:
            try:
                txt_name = self.txt_names[self.index + i]
                img = cv2.resize(cv2.imread(txt_name[:-8] + '.jpg'), cfg.IMG_SIZE)
                label_mask = self.get_labelmask(txt_name)
                imgs.append(img)
                labels.append(label_mask)
                i = i + 1
            except:
                handlee()
                i=i+1

        batch_imgs = np.array(imgs)
        batch_labels = np.array(labels)
        self.index+=cfg.BATCH_SIZE
        return (batch_imgs,batch_labels)

if __name__ == '__main__':

    rootdir = 'F:\\ADE20K_2016_07_26\\validation'
    if cfg.TEST_MODE == 'on':
        print('TEST_MODE ON')
        rootdir = './ADETEST'
    ade20k = ADE20K(rootdir, './fcs.xlsx')
    cfg.BATCH_SIZE = 1
    for i in range(100):
        batch = ade20k.next_batch()
        el = batch[1][:, :, :, 0].reshape([256, 256])
        poo = batch[1][:, :, :, 1].reshape([256, 256])
        el = (el * 10).astype(np.uint8)
        poo = (poo * 500).astype(np.uint8)
        seg = np.dstack((np.ones([256, 256]), poo, el))
        cv2.imshow('seg', seg)
        key = cv2.waitKey(0)
        if key == 115:
            cv2.imwrite('seg1.jpg', seg)
            print('writedown')
        else:
            pass

