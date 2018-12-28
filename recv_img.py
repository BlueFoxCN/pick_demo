import struct
import cv2
import datetime
import copy
import sys
import numpy as np
from scipy import misc
import time
import socket
import pickle
import os
from threading import Thread

from cfgs.config import cfg

def construct_conn():

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    addr = (cfg.img_ip, cfg.img_port)
    sock.bind(addr)
    sock.listen(1)
    print('Waiting for image connection...')
    conn, addr= sock.accept()
    print('image connected with %s:%s' % addr)

    return conn

class RecvImgThread(Thread):

    def __init__(self, conn, show_img=False):
        Thread.__init__(self)
        self.conn = conn
        self.depth_imgs = [None] * cfg.img_list_len
        self.color_imgs = [None] * cfg.img_list_len
        self.recv = True
        self.show_img = show_img

    def run(self):
        depth_img_size = cfg.img_h * cfg.img_w
        cur_frame = [None] * depth_img_size
        line_num = 1

        while self.recv:

            # receive the indicator (depth img or color img)
            data = self.conn.recv(4)
            ind = struct.unpack('i', data)[0]

            cur_idx = 0
            if ind == 0:
                # print('receive a depth image')
                # then receive a depth image
                while cur_idx < depth_img_size:
                    recv_len = min(cfg.img_w * line_num * 2, (depth_img_size - cur_idx) * 2)
                    data = self.conn.recv(recv_len)
                    value = struct.unpack('<%dH' % (len(data) // 2), data)
                    cur_frame[cur_idx:cur_idx + len(value)] = value
                    cur_idx += len(value)
                cur_depth_img = np.array(cur_frame).reshape(cfg.img_h, cfg.img_w)

                # insert data to the head of the list
                for i in range(cfg.img_list_len - 1, 0, -1):
                    self.depth_imgs[i] = self.depth_imgs[i-1]
                self.depth_imgs[0] = cur_depth_img

                if self.show_img == True:
                    cv2.imshow('depth_image', cur_depth_img)
                    cv2.waitKey(1)
            else:
                # print('receive a color image')
                # then receive a color image
                data = self.conn.recv(4)
                color_img_size = struct.unpack('i', data)[0]
                buf = b''
                while cur_idx < color_img_size:
                    recv_len = min(1000, (color_img_size - cur_idx))
                    data = self.conn.recv(recv_len)
                    buf = buf + data
                    cur_idx += len(data)
                nparr = np.fromstring(buf, np.uint8)
                cur_color_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                cur_color_img = cv2.cvtColor(cur_color_img, cv2.COLOR_BGR2RGB)

                # insert data to the head of the list
                for i in range(cfg.img_list_len - 1, 0, -1):
                    self.color_imgs[i] = self.color_imgs[i-1]
                self.color_imgs[0] = cur_color_img

                if self.show_img == True:
                    cv2.imshow('color_image', cur_color_img)
                    cv2.waitKey(1)



    def stop(self):
        self.recv = False

    def get_img(self, dir_path):
        imgs_buffer = copy.deepcopy(self.depth_imgs)

        imgs_buffer = np.array(imgs_buffer)
        # img_process = np.zeros((cfg.img_h, cfg.img_w))

        imgs_sum = np.sum(imgs_buffer, 0)
        eff_pt_num = np.sum((imgs_buffer != 0).astype(np.int), 0)
        pt_num = np.maximum(np.ones_like(eff_pt_num), eff_pt_num)
        img_process = imgs_sum / pt_num

        done_avg = time.time()

        c = 0
        while((img_process==0).any() and c < 2):
            c += 1
            pad_arg = np.argwhere(img_process==0)
            for i, j in pad_arg:
                grid = img_process[i-1:i+2, j-1:j+2]
                num = np.argwhere(grid!=0).shape[0]
                if num:
                    img_process[i, j] = int(np.sum(grid) / num)

        depth_img_save_path = os.path.join(dir_path, 'depth.jpg')
        misc.imsave(depth_img_save_path, img_process)
        color_img_save_path = os.path.join(dir_path, 'color.jpg')
        misc.imsave(color_img_save_path, self.color_imgs[0])
        f = open(os.path.join(dir_path, 'depth.pkl'), 'wb')
        pickle.dump(img_process, f)
        print('Done process imgs')
        return [self.color_imgs[0], img_process]


if __name__ == "__main__":
    conn = construct_conn()
    recv_img_thread = RecvImgThread(conn, show_img=True)
    recv_img_thread.run()
