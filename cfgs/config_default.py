import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.img_w = 640
cfg.img_h = 480

cfg.img_ip = "127.0.0.1"

cfg.img_port = 9003
cfg.img_list_len = 10

cfg.seg_model_path = "model_files/seg"
cfg.det_model_path = "model_files/det"

cfg.conf_th = 0.25
cfg.xpd_ratio = 1.1

cfg.depth_mat = np.array([[578.500977, 0,          323.388000, 0],
                          [0,          578.500977, 252.487000, 0],
                          [0,          0,          1,          0],
                          [0,          0,          0,          1]])

cfg.color_mat = np.array([[518.468018, 0,          312.657990, 0],
                          [0,          518.468018, 239.076004, 0],
                          [0,          0,          1,          0],
                          [0,          0,          0,          1]])

cfg.d2c_mat = np.array([[ 0.999996, -0.002270,  0.001785, -25.179001],
                        [ 0.002274,  0.999995, -0.001989, -0.102628],
                        [-0.001780,  0.001993,  0.999996,  0.314967],
                        [0,          0,          0,          1]])

cfg.final_mat = cfg.color_mat.dot(cfg.d2c_mat).dot(np.linalg.inv(cfg.depth_mat))
