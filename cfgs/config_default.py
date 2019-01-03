import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.img_w = 640
cfg.img_h = 480

cfg.img_ip = "127.0.0.1"
cfg.img_port = 9003
cfg.img_list_len = 10

cfg.simu_ip = "127.0.0.1"
cfg.simu_robot_port = 9004
cfg.simu_pc_port = 9005

cfg.seg_model_path = "models/seg"
cfg.det_model_path = "models/det"

cfg.conf_th = 0.5
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

cfg.c2g_mat = np.array([[ 0.055371,  0.997846,  0.03518,  -68.883059],
                        [ 0.998188, -0.05449,  -0.025517, -28.982622],
                        [-0.023545,  0.036529, -0.999055, -18.865048],
                        [ 0.0,       0.0,       0.0,       1.0]])

cfg.final_mat = cfg.color_mat.dot(cfg.d2c_mat).dot(np.linalg.inv(cfg.depth_mat))

cfg.tool_len = 90
cfg.pick_dist = 50

cfg.ang_rng = [[-150, 150],
               [-90, 90],
               [-150, 150],
               [-180, 180],
               [-90, 90],
               [-180, 180]]

cfg.obs_loc = [0, 75, -145, 0, -30, 0]
cfg.ready_loc = [0, 15, -95, 0, -30, 0]

cfg.result_dir = "vis"
