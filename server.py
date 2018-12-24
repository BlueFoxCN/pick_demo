import socket
import pickle
import datetime
import argparse
import time
import struct
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorpack import *

from recv_img import *
from robot import *

from code_seg.train import Model as SegModel
from code_det.train import Model as DetModel
from cfgs.config import cfg

from code_det.cfgs.config import cfg as det_cfg
from code_seg.cfgs.config import cfg as seg_cfg

from code_det.predict import postprocess
from code_seg.utils import enlarge_box, filter_pc, rotate_pc

'''
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', help='In debug mode, user input is used instead of the remote robot app.')
args = parser.parse_args()
'''

# Load detection model and point cloud segmentation model
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

det_sess_init = SaverRestore(cfg.det_model_path)
det_model = DetModel()
det_predict_config = PredictConfig(session_init=det_sess_init,
                                   model=det_model,
                                   input_names=["input", "spec_mask"],
                                   output_names=["pred_x", "pred_y", "pred_w", "pred_h", "pred_conf", "pred_prob"])
det_predict_func = OfflinePredictor(det_predict_config)

seg_sess_init = SaverRestore(cfg.seg_model_path)
seg_model = SegModel() 
seg_predict_config = PredictConfig(session_init=seg_sess_init,
                                   model=seg_model,
                                   input_names=["input"],
                                   output_names=["pred"])
seg_predict_func = OfflinePredictor(seg_predict_config)


# Connect with depth camera
img_conn = construct_conn()
recv_img_thread = RecvImgThread(img_conn)
recv_img_thread.start()

# go back to the observe location
go_observe_location()

while True:
    msg = input('Continue(c) or Quit(q)?')

    if msg == 'c':
        # 1. obtain the depth and color image
        start_time = time.time()
        color_img, depth_img = recv_img_thread.get_img(sub_dir)
        done_recv_img = time.time()
        print('recv time: %g' % (done_recv_img - start_time))

        # 2. convert to point cloud (under depth camera frame) and colorize
        pc_xyz = []
        pc_rgb = []
        for i in range(cfg.img_h):
            for j in range(cfg.img_w):
                if depth_img[i, j] == 0:
                    continue
                z = depth_img[i, j]
                uv_depth_homo = np.zeros((4))
                uv_depth_homo[0] = j * z
                uv_depth_homo[1] = i * z;
                uv_depth_homo[2] = z;
                uv_depth_homo[3] = 1;

                uv_color_homo = cfg.final_mat.dot(uv_depth_homo);
                u_color = int(uv_color_homo[0] / uv_color_homo[2]);
                v_color = int(uv_color_homo[1] / uv_color_homo[2]);

                xyz_depth_homo = np.linalg.inv(cfg.depth_mat).dot(uv_depth_homo)

                x = xyz_depth_homo[0]
                y = xyz_depth_homo[1]

                pc_xyz.append(xyz_depth_homo[:3])
                pc_rgb.append(color_img[v_color, u_color])

        pc = np.hstack([np.array(pc_xyz), np.array(pc_rgb)])


        # 3. detect the targets from the color image
        input_img = cv2.resize(color_img, (det_cfg.img_w, det_cfg.img_h))
        input_img = np.expand_dims(input_img, axis=0)
        spec_mask = np.zeros((1, det_cfg.n_boxes, det_cfg.img_h // 32, det_cfg.img_w // 32), dtype=float) == 0
        predictions = det_predict_func(input_img, spec_mask)
        boxes = postprocess(predictions, image_shape=color_img.shape[:2], det_th=cfg.conf_th)

        # for each detected target
        picks = []  # each pick is described with a location and coordinate
        for box in boxes:
            # 3. get frustum pointset
            ext_box = enlarge_box(box, cfg.xpd_ratio, cfg.img_h, cfg.img_w)
            pc = filter_pc(point_cloud, ext_box)

            # 4. rotate the pointset
            ori_pc = np.copy(pc)
            pc = rotate_pc(pc, ext_box)

            # 5. segment the pointset
            predictions = seg_predict_func(pc)[0]
            seg_idxs = np.where(predictions[0])[0]
            seg_pc = ori_pc[0, seg_idxs]

            # 6. calculate the location and direction
            tgt_pt = np.mean(seg_pc[:, :3])
            direction = np.array([1, 0, 0])
            picks.append({"pt": tgt_pt,
                          "dir": direction})

        # for each calculated pick
        for pick in picks:
            # 7. execute the pick
            end_pt = pick['tgt_pt']
            start_pt = end_pt - cfg.pick_dist * pick['dir']
            go_cts_location(start_pt)
            go_cts_location(end_pt)

        # 8. go back to the observe location
        go_observe_location()



    elif msg == 'q':
        break

recv_img_thread.stop()
recv_img_thread.join()
