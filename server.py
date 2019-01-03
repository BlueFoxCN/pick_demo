import datetime
import argparse
import time
import os
import numpy as np
import cv2

from tensorpack import *

from code_seg.train import Model as SegModel
from code_det.train import Model as DetModel
from cfgs.config import cfg

from code_det.cfgs.config import cfg as det_cfg
from code_seg.cfgs.config import cfg as seg_cfg

from code_det.predict import postprocess
from code_seg.utils import enlarge_box, filter_pc, rotate_pc, save_ply_file

from recv_img import *
from robot_control import *

parser = argparse.ArgumentParser()
parser.add_argument('--sim', action='store_true', help='In simulation mode, the robot will not move.')
parser.add_argument('--debug', action='store_true', help='In debug mode, images are read from files.')
parser.add_argument('--vis', action='store_true', help='When set, save visulization results.')
parser.add_argument('--img_path', help='debug image path')
parser.add_argument('--depth_path', help='debug depth file path')
args = parser.parse_args()

if os.path.isdir(cfg.result_dir) == False:
    os.mkdir(cfg.result_dir)

r = Robot(args.sim)
if args.sim:
    pc_server = PointCloudServer()

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
if args.debug == True:
    color_img = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)
    f = open(args.depth_path, 'rb')
    depth_img = pickle.load(f)
else:
    img_conn = construct_conn()
    recv_img_thread = RecvImgThread(img_conn)
    recv_img_thread.start()

# go back to the observe location
r.go_observe_location()

while True:
    msg = input('Continue(c) or Quit(q)?')

    if msg == 'c':
        # 0. generate result saving dir
        vis_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        vis_dir_path = os.path.join(cfg.result_dir, vis_dir)
        os.mkdir(vis_dir_path)

        # 1. obtain the depth and color image
        start_time = time.time()
        if args.debug == False:
            color_img, depth_img = recv_img_thread.get_img(vis_dir_path)
        done_recv_img = time.time()
        print('recv time: %g' % (done_recv_img - start_time))

        # 2. convert to point cloud (under depth camera frame) and colorize
        start_time = time.time()

        widths = np.array(range(cfg.img_w))
        widths = np.tile(widths, cfg.img_h)
        heights = np.array(range(cfg.img_h))
        heights = np.tile(np.expand_dims(heights, 1), cfg.img_w).reshape(-1)
        ones = np.ones((cfg.img_w * cfg.img_h))
        uvs_depth = np.vstack([widths, heights, ones])

        zs = depth_img.reshape(-1)
        uvs_depth = uvs_depth * zs
        uvs_depth = uvs_depth[:, np.any(uvs_depth > 0, axis=0)]
        uvs_depth_homo = np.vstack([uvs_depth, np.ones(uvs_depth.shape[1])])
        uvs_color_homo = cfg.final_mat.dot(uvs_depth_homo)

        us_color = (uvs_color_homo[0] / uvs_color_homo[2]).astype(np.int)
        vs_color = (uvs_color_homo[1] / uvs_color_homo[2]).astype(np.int)

        depth_mat_inv = np.linalg.inv(cfg.depth_mat)
        xyzs_depth_homo = depth_mat_inv.dot(uvs_depth_homo)
        pc_xyz = xyzs_depth_homo[:3].transpose()

        pc_rgb = color_img[vs_color, us_color]
        pc = np.hstack([pc_xyz, pc_rgb])

        done_cvt_pc = time.time()
        print('convert point cloud time: %g' % (done_cvt_pc - start_time))

        if args.vis == True:
            start_time = time.time()
            save_ply_file(pc, os.path.join(vis_dir_path, 'pc.ply'))
            done_save_pc = time.time()
            print('point cloud save time: %g' % (done_save_pc - start_time))

        if args.sim == True:
            pc_server.send_pc(pc)

        # 3. detect the targets from the color image
        start_time = time.time()
        input_img = cv2.resize(color_img, (det_cfg.img_w, det_cfg.img_h))
        input_img = np.expand_dims(input_img, axis=0)
        spec_mask = np.zeros((1, det_cfg.n_boxes, det_cfg.img_h // 32, det_cfg.img_w // 32), dtype=float) == 0
        predictions = det_predict_func(input_img, spec_mask)
        boxes = postprocess(predictions, image_shape=color_img.shape[:2], det_th=cfg.conf_th)
        boxes = boxes['taozi'] if 'taozi' in boxes.keys() else []

        for idx, box in enumerate(boxes):
            [conf, xmin, ymin, xmax, ymax] = box
            cv2.rectangle(color_img,
                          (int(xmin), int(ymin)),
                          (int(xmax), int(ymax)),
                          (255, 0, 0),
                          2)
            cv2.putText(color_img,
                        str(round(conf, 2)),
                        (int(xmin + 3), int(ymax) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0))
            cv2.putText(color_img,
                        str(idx),
                        (int(xmin + 3), int(ymax) - 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0))

        misc.imsave(os.path.join(vis_dir_path, 'det.jpg'), color_img)
        done_det = time.time()
        print('detection time: %g' % (done_det - start_time))

        print('%d boxes' % len(boxes))
        # for each detected target
        picks = []  # each pick is described with a location and coordinate

        # 3. get frustum pointset
        start_time = time.time()
        ext_boxes = [enlarge_box(e[1:], cfg.xpd_ratio, cfg.img_h, cfg.img_w) for e in boxes]
        filtered_idxes_list = filter_pc(pc, ext_boxes)
        done_filter_pc = time.time()
        print('filter pc time: %g' % (done_filter_pc - start_time))

        for idx, ext_box in enumerate(ext_boxes):

            # 4. rotate the pointset
            start_time = time.time()
            frustum_pc = pc[filtered_idxes_list[idx]]
            ori_frustum_pc = np.copy(frustum_pc)
            frustum_pc = rotate_pc(frustum_pc, ext_box)
            done_rotate_pc = time.time()
            print('rotate pc time: %g' % (done_rotate_pc - start_time))

            # 5. segment the pointset
            start_time = time.time()
            save_ply_file(ori_frustum_pc, os.path.join(vis_dir_path, '%d.ply' % idx))
            frustum_pc = np.expand_dims(frustum_pc, 0)
            predictions = seg_predict_func(frustum_pc)[0]
            seg_idxs = np.where(predictions[0])[0]
            seg_frustum_pc = ori_frustum_pc[seg_idxs]
            seg_idxs_ori = np.array(filtered_idxes_list[idx])[seg_idxs]
            save_ply_file(seg_frustum_pc, os.path.join(vis_dir_path, '%d_seg.ply' % idx))
            done_segment_pc = time.time()
            print('segment pc time: %g' % (done_segment_pc - start_time))

            # 6. transform to the robot frame
            start_time = time.time()
            g2b_list = g2b(cfg.obs_loc)
            g2b_mat = np.identity(4)
            for t in g2b_list:
                g2b_mat = np.matmul(t, g2b_mat)
            c2b_mat = g2b_mat.dot(cfg.c2g_mat)
            for point in seg_frustum_pc:
                point_homo = np.hstack([point[:3], 1])
                point_base = c2b_mat.dot(point_homo)[:3]
                point[:3] = point_base
            done_transform_pc = time.time()
            print('transform pc time: %g' % (done_transform_pc - start_time))

            # 7. calculate the location and direction
            tgt_pt = np.mean(seg_frustum_pc[:, :3], 0)
            direction = np.array([1, 0, -1])
            direction = direction / np.linalg.norm(direction)
            picks.append({"pt": tgt_pt,
                          "dir": direction,
                          "seg_idxs": seg_idxs_ori})

        # 8. update the point cloud if in simulation mode
        if args.sim or args.vis:
            pc_copy = np.copy(pc)
            for pick in picks:
                num = len(pick['seg_idxs'])
                pc_copy[pick['seg_idxs'], 3:] = np.hstack([255 * np.ones((num, 1)),np.zeros((num, 2))])
            if args.sim:
                pc_server.send_pc(pc_copy)
            if args.vis == True:
                save_ply_file(pc_copy, os.path.join(vis_dir_path, 'pc_seg.ply'))


        # 9. execute each calculated pick
        for pick in picks:
            end_pt = pick['pt'] - cfg.tool_len * pick['dir']
            start_pt = end_pt - cfg.pick_dist * pick['dir']
            start_coord = np.hstack([start_pt, pick['dir']])
            end_coord = np.hstack([end_pt, pick['dir']])
            r.go_cts_locations([start_coord, end_coord])
            r.go_ready_location()

        # 10. go back to the observe location
        r.go_observe_location()

    elif msg == 'q':
        break

if not args.debug:
    recv_img_thread.stop()
    recv_img_thread.join()

r.stop()
