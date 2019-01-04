import serial
from queue import Queue
import struct
import socket
import threading
import time

from cfgs.config import cfg
from robot_kinematics import *

def ang2rad(angle):
    return angle * np.pi / 180

class PointCloudServer:
    def __init__(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            simu_server_addr = (cfg.simu_ip, cfg.simu_pc_port)
            self.sock.connect(simu_server_addr)
            self.connected = True
        except ConnectionRefusedError:
            self.connected = False
            print('Connection to point cloud server is refused!')

        if self.connected:
            self.result_queue = Queue(10)
            send_th = threading.Thread(target=self.send_pc_bg)
            send_th.start()

    def send_pc_bg(self):
        while True:
            pc = self.result_queue.get()
            point_num = pc.shape[0]
            self.sock.sendall(struct.pack("=i", point_num))
            '''
            pc_xyz = pc[:, :3]
            pc_rgb = pc[:, 3:]
            pc_xyz_data = struct.pack("=%d%f" % (3 * point_num), *pc_xyz.flatten('F'))
            pc_rgb_data = struct.pack("=%d%i" % (3 * point_num), *pc_rgb.flatten('F'))
            # self.sock.sendall(pc_xyz_data)
            # self.sock.sendall(pc_rgb_data)
            '''
            point_data_list = []
            for point in pc:
                point_data = struct.pack("=3f3B", *(point[:3]), *(point[3:].astype(np.int)))
                point_data_list.append(point_data)
            pc_data = b"".join(point_data_list)
            self.sock.sendall(pc_data)


    def send_pc(self, pc):
        if self.connected:
            self.result_queue.put(pc)


class Robot:
    def __init__(self, sim):
        ''' initialize the connection with the robot arm
        '''
        self.sim = sim
        self.timeout = 5
        if not self.sim:
            self.arm_com = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)
            # send the teach command
            send_data = bytes([0x30, 0x10, 0x14])
            self.arm_com.write(send_data)

            # send the callback command
            send_data = bytes("G07 GCM=1\r\n",'ascii')
            self.arm_com.write(send_data)

            self.arrived = False
            self.stop_recv = False
            self.recv_thread = threading.Thread(target=self.receive_data)
            self.recv_thread.setDaemon(True)
            self.recv_thread.start()
        else:
            self.arm_com = None
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                simu_server_addr = (cfg.simu_ip, cfg.simu_robot_port)
                self.sock.connect(simu_server_addr)
                self.connected = True
            except ConnectionRefusedError:
                self.connected = False
                print('Connection to robot server is refused!')

    def receive_data(self):
        while not self.stop_recv:
            time.sleep(0.1)
            data = self.arm_com.read(1024)
            if data == b'%':
                self.arrived = True

    def go_ready_location(self):
        '''
        let the robot move to the ready location
        '''
        self.go_phi_location(cfg.ready_loc)

    def go_observe_location(self):
        '''
        let the robot move to the observation location
        return when the action is finished
        '''
        self.go_phi_location(cfg.obs_loc)

    def go_phi_location(self, phi_ary):
        '''
        let the robot move to the specific location defined by angles of the six joints

        variable names:
        * ``phi_ary``: six joints' angles, can be list or a numpy array

        return True when the action is finished, return False if the location is not available
        '''
        is_available = self.check_available(phi_ary)
        if not is_available:
            return False
        if self.sim:
            # send joint data to the simulator
            # six angles are encoded into 24 bytes (each one is a float number and encoded into 4 bytes)
            print("Go to location: ")
            print(' '.join([str(round(e, 2)) for e in phi_ary]))
            if self.connected:
                data = struct.pack("=6f", *np.array([ang2rad(e) for e in phi_ary]))
                self.sock.sendall(data)
                time.sleep(2)
        else:
            # send move command to the serial port
            phi_list = [str(round(e, 2)) for e in phi_ary]
            cmd_eles = ["J%d=%s" % (idx + 1, e) for idx, e in enumerate(phi_list)]
            cmd = "G00 " + " ".join(cmd_eles) + "\r\n"
            print(cmd)
            send_data = bytes(cmd, 'ascii')
            ret = self.arm_com.write(send_data)
            print(ret)
            time_passed = 0
            while (not self.arrived):
                time.sleep(0.1)
                time_passed += 0.1
                if time_passed > self.timeout:
                    break
            self.arrived = False

        return True

    def check_available(self, angles):
        '''
        check whether the given angels are achivable
        '''
        for idx, angle in enumerate(angles):
            if not cfg.ang_rng[idx][0] < angle < cfg.ang_rng[idx][1]:
                return False
        return True

    def phi_ary_diff(self, phi_ary_1, phi_ary_2):
        weights = np.array([1, 0.5, 0.25, 0.125, 0.0625, 0])
        diff_ary = np.abs(phi_ary_1 - np.array(phi_ary_2))
        diff = np.sum(weights * diff_ary)
        return diff


    def go_cts_locations(self, coord_list):
        '''
        let the robot move to a series of specific locations defined by the cartesian coordinate

        variable names:
        * ``coord_list``: the list of cartisian coordinate, including the xyz and the direction, e.g., [100, 100, 100, 0, 0, -1]

        return True when the action is finished, reutrn False if the location is not available
        '''
        sol_list_list = []
        for idx, coord in enumerate(coord_list):
            sol_list = inverse_kinematics(coord)
            sol_list = [e for e in sol_list if self.check_available(e)]
            if len(sol_list) == 0:
                print("Coord %d is not achievable!" % idx)
                return False
            sol_list_list.append(sol_list)

        phi_ary_list = []
        last_phi_ary = None
        for sol_list in sol_list_list:
            if last_phi_ary is None:
                phi_ary_list.append(sol_list[0])
                last_phi_ary = sol_list[0]
            else:
                # find the one in sol_list which is nearest to the last_phi_list
                diffs = [self.phi_ary_diff(last_phi_ary, e) for e in sol_list]
                min_idx = np.argmin(diffs)
                phi_ary_list.append(sol_list[min_idx])
                last_phi_ary = sol_list[min_idx]

        for phi_ary in phi_ary_list:
            self.go_phi_location(phi_ary)

    def go_cts_location(self, coord):
        '''
        let the robot move to the specific location defined by the cartesian coordinate

        variable names:
        * ``coord``: the cartisian coordinate, including the xyz and the direction, e.g., [100, 100, 100, 0, 0, -1]

        return True when the action is finished, reutrn False if the location is not available
        '''
        sol_list = inverse_kinematics(coord)
        phi_ary = None
        for sol in sol_list:
            if self.check_available(sol):
                phi_ary = sol
                break
        if phi_ary is None:
            print("Location unachievable!")
            return False
        return self.go_phi_location(phi_ary)

    def stop(self):
        if not self.sim:
            self.stop_recv = True
            self.recv_thread.join()
