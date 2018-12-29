import serial
import time

from cfgs.config import cfg
from robot_kinematics import *

class Robot:
    def __init__(self, sim):
        ''' initialize the connection with the robot arm
        '''
        self.sim = sim
        self.sleep_time = 5
        if not self.sim:
            self.arm_com = serial.Serial('/dev/ttyUSB0', 115200, timeout = 0.5)
        else:
            self.arm_com = None

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
            # move the robot
            print("Go to location: ")
            print(' '.join([str(round(e, 2)) for e in phi_ary]))
            pass
        else:
            # send move command to the simulator
            phi_list = [str(round(e, 2)) for e in phi_ary]
            cmd_eles = ["J%d=%s" % (idx + 1, e) for idx, e in enumerate(phi_list)]
            cmd = "G00 " + " ".join(cmd_eles) + "\r\n"
            print(cmd)
            send_data = bytes(cmd, 'ascii')
            ret = self.arm_com.write(send_data)
            print(ret)
            time.sleep(self.sleep_time)

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
        # import pdb
        # pdb.set_trace()
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
