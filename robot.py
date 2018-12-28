from cfgs.config import cfg
from robot_kinematics import *


class Robot:
    def __init__(self, sim):
        ''' initialize the connection with the robot arm
        '''
        self.sim = sim

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
        print("Go to location: ")
        print(' '.join([str(round(e, 2)) for e in phi_ary]))
        if self.sim == True:
            # move the robot
            pass
        else:
            # send move command to the simulator
            pass
        return True

    def check_available(self, angles):
        '''
        check whether the given angels are achivable
        '''
        for idx, angle in enumerate(angles):
            if not cfg.ang_rng[idx][0] < angle < cfg.ang_rng[idx][1]:
                return False
        return True
    
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
