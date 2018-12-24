


def go_observe_location():
    pass

def go_phi_location(phi_ary)
    pass

def go_cts_location(coord):
    sol_list = inverse_kinematics(coord)
    phi_ary = sol_list[0]
    go_phi_location(phi_ary)
