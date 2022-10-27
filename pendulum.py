def pendulum(initial_time,init_cons,util_arr,force):
    """
    conventions are as follows:
    initial conditions array goes in this order:
    -theta 0
    -omega 1
    -x 2
    -v 3

    utility array goes in this order:
    -rod mass 0
    -ball mass 1
    -gravity 2
    -damping 3
    -rod length 4
    -car mass 5
    """
    import numpy as np
    theta = init_cons[0][0]
    omega = init_cons[0][1]
    x = init_cons[0][2]
    v = init_cons[0][3]

    rod_mass = util_arr[0]
    ball_mass = util_arr[1]
    gravity = util_arr[2]
    damping = util_arr[3]
    rod_length = util_arr[4]
    car_mass = util_arr[5]
    R = ((util_arr[1]+util_arr[0]/2)/((util_arr[1]+util_arr[0])))*util_arr[4]
    M = rod_mass+ball_mass
    I = rod_length**2*(1/3*rod_mass+ball_mass)
    L_m = M*R
    M_T = car_mass+M
    
    next_array = np.array([0.1,0.1,0.1,0.1])#[0.1,0.1,0.1,0.1]) #initialise with 0.1 because 0 doesnt work for some reason¯\_(ツ)_/¯
    
    next_array[0] = omega
   
    next_array[1] = L_m*gravity*M_T*np.sin(theta)-damping*omega*M_T-L_m**2*omega**2*np.sin(theta)*np.cos(theta)-force*L_m*np.cos(theta)
    next_array[1] = next_array[1]/(I*M_T-L_m**2*np.cos(theta)**2)
    next_array[2] = v

    next_array[3] = damping*omega*L_m*np.cos(theta)+L_m**2*gravity*np.sin(theta)*np.cos(theta)-(I)*(L_m*np.sin(theta)*omega**2+force)
    next_array[3] = next_array[3]/(L_m**2*np.cos(theta)**2-I*M_T)
    return next_array

