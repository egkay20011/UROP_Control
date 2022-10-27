def energy_control(utility_array,initial_conditions,gain):
    import numpy as np
    
    
    rod_mass = utility_array[0]
    ball_mass = utility_array[1]
    gravity = utility_array[2]
    damping = utility_array[3]
    rod_length = utility_array[4]
    car_mass = utility_array[5]


    M = utility_array[0] + utility_array[1]
    M_T = car_mass+M
    R = (utility_array[0]/2 + utility_array[1])/M *rod_length
    gravity = utility_array[2]
    I = rod_length**2*(1/3*rod_mass+ball_mass)
    theta = initial_conditions[0][0]
    omega = initial_conditions[0][1]
    
    #rbKE = ((M)*0.5*initial_conditions[3]**2) #linear COM KE
    RKE = (0.5*I*initial_conditions[0][1]**2) #rotational KE
    GPE = [(M)*gravity*np.cos(initial_conditions[0][0])*R]
    
    energy_total = RKE+GPE

    #u_torque = -gain*initial_conditions[1]*(energy_total-M*gravity*R)
    desired_acceleration =(energy_total-M*gravity*R)*np.cos(theta)*omega*gain[0] -gain[1]*initial_conditions[0][2]-gain[2]*initial_conditions[0][3]
    #can add pd to control displacement and velocity

    term_1 = desired_acceleration*(M_T-(((M*R*np.cos(theta))**2)/I))
    term_2 = (M*R)**2*np.sin(theta)*np.cos(theta)*gravity/I
    term_3 = -omega*(damping*R*M*np.cos(theta)/I+R*M*omega*np.sin(theta))
    control_force = (term_1+term_2+term_3)


    return control_force
    
