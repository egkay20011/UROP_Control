import numpy as np
import RK4solver
import matplotlib.pyplot as plt
import MLactuator
def SimPend(utility_array,initial_conditions,initial_time,target_time,step_size,weights):
    
    #lists to keep track of things
##    ang_disp_arr = [initial_conditions[0]]
##    ang_vel_arr = [initial_conditions[1]]
##    lin_disp_arr = [initial_conditions[2]]
##    lin_vel_arr = [initial_conditions[3]]
##    tim_arr = [0]
    def logit(inputs):
        inputs = np.log(np.divide(inputs,np.add(1,-inputs)))
        return inputs
    force_array = []
    reward_function = 0
    rod_mass = utility_array[0]
    ball_mass = utility_array[1]
    gravity = utility_array[2]
    damping = utility_array[3]
    rod_length = utility_array[4]
    car_mass = utility_array[5]
    R = ((utility_array[1]+utility_array[0]/2)/((utility_array[1]+utility_array[0])))*utility_array[4]
    M = rod_mass+ball_mass
    I = rod_length**2*(1/3*rod_mass+ball_mass)
    L_m = M*R
    M_T = car_mass+M
    D = I*M_T-L_m**2
    engage_time_array = []
    start_activations = []
    while initial_time <target_time:
        start_activations.append(initial_conditions)
        imed = MLactuator.MLM(initial_conditions,weights,True)
        x_bar = imed[0][0]
        sigma = imed[0][1]
        
        x_bar = logit(x_bar)
        sigma = 5*(sigma)*abs(x_bar)
        force = np.random.normal(x_bar,sigma)
        initial_conditions,initial_time = RK4solver.RK4(initial_time,initial_conditions,step_size,utility_array,force)
        force_array.append(force)
        
        #decide on rewards/penalties
        reward_function += (1-np.cos(initial_conditions[0][0]))*-10
        reward_function += abs(initial_conditions[0][1])*-5
        reward_function += abs(initial_conditions[0][2])*-2
        reward_function += abs(initial_conditions[0][3])*-2

        
        if np.sum(initial_conditions)>1000000:
            reward_function = -10000000000
            break
##        ang_disp_arr.append(initial_conditions[0])
##        ang_vel_arr.append(initial_conditions[1])
##        lin_disp_arr.append(initial_conditions[2])
##        lin_vel_arr.append(initial_conditions[3])
##        tim_arr.append(initial_time)

    return force_array,reward_function,start_activations
