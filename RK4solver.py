import numpy as np
import pendulum

def RK4(initial_time,initial_conditions,step_size,utility_array,force):
    """
    this is the runge kutta order 4 solver used to advance the pendulum ODEs in time
    
    """

    k_1 = pendulum.pendulum(initial_time,initial_conditions,utility_array,force)

    k_2 = pendulum.pendulum(initial_time+step_size/2,initial_conditions+k_1*step_size/2,utility_array,force)
    k_3 = pendulum.pendulum(initial_time+step_size/2,initial_conditions+k_2*step_size/2,utility_array,force)
    k_4 = pendulum.pendulum(initial_time+step_size,initial_conditions+k_3*step_size/2,utility_array,force)
    T_4 = (1/6)*(k_1+2*k_2+2*k_3+k_4)

    initial_conditions = initial_conditions + T_4*step_size
    initial_time += step_size

    return initial_conditions,initial_time
