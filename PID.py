def PID_controller(error_array,time_steps,gain_list):
    """
    inputs:
    error_array is a 1xn array, where n is the number of time steps that have
    occured so far in the control loop, this array holds the magnitude of the
    error wrt time

    time_steps is a scalar quantity that represents the size of each time step
    where control is applied, and where the motion is simulated

    gain_list is a 1x3 array of gains for each path of the PID controller:
    Proportional, Integral, Derivative, in that order

    outputs:
    control signal, number of newtons to apply

    
    
    """
    #determine proportion term
    traps = 0
    control_signal = 0

    control_signal += gain_list[0]*error_array[len(error_array)-1]

    #determine integral output
    #trapezium rule for numerical integration

    for i in range(1,len(error_array)):
        traps += time_steps*(error_array[i]+error_array[i-1])/2
    control_signal += gain_list[1]*traps

    #determine derivative output
    #slope between current error and previous error
    #rough and ready, more efficient ways of doing it

    control_signal += gain_list[2]*(error_array[len(error_array)-1] - error_array[len(error_array)-2])/time_steps

    return control_signal

