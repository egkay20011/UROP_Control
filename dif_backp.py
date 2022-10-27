def propagate(initial_conditions,force,weight_array):
    """
    result of this minimises cost function
    take negative to maximise it
    """
    def sigmoid(inputs):
        import numpy as np
        inputs = np.exp(-inputs)
        inputs = np.add(1,inputs)
        inputs = np.divide(1,inputs)
        return inputs
    import numpy as np
    import MLactuator
    #re-create activations
    result,second_layer,first_layer = MLactuator.MLM(initial_conditions,weight_array,False)

    
    cost_arr = np.array([[sigmoid(force),0]])#np.array([[sigmoid(force),0]])
    h = 0.00001
    
    
    #calculate final layer weight change
    grad_final_array = np.zeros(np.shape(weight_array[2]))
    fin_div_dev = 0
    for i in range(np.shape(weight_array[2])[1]):
        for y in range (np.shape(weight_array[2])[0]):
            z = weight_array[2][y][i]*second_layer[0][y]
            grad_final_array[y][i] = ((sigmoid(second_layer[0][y]*(weight_array[2][y][i]+h))-cost_arr[0][i])**2 - (sigmoid(z)-cost_arr[0][i])**2)/h
            
    #calculate second layer weight change
    grad_second_array = np.zeros(np.shape(weight_array[1]))
    for i in range(np.shape(weight_array[1])[1]): #for every neuron
        for y in range (np.shape(weight_array[1])[0]):
            prog_func = 0
            for ki in range(np.shape(weight_array[2])[1]):
                w_2_b = weight_array[1][i][y] * first_layer[0][i]
                w_3_c = weight_array[2][y][ki] * sigmoid(w_2_b)
                prog_func += ((sigmoid(sigmoid((weight_array[1][i][y]+h)* first_layer[0][i])*weight_array[2][y][ki])-cost_arr[0][ki])**2 - (sigmoid(sigmoid(w_2_b)*weight_array[2][y][ki])-cost_arr[0][ki])**2)/h
                
            grad_second_array[i][y] = prog_func
    
    #calculate first layer weight change
    grad_first_array = np.zeros(np.shape(weight_array[0]))
    for i in range(np.shape(weight_array[0])[0]): #for each neuron in first layer
        for y in range(np.shape(weight_array[0])[1]): #for each weight assigned to a first layer neuron
            prog_func = 0
            for x in range(np.shape(weight_array[1])[1]): # for each weight leaving the receptacle neuron
                for k in range(np.shape(weight_array[2])[1]):#for each final layer connection
                    w_1_a = weight_array[0][i][y] * initial_conditions[0][i]
                    w_2_b = weight_array[1][y][x] * sigmoid(w_1_a)
                    w_3_c = weight_array[2][x][k] * sigmoid(w_2_b)
                    prog_func += (((sigmoid(sigmoid(sigmoid((weight_array[0][i][y]+h) * initial_conditions[0][i])*weight_array[1][y][x])*weight_array[2][x][k])-cost_arr[0][k])**2)- ((sigmoid(sigmoid(sigmoid(w_1_a)*weight_array[1][y][x])*weight_array[2][x][k])-cost_arr[0][k])**2))/h
                    #prog_func = prog_func - ((sigmoid(sigmoid(sigmoid(w_1_a)*weight_array[1][y][x])*weight_array[2][x][k])-cost_arr[0][k])**2)/h
            grad_first_array[i][y] = prog_func
    return grad_first_array,grad_second_array,grad_final_array
