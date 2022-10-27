import numpy as np
import MLactuator


def propagate(initial_conditions,force,weight_array):
    """
    result of this minimises cost function
    take negative to maximise it

    done in an unfortunate manner using tensors
    the dimensions are right
    im quite uncertain if the numbers are
    
    
    """
    def sigmoid(inputs):
        import numpy as np
        inputs = np.exp(-inputs)
        inputs = np.add(1,inputs)
        inputs = np.divide(1,inputs)
        return inputs

    def sigmoid_prime(inputs):
        inputs = np.multiply(sigmoid(inputs),(1-sigmoid(inputs)))
        return inputs
    

    
    #re-create activations
    result,second_layer,first_layer = MLactuator.MLM(initial_conditions,weight_array,False) 
    cost_arr = [sigmoid(force),0]
    h = 0.01
    initial_conditions = np.transpose(initial_conditions)
    result = np.transpose(result)
    second_layer = np.transpose(second_layer)
    first_layer = np.transpose(first_layer)

    
    #calculate final layer weight change
    #grad_final_array = np.zeros(np.shape(weight_array[2]))
    c_w_3 = np.multiply(weight_array[2],second_layer)
    grad_final_array = np.multiply(np.multiply(second_layer,sigmoid_prime(c_w_3)),np.multiply(2,np.subtract(sigmoid(c_w_3),cost_arr)))
    
    #calculate second layer weight change
    #grad_second_array = np.zeros(np.shape(weight_array[1]))
    b_w_2 = np.multiply(weight_array[1],first_layer)
    grad_second_array = np.multiply(np.divide(grad_final_array,second_layer),weight_array[2])
    tensor_create = np.multiply(first_layer,sigmoid_prime(b_w_2))
    tensor_create = np.array([tensor_create,tensor_create])#shape is 2,32,32
    grad_second_array = np.transpose(grad_second_array)
    grad_second_array = np.array([grad_second_array]) #creates 1,2,32, now need to transpose it to 2,32,1, somehow
    grad_second_array = np.reshape(grad_second_array,[2,32,1]) #ngl im not even sure if this is right, but it doesnt raise an error
    grad_second_array = np.multiply(grad_second_array,tensor_create) #final output in format 2,32,32
    
    #calculate first layer weight change
    b_a_1 = np.multiply(initial_conditions,weight_array[0])
    grad_first_array = np.multiply(np.divide(grad_second_array,first_layer),weight_array[1]) #creates 2,32,32
    tensor_create = np.multiply(initial_conditions,sigmoid_prime(b_a_1)) #creates 4,32, i think we need a 2,32,32,4 copy of this
    tensor_create = np.repeat(tensor_create[:, :, np.newaxis], 32, axis=2) #creates 4,32,32
    tensor_create = np.repeat(tensor_create[:, :,:, np.newaxis], 2, axis=3) #creates 4,32,32,2
    tensor_create = np.reshape(tensor_create,[2,32,32,4]) #creates 2,32,32,4
    
    grad_first_array = np.array([grad_first_array]) #creates 1,2,32,32
    grad_first_array = np.reshape(grad_first_array,[2,32,32,1]) #creates 2,32,32,1
    grad_first_array = np.multiply(grad_first_array,tensor_create) #final output in format 2,32,32,4               

    grad_second_array = np.average(grad_second_array,0)
    grad_first_array = np.average(grad_first_array,0)
    grad_first_array = np.transpose(np.average(grad_first_array,0))
    
    return grad_first_array,grad_second_array,grad_final_array
