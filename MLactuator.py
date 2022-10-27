import numpy as np
def MLM(inputs,weights,fast_flag):
    """
    this function is the body of the ML program, it converts input dynamics into a normal distribution describing the optimal force to apply
    no biases, has sigmoid function for memory
    """
    def sigmoid(inputs):
        import numpy as np
        inputs = np.exp(-inputs)
        inputs = np.add(1,inputs)
        inputs = np.divide(1,inputs)
        return inputs
    

    
    #I am unsure if it works, if program is bad check here first, make sure dims etc line up
    first_layer= np.zeros((np.shape(inputs)[1],np.shape(weights[0])[1]))
    first_layer = np.dot(inputs,weights[0])
    first_layer = sigmoid(first_layer)
    second_layer = np.dot(first_layer,weights[1])
    second_layer = sigmoid(second_layer)
    output_layer = np.dot(second_layer,weights[2])
    output_layer = sigmoid(output_layer)

    if fast_flag == True:
        return output_layer
    elif fast_flag == False:
        return output_layer,second_layer,first_layer 
