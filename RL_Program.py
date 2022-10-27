import matplotlib.pyplot as plt
import numpy as np
import NonLinPend
import tensorflow as tf
import MLactuator
import better_backp
import backp
import dif_backp

def logit(inputs):
    inputs = np.log(np.divide(inputs,np.add(1,-inputs)))
    return inputs

#wrapper function for python control practice
##initial conditions array goes in this order:
##    -theta 0
##    -omega 1
##    -x 2
##    -v 3
##    -a 4
##
##    utility array goes in this order:
##    -rod mass 0
##    -ball mass 1
##    -gravity 2
##    -damping 3
##    -rod length 4
##    -car mass 5

utility_array = np.array([1,1,9.81,1,1,1]) #
initial_conditions = np.array([[-7*(np.pi)/12,0,0,0]]) #0 doesnt work
initial_time = 0
target_time = 10
step_size = 0.001#0.001

#define ML model manually

FIRST_NEURONS = 32
SECOND_NEURONS = 32



#first layer

#issue with np.random.rand is that it is bounded by [0,1]
#it needs to be bounded between [-infinity,infinity]

#randomly generate weights first time
##first_layer_weights = np.random.rand(4,FIRST_NEURONS)
##first_layer_weights = logit(first_layer_weights)
##second_layer_weights = np.random.rand(FIRST_NEURONS,SECOND_NEURONS)
##second_layer_weights = logit(second_layer_weights)
###final layer
##final_layer_weights = np.random.rand(SECOND_NEURONS,2)
##final_layer_weights = logit(final_layer_weights)



first_layer_weights = np.loadtxt("weight_array1.txt")
second_layer_weights = np.loadtxt("weight_array2.txt")
final_layer_weights = np.loadtxt("weight_array3.txt")


weight_array = [first_layer_weights,second_layer_weights,final_layer_weights]


TRAINING_CYCLES = int(input("Enter number of training cycles"))
GAMES = int(input("Enter number of games per cycle"))

cycles = int(target_time/step_size)
for v in range(0,TRAINING_CYCLES): #100 training cycles
    print("beginning training cycle number,",v)
    avg_master = []
    dev_master = []
    force_master = []
    reward_master =[]
    activations_master = []
    first_master = np.zeros(np.shape(first_layer_weights))
    second_master = np.zeros(np.shape(second_layer_weights))
    third_master = np.zeros(np.shape(final_layer_weights))
    
    for u in range(0,GAMES): # of 100 games each
        print("beginning game",u)
        force_array,reward_function,activations = NonLinPend.SimPend(utility_array,initial_conditions,initial_time,target_time,step_size,weight_array)
        force_master.append(force_array)
        reward_master.append(reward_function)
        activations_master.append(activations)
    print("average reward this cycle is:",np.average(reward_master))
    avg = np.average(reward_master)
    std = np.std(reward_master)
    reward_master = np.add(reward_master,-avg)
    reward_master = np.divide(reward_master,std)
    for i in range(0,GAMES):
        for y in range (0,cycles):    
            a,b,c = better_backp.propagate(activations_master[i][y],force_master[i][y],weight_array)
            if reward_master[i] >= 0:
                first_master = np.add(first_master,a)
                second_master = np.add(second_master,b)
                third_master = np.add(third_master,c)
            elif reward_master[i] <= 0:
                first_master = np.subtract(first_master,a)
                second_master = np.subtract(second_master,b)
                third_master = np.subtract(third_master,c)
        print("game,",i,"out of,",GAMES,"has been analysed")
    first_master = np.divide(first_master,GAMES*cycles)
    second_master = np.divide(second_master,GAMES*cycles)
    third_master = np.divide(third_master,GAMES*cycles)
    weight_array[0] = np.add(weight_array[0],first_master)
    weight_array[1] = np.add(weight_array[1],second_master)
    weight_array[2] = np.add(weight_array[2],third_master)
    print("training cycle number,",v,"is complete, stats:")
    print("total sum of weight adjustments:,",np.sum(first_master)+np.sum(second_master)+np.sum(third_master))   
np.savetxt("weight_array1.txt",weight_array[0])
np.savetxt("weight_array2.txt",weight_array[1])
np.savetxt("weight_array3.txt",weight_array[2])
