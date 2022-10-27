"""
This is the non linear control program that uses
nonlinear dynamics to kick the pendulum into a homoclinic orbit
from where it is then fully stablised with help of an LQR controller
"""
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
import numpy as np
import RK4solver
import matplotlib.pyplot as plt
import time
from control import lqr
import non_linear
start_time = time.time()


#initialisation
utility_array = np.array([1,1,9.81,0,1,1]) #
initial_conditions = np.array([[-7*(np.pi)/12,0,0,0]]) #0 doesnt work
initial_time = 0
target_time = 10
step_size = 0.001

#lists to keep track of things
ang_disp_arr = [initial_conditions[0][0]]
ang_vel_arr = [initial_conditions[0][1]]
lin_disp_arr = [initial_conditions[0][2]]
lin_vel_arr = [initial_conditions[0][3]]
tim_arr = [0]


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

Q_mat = np.array([[50,0,0,0],
         [0,10,0,0],
         [0,0,1,0],
         [0,0,0,10]])
R_mat = 0.001


impulse = 0
A_mat = np.array([[0,1,0,0],[L_m*gravity*M_T/D,damping*M_T/D,0,0],[0,0,0,1],[-L_m**2*gravity/D,-damping*L_m/D,0,0]])
B_mat = np.array([[0],[-L_m/D],[0],[I/D]])
gain = [20,10,10]
engage_time_array = []
force_array = []
while initial_time <target_time:
    
    #decide on controller

    #LQR control in linear part
    if 1 - np.cos(initial_conditions[0][0]) < 0.1:
        k_matrix,s,e = lqr(A_mat,B_mat,Q_mat,R_mat) # u =  - kx
        force = np.matmul(-k_matrix,np.transpose(initial_conditions))

    #energy control in non-linear part
    else:
        force = non_linear.energy_control(utility_array,initial_conditions,gain)
    
    force_array.append(force)
    initial_conditions,initial_time = RK4solver.RK4(initial_time,initial_conditions,step_size,utility_array,force)
    ang_disp_arr.append(initial_conditions[0][0])
    ang_vel_arr.append(initial_conditions[0][1])
    lin_disp_arr.append(initial_conditions[0][2])
    lin_vel_arr.append(initial_conditions[0][3])
    tim_arr.append(initial_time)
end_time = time.time()
print(end_time-start_time)
fig1 = plt.figure()
for i in range(1,len(ang_vel_arr)):
    if abs(ang_vel_arr[i])<10**-3 and abs(lin_vel_arr[i])<10**-3:
        print("time to stabilise is:")
        print(tim_arr[i])
        break
print("total impulse used is:")
print(impulse)
plt.ion()
plt.xlabel("Angular Displacement")
plt.ylabel("Angular Velocity")
plt.title("Test Case 1 RK4")
plt.plot(tim_arr,ang_disp_arr)
plt.plot(tim_arr,ang_vel_arr)
plt.plot(tim_arr,lin_disp_arr)
plt.plot(tim_arr,lin_vel_arr)
#plt.plot(tim_arr[1::],force_array)
#plt.plot(tim_arr,force_array)
#plt.plot(ang_disp_arr,ang_vel_arr)
plt.legend(["ang disp"," ang vel","lin disp","lin vel","total energy"])
plt.show()


