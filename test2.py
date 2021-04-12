# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:12:32 2021

@author: loklu
"""

import numpy as np
from ARS_benchmark import ARSTrainer
import matplotlib.pyplot as plt
# Here is the code:
A = ARSTrainer()

#The QSL should be around
T = 0.33

Noise = 0
N = 20
v = 0.01
alpha = 0.3
L = 20
maxepochs = 20
max_ite = 40
# Make starting pulse u0

discrete_numbers = np.linspace(-1,1,num = L)
pulse = np.random.rand(L,N)
pulse_B = np.random.rand(N)

f_story_A = np.zeros((maxepochs,max_ite))
times_A = np.zeros((maxepochs,max_ite))
f_story_B = np.zeros((maxepochs,max_ite))
times_B = np.zeros((maxepochs,max_ite))
m = 0
T_list = []
for i in range(max_ite):
    T += 0.1
    M,f_story_A[:,i],times_A[:,i] = A.train(pulse,N,T,alpha,v,L=5,A = discrete_numbers,Noise = Noise)
    M,f_story_B[:,i],times_B[:,i] = A.train(pulse_B,N,T,alpha,v,L=5,Noise = Noise)
    pulse = np.random.rand(L,N)
    pulse_B = np.random.rand(N)
    T_list.append(T)
time_average_A = np.average(times_A,axis = 0)
f_story_avg_A = np.average(f_story_A,axis = 0)
f_story_std_A = np.std(f_story_A,axis = 0)
f_story_max_A = np.max(f_story_A,axis = 0)
fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax1.plot(time_average_A,f_story_max_A,'*',label = "Fidelity mean Wavey")
ax1.plot(T_list,f_story_max_A,'g*',label = "Fidelity discrete max Waveless")
ax1.plot(T_list,f_story_avg_A,'r*',label = "Fidelity discrete mean Waveless")
ax1.plot(T_list,np.max(f_story_B,axis = 0),'g+',label = "Fidelity max Waveless")
ax1.plot(T_list,np.average(f_story_B,axis = 0),'r+',label = "Fidelity mean Waveless")
ax1.set_xlabel('T')
ax1.set_ylabel('Fidelity')
ax1.set_title('QSL ARS discrete')
ax1.legend()
