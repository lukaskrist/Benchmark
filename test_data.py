#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:21:25 2021

@author: lukas
"""
import numpy as np
#import benchmark_class
import ARS_benchmark
import matplotlib.pyplot as plt
N = 20

noise = 0.00 # remember noise is optional

Tlist = []
# Note, if the noise is different from zero, you get a different result each time. 
pulse = np.random.rand(N)
alpha = 0.1
v = 0.1
maxite = 10
maxepochs = 30
f_story = np.zeros((maxepochs,maxite))
times = np.zeros((maxepochs,maxite))
M_list = np.zeros((N,maxite))
f_story2 = np.zeros((maxepochs,maxite))
times2 = np.zeros((maxepochs,maxite))
M_list2 = np.zeros((N,maxite))
ARS = ARS_benchmark.ARSTrainer()

T = 3+1/3

for i in range(maxite):
    M_list[:,i],f_story[:,i],times[:,i] = ARS.train(pulse,N,T,alpha,v,L=5,data = np.ones(2), Noise = noise,maxepochs=maxepochs)
    pulse = np.random.rand(N)
    M_list2[:,i],f_story2[:,i],times2[:,i] = ARS.train(pulse,N,T,alpha,v,L=5, Noise = noise,maxepochs=maxepochs)


time_average_A = np.average(times,axis = 1)
f_story_avg_A = np.average(f_story,axis = 1)
f_story_std_A = np.std(f_story,axis = 1)
f_story_max_A = np.max(f_story,axis = 1)
time_average_B = np.average(times2,axis = 1)
f_story_avg_B = np.average(f_story2,axis = 1)
f_story_std_B = np.std(f_story2,axis = 1)
f_story_max_B = np.max(f_story2,axis = 1)
fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax1.plot(time_average_A,1-f_story_avg_A,'*',label = "Fidelity mean Waveless")
ax1.plot(time_average_A,1-f_story_max_A,'+',label = "Data fidelity max Waveless")
ax1.plot(time_average_B,1-f_story_max_B,'+',label = "Fidelity max Waveless")
#ax1.plot(time_average_A,1-(f_story_avg_A+f_story_std_A),'g--',label = "Fidelity std Waveless")
#ax1.plot(time_average_A,1-(f_story_avg_A-f_story_std_A),'r--',label = "Fidelity std Waveless")
#ax1.plot(times,f_story,'*',label = "Fidelity max Waveless")
#ax1.plot(Tlist,f_story[-1,:],'*',label = "Fidelity max Waveless")
ax1.set_xlabel('Wall time ')
ax1.set_ylabel('Fidelity')
ax1.set_title('QSL ARS graph-Max')
ax1.set_yscale('log')
ax1.legend()
ax1.set_title('QSL ARS graph')
fig.show()
