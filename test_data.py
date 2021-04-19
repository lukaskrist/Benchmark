#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:21:25 2021

@author: lukas
"""
import numpy as np
#import benchmark_class
import ARS_benchmark
import ARS_Alt
import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
N = 20

noise = 0.00 # remember noise is optional

Tlist = []
# Note, if the noise is different from zero, you get a different result each time. 
pulse = np.random.rand(N)
alpha = 0.075
v = 0.15
alpha2 = 0.1
v2 = 0.4
maxite = 20
maxepochs = 20
f_story = np.zeros((maxepochs,maxite))
times = np.zeros((maxepochs,maxite))
M_list = np.zeros((100,maxite))
f_story2 = np.zeros((maxepochs,maxite))
times2 = np.zeros((maxepochs,maxite))
M_list2 = np.zeros((int(2000/20),maxite))
ARS = ARS_benchmark.ARSTrainer()
ALT = ARS_Alt.ARSTrainer()
T = 3+1/3

for i in range(maxite):
    pulse = np.random.rand(N)
    M_list[:,i],f_story[:,i],times[:,i] = ARS.train(pulse,N,T,alpha,v,L=5,data = np.ones(2), Noise = noise,maxepochs=maxepochs)
    #pulse = np.random.rand(N)
    #M_list2[:,i],f_story2[:,i],times2[:,i] = ARS.train(pulse,N,T,alpha2,v2,L=5, Noise = noise,maxepochs=maxepochs)
    pulse = np.random.rand(N)
    M_list2[:,i],f_story2[:,i],times2[:,i] = ALT.train(pulse,N,T,alpha,v,L=5,data = np.ones(2), Noise = noise,maxepochs=maxepochs)
    #pulse = np.random.rand(N)


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
ax1.plot(time_average_A,1-f_story_avg_A,'g*',label = "A Fidelity mean Waveless")
ax1.plot(time_average_A,1-f_story_max_A,'g+',label = "A fidelity max Waveless")
ax1.plot(time_average_B,1-f_story_avg_B,'b*',label = "B Fidelity mean Waveless")
ax1.plot(time_average_B,1-f_story_max_B,'b+',label = "B Fidelity max Waveless")
ax1.plot(time_average_A,1-(f_story_avg_A+f_story_std_A),'g--',label = "A Fidelity std Waveless")
ax1.plot(time_average_A,1-(f_story_avg_A-f_story_std_A),'g--')
ax1.plot(time_average_B,1-(f_story_avg_B+f_story_std_B),'b--',label = "B Fidelity std Waveless")
ax1.plot(time_average_B,1-(f_story_avg_B-f_story_std_B),'b--')
#ax1.plot(times,f_story,'*',label = "Fidelity max Waveless")
#ax1.plot(Tlist,f_story[-1,:],'*',label = "Fidelity max Waveless")
ax1.set_xlabel('Wall time ')
ax1.set_ylabel('Fidelity')
ax1.set_title('QSL ARS graph-Max')
ax1.set_yscale('log')
ax1.legend()
ax1.set_title('QSL ARS graph')
fig.show()
space = np.linspace(0,100,100)
time_average_A = np.average(times,axis = 1)
f_story_avg_A = np.average(M_list,axis = 1)
f_story_std_A = np.std(M_list,axis = 1)
f_story_max_A = np.max(M_list,axis = 1)
time_average_B = np.average(M_list2,axis = 1)
f_story_avg_B = np.average(M_list2,axis = 1)
f_story_std_B = np.std(M_list2,axis = 1)
f_story_max_B = np.max(M_list2,axis = 1)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(space,1-f_story_avg_A,'g*',label = "A Fidelity mean Waveless")
ax1.plot(space,1-f_story_max_A,'g+',label = "A fidelity max Waveless")
ax1.plot(space,1-f_story_avg_B,'b*',label = "B Fidelity mean Waveless")
ax1.plot(space,1-f_story_max_B,'b+',label = "B Fidelity max Waveless")
ax1.plot(space,1-(f_story_avg_A+f_story_std_A),'g--',label = "A Fidelity std Waveless")
ax1.plot(space,1-(f_story_avg_A-f_story_std_A),'g--')
ax1.plot(space,1-(f_story_avg_B+f_story_std_B),'b--',label = "B Fidelity std Waveless")
ax1.plot(space,1-(f_story_avg_B-f_story_std_B),'b--')
ax1.set_xlabel('Iteration ')
ax1.set_ylabel('Fidelity')
ax1.set_title('QSL ARS graph-Max')
ax1.set_yscale('log')
ax1.legend()
ax1.set_title('QSL ARS graph')
fig.show()