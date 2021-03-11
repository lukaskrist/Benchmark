import numpy as np
import benchmark_class
import ARS_benchmark
import matplotlib.pyplot as plt
N = 40
T = 0.03

noise = 0.0 # remember noise is optional

two_level = benchmark_class.TwoLevel(N, T, noise)

alist = 2*(np.random.rand(N)-0.5)

# this is how you evaluate it, with step-per-step updates

#for step in range(0, N):
    
#    _, r = two_level.update(alist[step])
#    print("r:", r)

# this is how you use roll_out

G = two_level.roll_out(alist)
#print("G:", G)
Tlist = []
# Note, if the noise is different from zero, you get a different result each time. 
pulse = np.random.rand(N)
alpha = 0.2
v = 0.5
maxite = 40
maxepochs = 30
f_story = np.zeros((maxepochs,maxite))
times = np.zeros((maxepochs,maxite))
for i in range(maxite):
    T += 0.1
    ARS = ARS_benchmark.ARSTrainer()
    f_story[:,i],times[:,i] = ARS.train(pulse,N,T,alpha,v,L=2, Noise = noise,maxepochs=maxepochs)
    Tlist.append(T)
    
time_average_A = np.average(times,axis = 1)
f_story_avg_A = np.average(f_story,axis = 1)
f_story_std_A = np.std(f_story,axis = 1)
f_story_max_A = np.max(f_story,axis = 1)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(Tlist,np.max(f_story,axis = 0),'*',label = "Fidelity max Waveless")
ax1.set_xlabel('T ')
ax1.set_ylabel('Fidelity')
ax1.set_title('QSL ARS graph-Max')
ax1.legend()
ax1.set_title('QSL ARS graph')
fig.show()