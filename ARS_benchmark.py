# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:58:00 2021

@author: loklu
"""


import numpy as np
import time
import benchmark_class


def MaxFunc(M):
    maxval = 1
    for i in range(len(M)):
        if M[i]>maxval:
            M[i] = maxval
        if M[i]< -maxval:
            M[i] = -maxval
    return M
def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)
class ARSTrainer():

    def __init__(self):
        #print("ARS-Wave initialised")
        #initialize global parameters
        self.N = 20
        
        
    def train(self,pulse,N,T,alpha,v,maxepochs = 20,data = None,A = np.zeros(1), Noise = None,L = None):
        """
        Implement Basic random search
        psi = our start configuration of the wave function, will be updated along the way
        psi_t = the target function (will start with going from 0 state to 1 state, but will )
        u0 = the starting control vector, which will be updated. 
        alpha = step-size 
        N = number of directions sampled per iteration 
        v = standard deviation of the exploration noise
        p = number of horizons
        maxepochs = the maximum amount of epochs that we go through before converging
        theta = the update vector for the u0
        if we have data we put that in the data, but will first be implemented later, for now that is just none
        """
        if L != None:
            sp = benchmark_class.SpinChain(N,T,L,Noise)
        else:
            sp = benchmark_class.TwoLevel(N,T,Noise)
        ### assertions
        #assert psi0.size == psi_target.size
        ### initialize
        epoch = 0
        p = 2*N
        AccHist = []
        #d = len(psi0)
        M = pulse #np.zeros((l,N))
        ### main loop
        t0 = time.time()
        times = []
#        if data != 0:
#            thing = data
#            thing[0] = 1
        #if version = Waveless - Spin chain
        if A.all() == 0:
            while epoch < maxepochs:
                epoch += 1

                samples = np.random.normal(size = (p,N))
                r_plus_list = []
                r_minus_list = []
                M_update = np.zeros((N))
                for i in range(p):
                    delta_plus = M+samples[i,:]*v
                    delta_plus = MaxFunc(delta_plus)
                    delta_minus = M-samples[i,:]*v
                    delta_minus = MaxFunc(delta_minus)
                    
                    r_plus = sp.roll_out(delta_plus)
                    r_minus = sp.roll_out(delta_minus)
                    #if r_plus > 0.15 or r_minus > 0.15:
                    r_plus_list.append(r_plus)
                    r_minus_list.append(r_minus)
                    M_update += alpha/N *(r_plus-r_minus )*samples[i,:]
                    #print(M_update)
                
                # update by choosing the standard deviation

                std = np.std([r_plus_list,r_minus_list])
                #print(r_plus_list,r_minus_list)
                if std != 0:
                    M_update /= std

                
                #if len(r_plus_list)>1:
                M += M_update
                M = MaxFunc(M)
                AccHist.append(np.max([r_plus_list,r_minus_list]))
                #AccHist.append(sp.roll_out(M))
                times.append(time.time()-t0)
                ### END CODE
        if A.any != 0:
            delta_A = np.zeros(N)
            #print(dis_var)
            delta_B = np.zeros(N)
            L = len(pulse)
            while epoch < maxepochs:
                epoch += 1

                samples = np.random.normal(size = (p,L,N))
                r_plus_list = []
                r_minus_list = []
                M_update = np.zeros((L,N))
                for i in range(p):
                    delta_plus = M+samples[i,:,:]*v
                    delta_minus = M-samples[i,:,:]*v
                    #delta_plus = MaxFunc(delta_plus)
                    #delta_minus = MaxFunc(delta_minus)
                    probs_plus = softmax(delta_plus)
                    probs_minus = softmax(delta_minus)
                    for k in range(N):    
                        delta_A[k] = np.random.choice(A,p=probs_plus[:,k])
                        delta_B[k] = np.random.choice(A,p=probs_minus[:,k])
                    r_plus = sp.roll_out(delta_A)
                    r_minus = sp.roll_out(delta_B)
                    r_plus_list.append(r_plus)
                    r_minus_list.append(r_minus)
                    M_update += alpha/N *(r_plus-r_minus )*samples[i,:,:]
                
                # update the 
                std = np.std([r_plus_list,r_minus_list])
                #print(std)
                if std != 0:
                    M_update /= std
                M += M_update
                AccHist.append(np.max([r_plus_list,r_minus_list]))
                times.append(time.time()-t0)        
        return M,   AccHist,times