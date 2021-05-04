#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 12:19:03 2021

@author: lukas
"""
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

Tlist = np.linspace(1.0, 4.0, 10)
idx = 7

T = Tlist[int(idx)]
name = "T"+str(T)
name = name.replace('.','_')
data = np.load(name + ".npz")
pulses = data["pulses"] 
infidelities = data["infidelities"]

data2 = np.load("123.npz")
pulses2 = data2["pulses"]
infidelities2 = data2["infidelities"]

main = []
main2 = []
for i in range(1000):
    main.append(pulses[i])
for i in range(1000):
    main.append(pulses2[:,i])
X_tsne = TSNE().fit_transform(main)
#X_tsne2 = TSNE().fit_transform(main2)

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(X_tsne[1000:1999, 0], X_tsne[1000:1999, 1],label = "My data")
ax1.scatter(X_tsne[0:1000, 0], X_tsne[0:1000, 1],label = "Mogens Data")
ax1.set_title('T-SNE figure')
ax1.set_xlabel('X')
ax1.legend()
ax1.set_ylabel('Y')