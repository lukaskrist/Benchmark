import numpy as np
import scipy.linalg as la
import spin_chain


# Load data

# The data consists of ordered (x, 1-F(x)) pairs,
# where x denotes the pulses stored in pulses
# and 1-F the infidelity stored in infidelities
# The data was obtained by random seeding with GRAPE
# for each gradual update, so it contains both good and bad solutions.

Tlist = np.linspace(1.0, 4.0, 10)
idx = 7

T = Tlist[int(idx)]
name = "T"+str(T)
name = name.replace('.','_')
data = np.load(name + ".npz")



pulses = data["pulses"] 
infidelities = data["infidelities"] 

N = pulses.shape[1]
data_size = pulses.shape[0]

# Initialize system

Nqubits = 5

dt = T/N

SP = spin_chain.Spin_chain(Nqubits)

psi0 = np.zeros((SP.dim), dtype = complex)
psi0[0] = 1.0

psi_t = np.zeros((SP.dim), dtype = complex)
psi_t[-1] = 1.0

amax = 1.0
amin = -amax



H0 = -1.0*SP.ZZ

Hc = -1.0*np.array([SP.X])


# Test system


rand_int = np.random.randint(0, high = data_size)

pulse = pulses[rand_int]

psi = np.copy(psi0)

for step in range(0, N):
    H = H0 + pulse[step]*Hc[0]
    U = la.expm(-1j*H*dt)
    psi = np.dot(U, psi)
    
inf = 1 - np.abs(np.vdot(psi_t, psi))**2

print("calculated infidelity: ", inf)
print("stored infidelity: ", infidelities[rand_int])

print("error: ",inf - infidelities[rand_int])