import numpy as np
import benchmark_class

pulse = np.array([ 1.        ,  0.75186758,  0.18700439, -0.90246136, -1.        ,
       -1.        , -1.        , -1.        ,  0.37882948,  0.21337853,
        0.21337512,  0.37883106, -1.        , -1.        , -1.        ,
       -1.        , -0.90246192,  0.18700451,  0.75186756,  1.        ])


infidelity = 0.0005522221633704749


N = 20
T = 3 + 1/3 # QSL
L = 5


sp = benchmark_class.SpinChain(N, T, L)


F_test = sp.roll_out(pulse)



error = np.abs(1-F_test - infidelity)

print("error:", error)