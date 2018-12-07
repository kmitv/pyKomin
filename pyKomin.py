import numpy as np
from numpy.linalg import inv

x = np.array([10, 0.1, 0.2, 0.7, 12, 0.1, 0.15, 0.75])

for n in range(10):
    W=np.array([[x[0]*(x[1]+x[2])-x[4]*(x[5]+x[6])],
                [x[0]*(x[1]+2*x[2])-x[4]*(x[5]+2*x[6])],
                [x[0]*x[3]-x[4]*x[7]],
                [x[1]+x[2]+x[3]-1],
                [x[5]+x[6]+x[7]-1]])
      
    A=np.array([[x[1]+x[2],x[0], x[0], 0, -x[5]-x[6], -x[4], -x[4], 0],
                [x[1]+2*x[2], x[0], 2*x[0], 0, -x[5]-2*x[6], -x[4], -2*x[4], 0],
                [x[3], 0, 0, x[0], -x[7], 0, 0, -x[4]],
                [0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1]])
    
    M=np.diag((0.1*x)**2)

    V=M.dot(A.T).dot(inv(A.dot(M).dot(A.T)).dot(W))

    M=M-M.dot(A.T).dot(inv(A.dot(M).dot(A.T))).dot(A).dot(M)
      
    x=x-V.T

    x=x[0]

print('x=', x)

print('W=',W)