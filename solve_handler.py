import numpy as np
from cvxpy import *

def predirection(A, L, d):    
    num_e = A.shape[1]


    q = Variable(num_e*2,1)
    AA = np.concatenate((A,-A), axis=1)
    LL = np.concatenate((L,L), axis=0)

    obj = Minimize( (LL.T*q**3)/3.0 )
    constraints = [AA*q <= d, q >= 0]
    
    prob = Problem(obj, constraints)
    prob.solve(verbose=True, solver=MOSEK)
    
    # Flip a according to the solution
    for i in range(num_e):
        if q.value[i] < q.value[i + num_e]:
            A[:,i] *= -1 # A is passed by reference

    return obj.value

def solve_imaginary_flow(A, L1, dh_max, d):
    num_e = A.shape[1]

    q = Variable(num_e,1)
    constraints = [A*q <= d, q>=0]

    obj = Minimize( L1.T*q**3-q.T*dh_max )
    prob = Problem(obj, constraints)
    prob.solve(verbose=True, solver=MOSEK)
    
    return obj.value, q.value

def solve_imaginary_pressure(A, L1, dh_max, hc, pump_head_list, qq):
    num_v = A.shape[0]
    h = Variable(num_v,1)
    L1=np.matrix(L1)
    constraints=[A.T*h-np.diag(L1.A1)*np.power(qq,2)+dh_max>=0,
                 h[pump_head_list]==hc[pump_head_list],
                 h >= hc]
    obj=Minimize(norm(A.T*h-np.diag(L1.A1)*np.power(qq,2)+dh_max))

    prob=Problem(obj,constraints)
    prob.solve(verbose=True, solver=MOSEK)

    return obj.value, h.value

def solve_max_flow(A, L1, dh_max, d, hh):
    num_e = A.shape[1]
    num_v=A.shape[0]
    L1=np.matrix(L1)

    sourceList = np.matrix([0.0]*num_v).T
    sourceList[[y for y, x in enumerate(d) if x>0]]=1

    temp = np.matrix([0.0]*num_e).T
    for i in range(num_e):
        temp[i] = (A.T*hh + dh_max)[i]/L1[i]
    qU = sqrt(temp)
        
    q = Variable(num_e,1)
    constraints = [A*q <= d,
                   q >= 0,
                   q <= qU,
                   A.T*hh-np.diag(L1.A1)*np.power(q,2)+dh_max>=0]
    obj=Minimize(-sourceList.T*(A*q))

    prob=Problem(obj,constraints)
    prob.solve(verbose=True, solver=MOSEK)

    return obj.value, q.value