from al.algorithms.coreset import *

if __name__ == '__main__':
    features = np.random.randn(1000, 10)
    dist = compute_distance_matrix(features)
    print(dist)
    print(dist.shape)

import numpy
from gurobipy import *
import pickle
import numpy.matlib
import time
import pickle
import bisect

def solve_fac_loc(xx,yy,subset,n,budget):
    model = Model("k-center")
    x={}
    y={}
    z={}
    for i in range(n):
        # z_i: is a loss
        z[i] = model.addVar(obj=1, ub=0.0, vtype="B", name="z_{}".format(i))
 
    m = len(xx)
    for i in range(m):
        _x = xx[i]
        _y = yy[i]
        # y_i = 1 means i is facility, 0 means it is not
        if _y not in y:
            if _y in subset: 
                y[_y] = model.addVar(obj=0, ub=1.0, lb=1.0, vtype="B", name="y_{}".format(_y))
            else:
                y[_y] = model.addVar(obj=0,vtype="B", name="y_{}".format(_y))
        #if not _x == _y:
        x[_x,_y] = model.addVar(obj=0, vtype="B", name="x_{},{}".format(_x,_y))
    model.update()

    coef = [1 for j in range(n)]
    var = [y[j] for j in range(n)]
    model.addConstr(LinExpr(coef,var), "=", rhs=budget+len(subset), name="k_center")

    for i in range(m):
        _x = xx[i]
        _y = yy[i]
        #if not _x == _y:
        model.addConstr(x[_x,_y], "<", y[_y], name="Strong_{},{}".format(_x,_y))

    yyy = {}
    for v in range(m):
        _x = xx[v]
        _y = yy[v]
        if _x not in yyy:
            yyy[_x]=[]
        if _y not in yyy[_x]:
            yyy[_x].append(_y)

    for _x in yyy:
        coef = []
        var = []
        for _y in yyy[_x]:
            #if not _x==_y:
            coef.append(1)
            var.append(x[_x,_y])
        coef.append(1)
        var.append(z[_x])
        model.addConstr(LinExpr(coef,var), "=", 1, name="Assign{}".format(_x))
    model.__data = x,y,z
    return model
