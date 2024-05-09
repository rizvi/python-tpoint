import numpy as np
import pandas as pd

def pretreat(X, method, para1=None, para2=None):
    # data pretreatment
    # HD Li, Central South University
    if para1 is None and para2 is None:
        Mx, Nx = X.shape
        if method == 'autoscaling':
            para1 = np.mean(X, axis=0)
            para2 = np.std(X, axis=0)
        elif method == 'center':
            para1 = np.mean(X, axis=0)
            para2 = np.ones(Nx)
        elif method == 'unilength':
            para1 = np.mean(X, axis=0)
            para2 = np.zeros(Nx)
            for j in range(Nx):
                para2[j] = np.linalg.norm(X[:, j] - para1[j])
        elif method == 'minmax':
            para1 = np.min(X, axis=0)
            maxv = np.max(X, axis=0)
            para2 = maxv - para1
        elif method == 'pareto':
            para1 = np.mean(X, axis=0)
            para2 = np.sqrt(np.std(X, axis=0))
        else:
            print('Wrong data pretreat method!')
            return None
        for i in range(Nx):
            X[:, i] = (X[:, i] - para1[i]) / para2[i]
        return X, para1, para2
    elif para1 is not None and para2 is not None:
        Mx, Nx = X.shape
        for i in range(Nx):
            X[:, i] = (X[:, i] - para1[i]) / para2[i]
        return X, para1, para2
    else:
        print('Wrong number of input arguments!')
        return None

input = pd.read_csv('EQ.csv')
X = input.iloc[:, :-1].values
Y = input.iloc[:, -1].values

# Direct least square regression

least_square = np.dot((np.dot(np.linalg.inv(np.dot(X.T,X)),X.T)),Y)
After_Pretreat = pretreat(X,'autoscaling')

print(least_square)
print(After_Pretreat)
