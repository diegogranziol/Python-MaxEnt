    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:44:57 2018

@author: binxinru & diegogranziol
"""

import networkx as nx
from scipy import io,sparse, special
import matplotlib.mlab as mlab
from time import time
import numpy as np
import scipy as sp
import os
from scipy.optimize import minimize
import pickle

K = '-lj-'
print('reading graph')

g = nx.Graph()

with open('lj.ungraph.txt') as f:
    for i in xrange(4):
        f.next()
    for line in f:
        myList = [line.strip().split()]
        g.add_edges_from(myList)

A = nx.to_scipy_sparse_matrix(g)
print('file read')


D = A.sum(axis=1)
D1 = (1/np.sqrt(D))

joke2 = []
for i in range(0,(D.shape[0])):
    joke2.append(float(D[i]))
D = sparse.diags(joke2, 0)

joke = []
for i in range(0,(D1.shape[0])):
    joke.append(float(D1[i]))
D1 = sparse.diags(joke, 0)

L = D1.dot((D-A).dot(D1))
for K0 in range(0,1):
    Ld = L
    l = 100
    m = 200
    epsilon = 0
    gmin = 0-epsilon
    gmax = 1+epsilon
    gridlength = 1e4+1
    eigmax = 2
    M = Ld /eigmax
    nmat = M.shape[0]
    G = np.zeros((l,m))
    Z = np.random.randn(nmat,l) 
    Z = Z / sp.linalg.norm(Z, axis=0)
    q1 = Z
    q2 = M.dot(q1)
    
    for i in range(0,m):
        print('writing moment '+str(i))
        A = Z.T.dot(q2)
        G[:,i] = A.diagonal()
        q3 = 2*M.dot(q2) - q1
        q1 = q2
        q2 = q3
    moments = np.mean(G, axis=0)
    inputs = np.insert(moments,0,1)
    
    #% compute Chebyshev polynomial
    x = np.linspace(gmin,gmax,gridlength)
    
    v = np.diff(x)
    v = np.append(v,0)
    n = len(inputs)
    
    chebarray = np.zeros((n,int(gridlength)))
    for i1 in range(0,n):
        q = (special.eval_chebyt(i1, x, out=None))
        chebarray[i1,:]= q;
    
    typec = 'chebyshev'
    
    #% MaxEnt Algorithm
    entr = []
    momnum = []
    sharpness = (1e-3)
    
    
    #% store datas
    #Y = np.zeros((m,int(gridlength)))
    MaxEntdistri = np.zeros((m,int(gridlength)))
    MaxEntCoefficient  = []
    Xbound = np.zeros(m)
    Prediction = np.zeros(m)
    
    
    for ww in range(3,m-1):
    
       mu2 = inputs[0:ww]
       m = ww
    
       #entropic functional
       def s(alpha):
           j = 1+np.dot(alpha,chebarray[:-(n-len(alpha))])
           q = np.exp(-j)
           u = (sum(q*v))+np.dot(mu2,alpha[:]);
           return u
       #check if constraint is satistfied
       def cons(alpha):
           j = 1+np.dot(alpha,chebarray[:-(n-m)])
           q = np.exp(-j)
           u = sum(q*((special.eval_chebyt(alpha, x, out=None)))*v)
           return u
       #gradient of S wrt alpha
       def grad(alpha):
           u = []
           j = []
           j = 1+np.dot(alpha,chebarray[:-(n-len(alpha))])
           q = np.exp(-j)
           u = mu2-np.dot(chebarray[:-(n-len(alpha))],(q*v))
           return np.asarray(u)
       #hessian
       def hessian(alpha):
           j = []
           j = 1+np.dot(alpha,chebarray[:-(n-len(alpha))])
           q = np.exp(-j)
           sd = np.array(chebarray[:-(n-len(alpha))])
           ass = np.einsum('j...,i...->ij...',sd,sd)
           ds = ass*q*v
           dss = np.sum(ds,axis=2)
           hdss = 0.5*(dss + dss.transpose())+1e-3*np.eye(len(alpha))
           return hdss
      
       #oprimization
       #do trust-ncg with grad and hess
       t0 = time()
       res = minimize(s, x0 = np.ones(len(mu2)), method = 'trust-ncg', jac=grad, hess=hessian, options={'gtol': 1e-3, 'disp': True, 'maxiter': None})
       print(ww)
    #   print ("Elapsed time:", 1000*(time()-t0), "ms")
       #using of results
       entropy = res.fun
       alpha = res.x
       MaxEntCoefficient.append(alpha)
       
       j = 1;
       if (typec == 'power'):
           for i in range(0,ww):
           #use chebyt for chebyshev moments and x**i for power moments
               j = j + alpha[i]*x**i
       if (typec == 'chebyshev'):
           for i in range(0,ww):
           #use chebyt for chebyshev moments and x**i for power moments
               j = j + alpha[i]*((special.eval_chebyt(i, x, out=None)))
       if (typec == 'legendre'):
           for i in range(0,ww):
           #use chebyt for chebyshev moments and x**i for power moments
               j = j + alpha[i]*((special.eval_legendre(i, x, out=None)))
       #print j
       p = 1/np.exp(j)
       pplot = (1/np.exp(j))/(sum(p*v))
    
       MaxEntdistri[ww,:] = pplot
       momnum.append(ww)
    
       #% predicting the number of clusters by numerical integration
       idx=np.where(pplot==max(pplot))[0][0]
       xmax=x[idx]
       j = 200
       difj = 1
       while ( difj >= 1e-4):
           difj = np.diff(pplot)[idx - j]
           j = j+1
    
       xend = idx - j
#      print ('moments=' +str(ww))
#print 'xend = '+str(x[xend])
       a=v[0:xend]*pplot[0:xend] 
       clusters = sum(a)*nmat
#       print 'prediction=' + str(clusters)
       Xbound[ww] = xend
       Prediction[ww] = clusters
       
    #%% save files
       present_path = os.getcwd()
       os.chdir(present_path+'/Data')
       pickle.dump( Xbound, open( 'Xbound'+str(K), "wb" ) )
       pickle.dump( Prediction, open( 'Prediction'+str(K), "wb" ) )
       pickle.dump( MaxEntdistri, open( 'MaxEntdistri'+str(K), "wb" ) )
       pickle.dump( MaxEntCoefficient, open( 'MaxEntCoefficient'+str(K), "wb" ) )
    
       sp.io.savemat('MaxEntdistri'+str(K)+'.mat', mdict={'MaxEntdistri': MaxEntdistri})
       sp.io.savemat('MaxEntCoefficient'+str(K)+'.mat', mdict={'MaxEntCoefficient': MaxEntCoefficient})
       sp.io.savemat('Prediction'+str(K)+'.mat', mdict={'Prediction': Prediction})

       os.chdir(present_path)
