import networkx as nx
from scipy import io,sparse, special
import matplotlib.mlab as mlab
from time import time
import numpy as np
import scipy as sp
#import matplotlib.pyplot as plt
#import warnings
import os
#import seaborn as sns
from scipy.optimize import minimize
import pickle

filenames = ['roadNet-CA','roadNet-PAD','roadNet-TX','sx-mathoverflow','sx-stackoverflow','amazon','amazon0302']
for ii in range(0,len(filenames)):
    
    K = filenames[ii]
    
    print('reading graph ' +filenames[ii])
    
    g = nx.Graph()
    
    with open(filenames[ii]+'.txt') as f:
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
    #%%
    for K0 in range(0,1):
    #    K =K0+1
    #    Ld = sp.io.loadmat('Ln200p0'+str(K))['Ln'];
        Ld = L
        #%
        #
        #ng = 100;
        #p = 1e-3;
        #M = nx.erdos_renyi_graph(ng, p, seed=None, directed=False)
        #warnings.filterwarnings("ignore")
        #nx.draw(M, with_labels=True, font_weight='bold')
        #plt.show
        ##
        ###%% generate the network
        ##k = 4;
        ##M = nx.watts_strogatz_graph(ng, k, p, seed=None)
        ##nx.draw(M, with_labels=True, font_weight='bold')
        ##plt.show()
        #
        ##%% compute the adjacency and laplacian matrices
        #Ks = nx.adjacency_matrix(M, nodelist=None, weight='1')
        #Kd =Ks.todense()
        #Ls = nx.normalized_laplacian_matrix(M, nodelist=None, weight='1')
        #Ld = Ls.todense()
        #%eigval = np.real(nx.laplacian_spectrum(M, weight='1'))
        #eigval = np.linalg.eigvalsh(Ld)
        #% plot the eigenvalue graphs
        #sns.set_style('whitegrid')
        #sns.kdeplot(np.array(eigval), bw=0.01)
        #plt.show()
        ##############################################################################
        #% compute the chebyshev moments for dense matrix  
          # L - the normalised Laplacian matrix
          # l - number of random vectors
          # m - number of moments used in MaxEnt
        l = 100
        m = 200
        epsilon = 0
        gmin = 0-epsilon
        gmax = 1+epsilon
        gridlength = 1e4+1
        
        
        #L = np.absolute(Ld)
        #eigmax = float(max(L.sum(axis=1)))
        #M = Ld/eigmax
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
        pickle.dump(inputs, open( 'chebyshevmoments'+str(K), "wb" ) )
        
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
           
        #   titlespectra = 'Spectral Densities of MaxEnt with '+str(ww)+' moments'
        #
        #   line1, = plt.plot(x,y1, label="Gaussians")
        #   line2, = plt.plot(x,(pplot), label="MaxEnt")
        #   axes = plt.gca()
        #
        #   #set log or normal plot
        #   axes.set_xlim([gmin,gmax])
        #
        #   maxy = max(max(y1),max(pplot))
        #
        #   axes.set_ylim([0,maxy])
        #   plt.rcParams["figure.figsize"] = (10,10)
        ##       first_legend = plt.legend(handles=[line1], loc=1)
        #   # Add the legend manually to the current Axes.
        ##       ax = plt.gca().add_artist(first_legend)
        #
        #   # Create another legend for the second line.
        #   plt.legend(handles=[line2], loc=2)
        #   #plt.legend(handles=[line3], loc=2)
        #   plt.title(titlespectra)
        #   plt.show()
        
        #   print ('alpha')
        #   print (alpha)
        #   print ('hessian cond')
        #   t = (1/float(2))*(np.matrix(hessian(alpha))+np.matrix(hessian(alpha)).transpose())
        #   e = np.linalg.eigvals(t)
        #   condit = max(e)/min(e)
        #   print (condit)
           #print 'grad alpha'
           #print grad(alpha)
        #   print ('entropy')
        #   print (entropy)
           momnum.append(ww)
        #   entr.append(entropy)
        
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
           pickle.dump( Xbound, open( 'Xbound'+str(K)+'_'+str(ww), "wb" ) )
           pickle.dump( Prediction, open( 'Prediction'+str(K)+'_'+str(ww), "wb" ) )
           pickle.dump( MaxEntdistri, open( 'MaxEntdistri'+str(K)+'_'+str(ww), "wb" ) )
           pickle.dump( MaxEntCoefficient, open( 'MaxEntCoefficient'+str(K)+'_'+str(ww), "wb" ) )
    
        
           sp.io.savemat('MaxEntdistri'+str(K)+'_'+str(ww)+'.mat', mdict={'MaxEntdistri': MaxEntdistri})
           sp.io.savemat('MaxEntCoefficient'+str(K)+'_'+str(ww)+'.mat', mdict={'MaxEntCoefficient': MaxEntCoefficient})
           sp.io.savemat('Prediction'+str(K)+'_'+str(ww)+'.mat', mdict={'Prediction': Prediction})
    
           os.chdir(present_path)
