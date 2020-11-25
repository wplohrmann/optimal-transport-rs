from optimal_transport.rust import sinkhorn
import numpy as np
import matplotlib.pyplot as plt

plotp = lambda x,col: plt.scatter(x[0,:], x[1,:], s=150, edgecolors="k", c=col, linewidths=2)
def distmat(x,y):
    return np.sum(x**2,0)[:,None] + np.sum(y**2,0)[None,:] - 2*x.transpose().dot(y)

def mina_u(H,epsilon): return -epsilon*np.log( np.sum(a * np.exp(-H/epsilon),0) )
def minb_u(H,epsilon): return -epsilon*np.log( np.sum(b * np.exp(-H/epsilon),1) )

def mina(H,epsilon): return mina_u(H-np.min(H,0),epsilon) + np.min(H,0);
def minb(H,epsilon): return minb_u(H-np.min(H,1)[:,None],epsilon) + np.min(H,1);

def Sinkhorn(C,epsilon,f,niter = 500):    
    Err = np.zeros(niter)
    for it in range(niter):
        g = mina(C-f[:,None],epsilon)
        f = minb(C-g[None,:],epsilon)
        # generate the coupling
        P = a * np.exp((f[:,None]+g[None,:]-C)/epsilon) * b
        # check conservation of mass
        Err[it] = np.linalg.norm(np.sum(P,0)-b,1)
    return (P,Err,f)


n = 4
m = 200
a = np.ones((n,1))/n
b = np.ones((1,m))/m
a1= a[:,0].astype(np.float32)
b1 = b[0].astype(np.float32)
epsilon = 0.0001
niter = 300

z = np.random.randn(2,n)*.2
z[1] *= 0.5

z = [[-0.4,-0.2,0,0.2],[0.1,0,0.1,0]]

y = np.random.randn(2,m)*.2
y = np.random.randn(m//4,2)*.02
y = np.vstack((y+[0,-0.8],y+[0.2,-0.4],y+[0,0],y+[0.2,0.4])).transpose()
y[0,:] += 1



theta = 0
A = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
#S = np.ones(2)
S = 1
h = np.zeros(2)
Swap = np.array([[0,-1],[1,0]])

# step size for the descent
tau_S = 0.4
tau_h = 1
tau_theta = 4
# #iter for the gradient descent
giter = 200
ndisp = np.round( np.linspace(0,giter-1,6) )
kdisp = 0
f = np.zeros(n)
for j in range(giter):
    x = A.dot(z)*S+h[:,None]
    if ndisp[kdisp]==j:
        ax = plt.subplot(2,3,kdisp+1)
        ax.set_title(f"theta={theta:.2f}, h=[{h[0]:.2f},{h[1]:.2f}], S={S}")
        plotp(y, 'r')
        plotp(x, 'b')
        kdisp = kdisp+1
        plt.xlim(-.7,1.3)
        plt.ylim(-1,.7)
    (P,Err,f) = Sinkhorn(distmat(x,y), epsilon,f,niter)
    #P = sinkhorn(a1, b1, distmat(x,y).astype(np.float32), epsilon)
    v = a.transpose() * x - y.dot(P.transpose())
    A_deriv = Swap@A
    delta_theta = A_deriv@z*S
    nabla_theta = np.sum(delta_theta*v)
    delta_S = A@z
    nabla_S = np.sum(delta_S*v)
    nabla_h = np.sum(v,1)
    theta = theta - tau_theta * nabla_theta
    A = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    S = S - tau_S * nabla_S
    h = h - tau_h * nabla_h
    print(theta)
plt.show()
