from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from numpy import pi,cosh,exp,round,zeros,arange,real
from numpy.fft import fft,ifft
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import scipy.stats as st
from scipy import signal


def antialiasing(u_hat,v_hat):
    N = len(u_hat)
    M = 3 * int(N/2)
    u_hat_pad = np.concatenate((u_hat[0:int(N/2)],np.zeros((int(M-N))),u_hat[int(N/2):]))
    v_hat_pad = np.concatenate((v_hat[0:int(N/2)],np.zeros((int(M-N))),v_hat[int(N/2):]))
    
    u_pad = np.fft.ifft(u_hat_pad)
    v_pad = np.fft.ifft(v_hat_pad)
    
    w_pad = u_pad*v_pad
    w_pad_hat = np.fft.fft(w_pad)
    w_hat = 3.0/2 * np.concatenate((w_pad_hat[0:int(N/2)], w_pad_hat[M-int(N/2):M]))  
    #print (w_hat)
    return w_hat
    
def matrix_antialiasing(h,m):
   
    r = np.zeros((len(m),len(m))) 
    for i in range(0,len(m)): 
        r_temp= antialiasing(h[i,:],m[i,:])    
        
        r[i,:]= r_temp    
                
    return r      
### integrating factor method

    #######################################
    

######################################################################################



# Time-stepping by Runge-Kutta





def spectral_del(nmax,dt, N, H0,U0,V0,time,tmax):
    k = zeros(N)
    k[0:int(N/2)] = arange(0,int(N/2))
    k[int(N/2)+1:] = arange(int(-N/2)+1,0,1)
    #print(k)

    ik1 = 1j*k # 1j*k**3
    ik2 = 1j*k# 1j*k**3
    ik3 = 1j*k# 1j*k**3
    
    U = np.zeros((len(U0),len(U0),3))
    U[:,:,0]=H0#=swe(U0,V0,H0)
    U[:,:,1]=H0*U0
    U[:,:,2]=H0*V0


    
    n=0
   
    for n in range(0,nmax+1):
        v = np.fft.fft2(U[:,:,0])
        v1 = np.fft.fft2(U[:,:,1])
        v2 = np.fft.fft2(U[:,:,2])
        t = float(n*dt); g = ik1*dt
       
        E = (U0+V0)#np.dot(ik1,1)
        E2 = matrix_antialiasing(E,E)
        a = g * matrix_antialiasing(v,v)
        sub1=matrix_antialiasing(E,(v+a/2))
        b = g * matrix_antialiasing(sub1,sub1) 
        sub2=matrix_antialiasing(E,(v+b/2)) 
        c = g * matrix_antialiasing(sub2,sub2) 
        sub3=matrix_antialiasing(E2,v)#matrix_antialiasing(E2,v)
        sub4=matrix_antialiasing(E,c)#matrix_antialiasing(E,c) 
        d = g * matrix_antialiasing(sub3+sub4,sub3+sub3) 
        
        v = E2*v + (E2*a + 2*E*(b+c) + d)/6
    
        h = real(np.fft.ifft2(v))
    ##########################
        g = ik2*dt
        E = np.exp(-dt*ik2*V0); E2 = matrix_antialiasing(E,E)
        suba=real(np.fft.ifft2( v1         ))
        a = g * np.fft.fft2(matrix_antialiasing(suba,suba))
        subb=matrix_antialiasing(E,(v1+a/2))
        subbb=real(np.fft.ifft2( subb       ))
        b = g * np.fft.fft2(matrix_antialiasing(subbb,subbb))
        suzz=matrix_antialiasing(E,(v1+b/2))
        suzzz=real(np.fft.ifft2( suzz       ))
        c = g * np.fft.fft2(matrix_antialiasing(suzzz,suzzz))
        subd=matrix_antialiasing(E2,v1)+matrix_antialiasing(E,c)#E2*v1+E*c
        subdd=real(np.fft.ifft2( subd       ))
        d = g * np.fft.fft2(matrix_antialiasing(subdd,subdd))
        v1 = E2*v1 + (E2*a + 2*E*(b+c) + d)/6
        
        h1 = real(np.fft.ifft2(v1))
    ######################
        g = ik3*dt
        E = np.exp(-dt*ik3*U0); E2 = matrix_antialiasing(E,E)
        suba=real(np.fft.ifft2( v2         ))
        a = g * np.fft.fft2(matrix_antialiasing(suba,suba))
        subb=matrix_antialiasing(E,(v2+a/2))
        subbb=real(np.fft.ifft2( subb       ))
        b = g * np.fft.fft2(matrix_antialiasing(subbb,subbb))
        suzz=matrix_antialiasing(E,(v2+b/2))
        suzzz=real(np.fft.ifft2( suzz       ))
        c = g * np.fft.fft2(matrix_antialiasing(suzzz,suzzz))
        subd=matrix_antialiasing(E2,v2)+matrix_antialiasing(E,c)#E2*v1+E*c
        subdd=real(np.fft.ifft2( subd       ))
        d = g * np.fft.fft2(matrix_antialiasing(subdd,subdd))
        
        v2 = E2*v2 + (E2*a + 2*E*(b+c) + d)/6
  
        h2 = real(np.fft.ifft2(v2))
        
        U[:,:,0]=h
        U[:,:,1]=h1
        U[:,:,2]=h2
        
        ux=U[:,:,1]/U[:,:,0]
        vy=U[:,:,2]/U[:,:,0]
        U0=ux
        V0=vy
        
        ux=np.amax(ux)
        vy=np.amax(vy)
        print(ux)
        print(vy)
        #U[0,:,1] = 0#  U[-1,:,1]
        
        #dt = (.95*dx)/(np.abs(ux+vy))
       
        print("t=")
        print(t)
        print("N=")
        print(N)
    
        print("dt=")
        print(dt)
        n=n+1
        print(n)
        
        if t==0.0:
            fig = plt.figure()
            ax = Axes3D(fig)
            z = real(U[:,:,0])
            y = x
            ax.plot_surface(xx,yy,z)
            ax.set_zlim(-.05, .2)
            ax.set_title("t = %f"%(t))
            plt.show()   
        
              
        if t==.0016:
            fig = plt.figure()
            ax = Axes3D(fig)
            z = real(U[:,:,0])
            y = x
            ax.plot_surface(xx,yy,z)
            ax.set_zlim(-.05, .2)
            ax.set_title("t = %f"%(t))
            plt.show()   
       
        if t==.004:
            fig = plt.figure()
            ax = Axes3D(fig)
            z = real(U[:,:,0])
            y = x
            ax.plot_surface(xx,yy,z)
            ax.set_zlim(-.05, .2)
            ax.set_title("t = %f"%(t))
            plt.show()     
            
       
        if t==.0056:
            fig = plt.figure()
            ax = Axes3D(fig)
            z = real(U[:,:,0])
            y = x
            ax.plot_surface(xx,yy,z)
            ax.set_zlim(-.05, .2)
            ax.set_title("t = %f"%(t))
            plt.show()           
    
        if t==.012:
            fig = plt.figure()
            ax = Axes3D(fig)
            z = real(U[:,:,0])
            y = x
            ax.plot_surface(xx,yy,z)
            ax.set_zlim(-.05, .2)
            ax.set_title("t = %f"%(t))
            plt.show() 
        
    return U
     


#ni=[100,120,140,160,180,200]
ni=[100]


for i in range(0,len(ni)):
    NN=ni[i]

    ######################################################################################lin
    time=0.0
    dt=4.0/NN**2  #5.0 is the best

    t1 = dt*50
    tmax = time + t1
    nmax= int(round(t1/dt))  
    print (nmax)

    
    x = (2*pi/NN)*np.linspace(-NN/2,NN/2,NN)#
    #np.cos((pi*arange(0,N))/N); 
    dx=np.abs(x[1]-x[0])
    dy=dx
    y = x
    [xx,yy] = np.meshgrid(x,y)    
    def gauss_swe(sig,NN,t0):
        
        x = (2*pi/NN)*np.linspace(-NN/2,NN/2,NN)
        dx=np.abs(x[1]-x[0])
        dy=dx
        y = x
        M =.9#
        g =-.2#.
        c1=14.0
        c2=11.0
        alpha= np.pi
        x_o=-2.0#
        y_o=-1.0#
        [xx,yy] = np.meshgrid(x,y)
        f = lambda t,xx,yy: -c2*((xx-x_o-M*t*np.cos(alpha))**2+(yy-y_o-M*t*np.sin(alpha))**2)  
        H0 = 1 - (c1**2/(4*c2*g))*np.exp(2*f(t0,xx,yy))
        U0 = M*np.cos(alpha)+c1*(yy-y_o-M*t0*np.sin(alpha))*np.exp(f(t0,xx,yy))    # 12/8 i changed the tmax to t
        V0 = M*np.sin(alpha)-c1*(xx-x_o-M*t0*np.cos(alpha))*np.exp(f(t0,xx,yy))
        return H0,U0,V0
    

    H0,U0,V0 = gauss_swe(40,NN,0.0)
   
    U0 = np.ones((len(x),len(x)))*.05
    V0 =np.ones((len(x),len(x))) *.05
    U=spectral_del(nmax,dt, NN, H0,U0,V0,time,tmax)
    
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
sol_plot = axes.pcolor(real(H0), cmap=plt.get_cmap('RdBu_r'))
#ax.plot_surface(X,Y,real(H0_two), cmap=plt.get_cmap('RdBu_r'))
cbar = fig.colorbar(sol_plot)  
axes.set_title("H actual t = %f"%(time)) 
plt.show() 
    
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
sol_plot = axes.pcolor(U[:,:,0], cmap=plt.get_cmap('RdBu_r'))
#ax.plot_surface(X,Y,real(H0_two), cmap=plt.get_cmap('RdBu_r'))
cbar = fig.colorbar(sol_plot)  
axes.set_title("H computed t = %f"%(tmax)) 
plt.show() 



fig = plt.figure()
ax = Axes3D(fig)
z = real(U[:,:,0])
y = x
ax.plot_surface(xx,yy,z)
ax.set_zlim(-.05, .2)
ax.set_title("t = %f"%(t1))
plt.show()



     

           