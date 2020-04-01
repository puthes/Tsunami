from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as st


from matplotlib.collections import LineCollection
from numpy import pi,cosh,exp,round,zeros,arange,real
from numpy.fft import fft,ifft
from matplotlib.pyplot import figure
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import signal



                
def chebfft(v,x):    #this is the first derivative  1-D
    N = len(v)-1;
    if N==0: return 0
    ii = np.arange(0,N); iir = np.arange(1-N,0); iii = np.array(ii,dtype=int)
    #print (v)
    V = np.hstack((v,v[N-1:0:-1]))
    U = np.real(fft(V))
    W = np.real(ifft(1j*np.hstack((ii,[0.],iir))*U))
    w = np.zeros(N+1)
    w[1:N] = -W[1:N]/np.sqrt(1-x[1:N]**2)     
    w[0] = sum(iii**2*U[iii])/N + .5*N*U[N]     
    w[N] = sum((-1)**(iii+1)*ii**2*U[iii])/N + .5*(-1)**(N+1)*N*U[N]
    return w






def convert(U,F,G,N,tmax,t,g1,M,g):
    H0 = U[:,:,0]
    U0 = U[:,:,1]/U[:,:,0]
    V0 = U[:,:,2]/U[:,:,0]
    x = np.cos((pi*arange(0,N))/N); 
    dx=np.abs(x[1]-x[0])
    dy=dx
    y = x
    [xx,yy] = np.meshgrid(x,y)
    
  
    #U = np.zeros((len(x),len(x),3)) 
    #F = np.zeros((len(x),len(x),3)) 
    #G = np.zeros((len(x),len(x),3)) 
    U1 = np.zeros((len(x),len(x),3)) 
    F1 = np.zeros((len(x),len(x),3)) 
    G1 = np.zeros((len(x),len(x),3)) 
    ###
 
    ############
   
    ############
    U1[:,:,0]=H0
    U1[:,:,1]=H0*U0
    U1[:,:,2]=H0*V0
    F1[:,:,0]=H0*U0
    F1[:,:,1]=H0*U0*U0+.5*g*H0*H0
    F1[:,:,2]=H0*U0*V0
    G1[:,:,0]=H0*V0
    G1[:,:,1]=H0*U0*V0
    G1[:,:,2]=H0*V0*V0+.5*g*H0*H0
        

    U[:,:,0] = (U1[:,:,0]) # fft computing and normalization
    U[:,:,1] = (U1[:,:,1]) 
    U[:,:,2] = (U1[:,:,2])
        
    F[:,:,0] = (F1[:,:,0]) # fft computing and normalization
    F[:,:,1] = (F1[:,:,1])
    F[:,:,2] = (F1[:,:,2])
         
    G[:,:,0] = G1[:,:,0] # fft computing and normalization
    G[:,:,1] = G1[:,:,1] 
    G[:,:,2]= G1[:,:,2]
    ##############
        
    return F,G   
    
    

      

def vortex(N,tmax,nmax,dt,g1,M,g,U0,V0,H0):
    
    tfinal=tmax
     
    dt1=dt
    
    m = N
  
    c1=20.0#7.0#.04
    c2=14.0#5.0#.02
    alpha= np.pi/6#-3*np.pi/4 #np.pi
    x_o=.4#.4#-20 
    y_o=.2#0.2#-10
    
    x = np.cos((pi*arange(0,N))/N); 
    dx=np.abs(x[1]-x[0])
    dy=dx
    y = x
    [xx,yy] = np.meshgrid(x,y)
    U_euler = np.zeros((len(x),len(x),3))
    U = np.zeros((len(x),len(x),3)) 
    F = np.zeros((len(x),len(x),3)) 
    G = np.zeros((len(x),len(x),3)) 
    U1 = np.zeros((len(x),len(x),3)) 
    F1 = np.zeros((len(x),len(x),3)) 
    G1 = np.zeros((len(x),len(x),3)) 
    ###

    ##
     
    
   
    ############
    ############
    U1[:,:,0]=H0
    U1[:,:,1]=H0*U0
    U1[:,:,2]=H0*V0
    F1[:,:,0]=H0*U0
    F1[:,:,1]=H0*U0*U0+.5*g*H0*H0
    F1[:,:,2]=H0*U0*V0
    G1[:,:,0]=H0*V0
    G1[:,:,1]=H0*U0*V0
    G1[:,:,2]=H0*V0*V0+.5*g*H0*H0
        

    U[:,:,0] = (U1[:,:,0]) # fft computing and normalization
    U[:,:,1] = (U1[:,:,1]) 
    U[:,:,2] = (U1[:,:,2])
    
        
    F[:,:,0] = (F1[:,:,0]) # fft computing and normalization
    F[:,:,1] = (F1[:,:,1])
    F[:,:,2] = (F1[:,:,2])
         
    G[:,:,0] = G1[:,:,0] # fft computing and normalization
    G[:,:,1] = G1[:,:,1] 
    G[:,:,2]= G1[:,:,2]
    ###############
    
     
    vold=U0  #H
    vv=H0  #H at dt
    vold2temp= U0  # U at 0
    vv2temp = U0 # U at dt
    vold2=U0    # HU
  
    vv2= U0 #np.dot(vold,vold2temp)  # HU at .1
    vold3temp= U0   #V
    vv3temp= V0 # V at dt
    vold3= U0  # HV
    vv3=U0#HV at .1
    
    vv_euler=vv    
    vv2_euler=vv2
    vv3_euler=vv3
    
      # on hu
         
    
        
   # U[:,-1,1] =  0#U[0,:,1]
    n=0
    t=0
    for i in range(0,nmax):
        t = n*dt
        print ("t=")
        print (t)
        print ("n=")
        print (n)
        print (N)
        #
        
        ##########################################################
        
        if t > 0:
            vold = vv
            vv = U[:,:,0]
            vold2 = vv2
            vv2 = U[:,:,1]
            vold3 =vv3
            vv3 = U[:,:,2]
            
            vv_euler = U_euler[:,:,0]
            vv2_euler= U_euler[:,:,1]
            vv3_euler= U_euler[:,:,2]
       
        for k in range(0,N):
         
            #F,G = convert(U,F,G,N,tmax,t) 
            
            x = np.cos((pi*arange(0,N))/N); 
            y=x
            dx=np.abs(x[0]-x[1])
            [xx,yy] = np.meshgrid(x,y)
              
            ux0=chebfft(F[:,k,0], x)
            uy0=chebfft(G[k,:,0], x)
            vvnew1 = 2*vv[k,:]-vold[k,:]+(dt**2)*(-ux0-uy0) #leapfrog
            vvnew1_euler = vv_euler[k,:]+dt*(-ux0-uy0) #euler
            vold[k,:]=vv[k,:]
            vv[k,:] = vvnew1
            U[k,:,0] = vvnew1# vv[k,:]
            
            vv_euler[k,:] = vvnew1_euler
            U_euler[k,:,0]= vvnew1_euler
        
########################################################################################################     

            
            #dt= (dx * .5)/max5
          
            
            ux1=chebfft(F[:,k,1], x)
            uy1=chebfft(G[k,:,1], x)
          
            vvnew2 = 2*vv2[k,:]-vold2[k,:]+(dt**2)*(-ux1-uy1) #leapfrog
            vvnew2_euler = vv2_euler[k,:]+dt*(-ux1-uy1) #euler
            vold2[k,:]=vv2[k,:]
            vv2[k,:] = vvnew2
            U[k,:,1] = vvnew2#vv2[k,:]
            
            vv2_euler[k,:] = vvnew2_euler
            U_euler[k,:,1]= vvnew2_euler
 ################################################################################################################                     
            
            #dt= (dx * .5)/max5
            t = n*dt
              
           
            ux2=chebfft(F[:,k,2], x)
            uy2=chebfft(G[k,:,2], x)
            vvnew3 = 2*vv3[k,:]-vold3[k,:]+(dt**2)*(-ux2-uy2) #leapfrog
            vvnew3_euler = vv3_euler[k,:]+dt*(-ux2-uy2) #euler
            vold3[k,:]=vv3[k,:]
            vv3[k,:] = vvnew3
            U[k,:,2] = vvnew3#vv3[k,:]                 
                                            
            vv3_euler[k,:] = vvnew3_euler
            U_euler[k,:,2]= vvnew3_euler 
            
        if t==0:
            fig = plt.figure()
            ax = Axes3D(fig)
            z = real(U[:,:,0])
            y = x
            ax.plot_surface(xx,yy,z)
            ax.set_title("t = %f"%(t))
            plt.show()
        if t==.01:
            fig = plt.figure()
            ax = Axes3D(fig)
            z = real(U[:,:,0])
            y = x
            ax.plot_surface(xx,yy,z)
            ax.set_title("t = %f"%(t))
            plt.show()
     
        if t==.05:
            fig = plt.figure()
            ax = Axes3D(fig)
            z = real(U[:,:,0])
            y = x
            ax.plot_surface(xx,yy,z)
            ax.set_title("t = %f"%(t))
            plt.show()
        if t==.09:
            fig = plt.figure()
            ax = Axes3D(fig)
            z = real(U[:,:,0])
            y = x
            ax.plot_surface(xx,yy,z)
            ax.set_title("t = %f"%(t))
            plt.show() 
        if t==.1:
            fig = plt.figure()
            ax = Axes3D(fig)
            z = real(U[:,:,0])
            y = x
            ax.plot_surface(xx,yy,z)
            ax.set_title("t = %f"%(t))
            plt.show()
        if t==.11:
            fig = plt.figure()
            ax = Axes3D(fig)
            z = real(U[:,:,0])
            y = x
            ax.plot_surface(xx,yy,z)
            ax.set_title("t = %f"%(t))
            plt.show()   
        if t==.12:
            fig = plt.figure()
            ax = Axes3D(fig)
            z = real(U[:,:,0])
            y = x
            ax.plot_surface(xx,yy,z)
            ax.set_title("t = %f"%(t))
            plt.show() 
        if t==.13:
            fig = plt.figure()
            ax = Axes3D(fig)
            z = real(U[:,:,0])
            y = x
            ax.plot_surface(xx,yy,z)
            ax.set_title("t = %f"%(t))
            plt.show()
                       
           
        F,G = convert(U,F,G,N,tmax,t,g1,M,g) 
        n = n+1
        
        
        #dt = (.95*dx)/(np.abs(max_valx+max_valy))
        
          # on hu
          
           
        
                                                                                                             
            
    return U,U_euler 


#ni = [50,55,60,65]

#ni= [50]
linf = np.zeros(1)
linf_euler = np.zeros(1)
N= np.zeros(1)
k=0

tmax=.13#1.55#.01#.01 is good 
NN=256
print (NN)
x = np.cos(pi*arange(0,NN)/NN); 
y=x
g=9.8  #acceleration due to gravity
g1=.5
M=.08
m = NN


dx=np.abs(x[1]-x[0])
dy=dx
[xx,yy] = np.meshgrid(x,y)
    

def gauss_swe(sig,NN):
        
    
        x = np.cos(pi*arange(0,NN)/NN); 
        dx=np.abs(x[1]-x[0])
        dy=dx
        y = x
        M =.08#
        g =-50.0#.
        c1=200.0
        c2=100.0
        alpha= 4*np.pi
        x_o=.5#
        y_o=.2#
        [xx,yy] = np.meshgrid(x,y)
        f = lambda t,xx,yy: -c2*((xx-x_o-M*t*np.cos(alpha))**2+(yy-y_o-M*t*np.sin(alpha))**2)  
        H0 = 1 - (c1**2/(4*c2*g))*np.exp(2*f(0.0,xx,yy))
       
        return H0
    

H0 = gauss_swe(15,NN)

U0 = np.ones((len(x),len(x))) 
V0 =np.ones((len(x),len(x)))


   
dt = .01#/NN**2#
nmax = int(tmax/(dt))
    
    
U,U_euler= vortex(NN,tmax,nmax,dt,g1,M,g,U0,V0,H0) 
    
k = k + 1






fig = plt.figure()
ax = Axes3D(fig)
z = real(U[:,:,0])
y = x
ax.plot_surface(xx,yy,z)
ax.set_title("t = %f"%(tmax))
plt.show()