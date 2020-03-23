from mpl_toolkits.mplot3d import Axes3D


from matplotlib.collections import LineCollection
from numpy import pi,cosh,exp,round,zeros,arange,real
from numpy.fft import fft,ifft
from matplotlib.pyplot import figure
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection




                
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




def antialiasing(u_hat,v_hat):
    N = len(u_hat)
    M = 3 * int(N/2)
    ss = (u_hat[int(N/2):])
    u_hat_pad = np.concatenate((u_hat[0:int(N/2)],np.zeros((int(M-N))),u_hat[int(N/2):]))
    v_hat_pad = np.concatenate((v_hat[0:int(N/2)],np.zeros((int(M-N))),v_hat[int(N/2):]))
    
    u_pad = np.fft.ifft(u_hat_pad)
    v_pad = np.fft.ifft(v_hat_pad)
    
    w_pad = u_pad*v_pad
    w_pad_hat = np.fft.fft(w_pad)
    w_hat = 3.0/2 * np.concatenate((w_pad_hat[0:int(N/2)], w_pad_hat[M-int(N/2):M]))   # 2/3 is better than 3.0/2
    #print (w_hat)
    return w_hat

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
    
    
def swe(time,N,g1,M,g):  
    
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
    f = lambda t,xx,yy: -c2*((xx-x_o-M*t*np.cos(alpha))**2+(yy-y_o-M*t*np.sin(alpha))**2)  
    H0 = 1 - (c1**2/(4*c2*g1))*np.exp(2*f(time,xx,yy))
    U0 = M*np.cos(alpha)+c1*(yy-y_o-M*time*np.sin(alpha))*np.exp(f(time,xx,yy))    # 12/8 i changed the tmax to t
    V0 = M*np.sin(alpha)-c1*(xx-x_o-M*time*np.cos(alpha))*np.exp(f(time,xx,yy))
   
   
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
    
    return U
      

def vortex(N,tmax,nmax,dt,g1,M,g):
    
    #t=0

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
    f = lambda t,xx,yy: -c2*((xx-x_o-M*t*np.cos(alpha))**2+(yy-y_o-M*t*np.sin(alpha))**2)  
    H0 = 1 - (c1**2/(4*c2*g1))*np.exp(2*f(0,xx,yy))
    H0_origin = 1 - (c1**2/(4*c2*g1))*np.exp(2*f(0,xx,yy))
    U0 = M*np.cos(alpha)+c1*(yy-y_o-M*0*np.sin(alpha))*np.exp(f(0,xx,yy))    # 12/8 i changed the tmax to t
    V0 = M*np.sin(alpha)-c1*(xx-x_o-M*0*np.cos(alpha))*np.exp(f(0,xx,yy))
    V0_two = M*np.sin(alpha)-c1*(xx-x_o-M*tmax*np.cos(alpha))*np.exp(f(tmax,xx,yy))
    H0_two = 1 - (c1**2/(4.0*c2*g1))*np.exp(2*f(tmax,xx,yy))
    U0_two = M*np.cos(alpha)+c1*(yy-y_o-M*tmax*np.sin(alpha))*np.exp(f(tmax,xx,yy))
    
   
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
    
     
    vold=1 - (c1**2/(4*c2*g1))*np.exp(2*f(0,xx,yy))  #H
    vv=1 - (c1**2/(4*c2*g1))*np.exp(2*f(dt1,xx,yy))  #H at dt
    vold2temp= M*np.cos(alpha)+c1*(yy-y_o-M*0*np.sin(alpha))*np.exp(f(0,xx,yy))   # U at 0
    vv2temp = M*np.cos(alpha)+c1*(yy-y_o-M*dt1*np.sin(alpha))*np.exp(f(dt1,xx,yy)) # U at dt
    vold2= U[:,:,1]    # HU
    U22 = swe(dt1,N,g1,M,g)
    vv2= U22[:,:,1] #np.dot(vold,vold2temp)  # HU at .1
    vold3temp= M*np.sin(alpha)-c1*(xx-x_o-M*0*np.cos(alpha))*np.exp(f(0,xx,yy))   #V
    vv3temp= M*np.sin(alpha)-c1*(xx-x_o-M*dt1*np.cos(alpha))*np.exp(f(dt1,xx,yy)) # V at dt
    vold3= U[:,:,2]  # HV
    vv3=U22[:,:,2] #HV at .1
    
    vv_euler=vv    
    vv2_euler=vv2
    vv3_euler=vv3
    
      # on hu
         
    U[0,:,2] = 0#  U[-1,:,1]
        
    U[-1,:,2] =  0#U[0,:,1]
  #  U[:,0,2] = 0#  U[-1,:,1]
        
   # U[:,-1,2] =  0#U[0,:,1]
           
           
       
        # on hu
           
    U[:,0,1] =  0#U[-1,:,2]
    U[:,-1,1] = 0# U[0,:,2
  #  U[:,0,1] = 0#  U[-1,:,1]
        
   # U[:,-1,1] =  0#U[0,:,1]
    n=0
    t=0
    while (t <tmax):
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
            dx=np.abs(x[0]-x[1])
               
              
            ux0=chebfft(F[k,:,0], x)
            uy0=chebfft(G[:,k,0], x)
            vvnew1 = 2*vv[k,:]-vold[k,:]+(dt**2)*(-ux0-uy0) #leapfrog
            vvnew1_euler = vv_euler[k,:]+dt*(-ux0-uy0) #euler
            vold[k,:]=vv[k,:]
            vv[k,:] = vvnew1
            U[k,:,0] = vvnew1# vv[k,:]
            
            vv_euler[k,:] = vvnew1_euler
            U_euler[k,:,0]= vvnew1_euler
        
########################################################################################################     

            
            #dt= (dx * .5)/max5
          
            
            ux1=chebfft(F[k,:,1], x)
            uy1=chebfft(G[:,k,1], x)
          
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
              
           
            ux2=chebfft(F[k,:,2], x)
            uy2=chebfft(G[:,k,2], x)
            vvnew3 = 2*vv3[k,:]-vold3[k,:]+(dt**2)*(-ux2-uy2) #leapfrog
            vvnew3_euler = vv3_euler[k,:]+dt*(-ux2-uy2) #euler
            vold3[k,:]=vv3[k,:]
            vv3[k,:] = vvnew3
            U[k,:,2] = vvnew3#vv3[k,:]                 
                                            
            vv3_euler[k,:] = vvnew3_euler
            U_euler[k,:,2]= vvnew3_euler 
            
        
           
        F,G = convert(U,F,G,N,tmax,t,g1,M,g) 
        n = n+1
        
        ux=U[:,:,1]/U[:,:,0]
        vy=U[:,:,2]/U[:,:,0]
        
        r = ux + np.sqrt(np.abs(g*U[:,:,0]))
        r1 = ux - np.sqrt(np.abs(g*U[:,:,0]))
        rr = vy + np.sqrt(np.abs(g*U[:,:,0]))
        rr1 = vy - np.sqrt(np.abs(g*U[:,:,0]))
        max1=np.amax(np.real(r))
        max2=np.amax(np.real(r1))
        max3=np.amax(np.real(rr))
        max4=np.amax(np.real(rr1))
        max5=np.amax(np.real(ux))
        max6=np.amax(np.real(vy))
        max_valx0=np.maximum(max1,max2)
        max_valx=np.maximum(max5,max_valx0)
        max_valy0=np.maximum(max3,max4)
        max_valy=np.maximum(max6,max_valy0)
        print(max_valx)
        print(max_valy)
        dt = (.95*dx)/(np.abs(max_valx+max_valy))
        
          # on hu
           
        U[0,:,2] = 0#  U[-1,:,1]
        
        U[-1,:,2] =  0#U[0,:,1]
       # U[:,0,2] = 0#  U[-1,:,1]
        
       # U[:,-1,2] =  0#U[0,:,1]
        
        # on hu
           
        #U[:,0,1] =  0#U[-1,:,2]
        #U[:,-1,1] = 0# U[0,:,2
        U[:,0,1] = 0#  U[-1,:,1]
        
        U[:,-1,1] =  0#U[0,:,1]
                                                                                                             
            
    return U, H0_two, H0, U_euler , U0, V0, H0



ni = [20,30,40,50,60,70]
#ni = [50]
linf = np.zeros(6)
linf_euler = np.zeros(6)
N= np.zeros(6)
k=0
for i in ni:

  
   
    tmax=.02#.01 is good 
    NN=i
    print (NN)
    x = np.cos(pi*arange(0,NN)/NN); 
    y=x
    g=.05  #acceleration due to gravity
    g1=.05
    M=.05
    m = N
    c1=20.0#7.0#.04
    c2=14.0#5.0#.02
    alpha= np.pi/6#-3*np.pi/4 #np.pi
    x_o=.4#.4#-20 
    y_o=.2#0.2#-10
    dx=np.abs(x[1]-x[0])
    dy=dx
    [xx,yy] = np.meshgrid(x,y)
    
    f = lambda t,xx,yy: -c2*((xx-x_o-M*t*np.cos(alpha))**2+(yy-y_o-M*t*np.sin(alpha))**2)  
    H0 = 1 - (c1**2/(4*c2*g1))*np.exp(2*f(0,xx,yy))
    H0_origin = 1 - (c1**2/(4*c2*g1))*np.exp(2*f(0,xx,yy))
    U0 = M*np.cos(alpha)+c1*(yy-y_o-M*0*np.sin(alpha))*np.exp(f(0,xx,yy))    # 12/8 i changed the tmax to t
    V0 = M*np.sin(alpha)-c1*(xx-x_o-M*0*np.cos(alpha))*np.exp(f(0,xx,yy))
    
    ux=U0
    vy=V0
        
    r = ux + np.sqrt(np.abs(g*H0))
    r1 = ux - np.sqrt(np.abs(g*H0))
    rr = vy + np.sqrt(np.abs(g*H0))
    rr1 = vy - np.sqrt(np.abs(g*H0))
    max1=np.amax(np.real(r))
    max2=np.amax(np.real(r1))
    max3=np.amax(np.real(rr))
    max4=np.amax(np.real(rr1))
    max5=np.amax(np.real(ux))
    max6=np.amax(np.real(vy))
    max_valx0=np.maximum(max1,max2)
    max_valx=np.maximum(max5,max_valx0)
    max_valy0=np.maximum(max3,max4)
    max_valy=np.maximum(max6,max_valy0)
    print(max_valx)
    print(max_valy)
    dt = (.95*dx)/(np.abs(max_valx+max_valy))
    nmax = int(tmax/(dt))
    
    U, H0_two, origin, U_euler, U0,V0,H0 = vortex(NN,tmax,nmax,dt,g1,M,g) 
    N[k]= dx # dx
    error = np.linalg.norm(dx*(np.abs(real(U[:,:,0]) - real(H0_two))), ord=np.inf)
    error2 = np.linalg.norm(dx*(np.abs(real(U_euler[:,:,0]) - real(H0_two))), ord=np.inf)
    linf[k]= error
    linf_euler[k]=error2
    #error = np.linalg.norm(N[k]*(np.abs(real(vvnew1) - real(H0))), ord=1)
    #linf[k]= error
    k = k + 1






fig = plt.figure()
axes = fig.add_subplot(1, 1, 1) 
order_C = lambda dx, error, order: np.exp(np.log(error) - order * np.log(dx))
axes.loglog(N, order_C(N[0], linf_euler[0], 1.0) * N**1.0, 'b--', label="1st Order")
axes.loglog(N, order_C(N[0], linf_euler[0], 2.0) * N**2.0, 'r--', label="2nd Order")     
axes.loglog(N, order_C(N[0], linf[0], 1.0) * N**1.0, 'b--', label="1st Order")
axes.loglog(N, order_C(N[0], linf[0], 2.0) * N**2.0, 'r--', label="2nd Order")     
axes.loglog(N, linf_euler, 'bs', label="Actual Euler")
axes.loglog(N, linf, 'rs', label="Leap Frog")
plt.xlabel('dx')
plt.ylabel('error')
axes.set_title("Chebyshev Differentiation Method from t = %f to t = %f"%(0.0,tmax))
axes.legend(loc=4)
plt.show()



    
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
sol_plot = axes.pcolor(real(U_euler[:,:,0]), cmap=plt.get_cmap('RdBu_r'))
cbar = fig.colorbar(sol_plot)  
axes.set_title("H computed T = .05 Euler") 
plt.show() 

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
sol_plot = axes.pcolor(real(U[:,:,0]), cmap=plt.get_cmap('RdBu_r'))
cbar = fig.colorbar(sol_plot)  
axes.set_title("H computed T = .005") 
plt.show() 


fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
sol_plot = axes.pcolor(real(origin), cmap=plt.get_cmap('RdBu_r'))
cbar = fig.colorbar(sol_plot)  
axes.set_title("H origin") 
plt.show() 


fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
    #sol_plot = axes.pcolor(fx_prime_hat, cmap=plt.get_cmap('RdBu_r'))
sol_plot = axes.pcolor(real(H0_two), cmap=plt.get_cmap('RdBu_r'))
cbar = fig.colorbar(sol_plot)  
axes.set_title("H actual t= 0.005") 
plt.show() 

u = real(U[:,:,1])/real(U[:,:,0])
v = real(U[:,:,2])/real(U[:,:,0])

fig0, ax0 = plt.subplots()
strm = ax0.streamplot(xx, yy, u, v, color=u, linewidth=2, cmap=plt.cm.autumn)
fig0.colorbar(strm.lines)
ax0.set_title('Computed leap frog at t=.005')
plt.show()


u = real(U_euler[:,:,1])/real(U_euler[:,:,0])
v = real(U_euler[:,:,2])/real(U_euler[:,:,0])

fig0, ax0 = plt.subplots()
strm = ax0.streamplot(xx, yy, u, v, color=u, linewidth=2, cmap=plt.cm.autumn)
fig0.colorbar(strm.lines)
ax0.set_title('Computed Euler at t=.005')
plt.show()


fig0, ax0 = plt.subplots()
strm = ax0.streamplot(xx, yy, U0, V0, color=U0, linewidth=2, cmap=plt.cm.autumn)
fig0.colorbar(strm.lines)
ax0.set_title('Actual at t=.005')
plt.show()


fig = plt.figure()
ax = Axes3D(fig)
z = real(U[:,:,0])
y = x
ax.plot_surface(xx,yy,z)
ax.set_title('3D line plot')
plt.show()

