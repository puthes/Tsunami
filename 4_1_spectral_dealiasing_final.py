from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from numpy import pi,cosh,exp,round,zeros,arange,real
from numpy.fft import fft,ifft
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
from scipy import signal
sns.set(font_scale=1.6)

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



def spectral_del(nmax,dt, N, H0,U0,V0,time,tmax,Cour):
    k = zeros(N)
    k[0:int(N/2)] = arange(0,int(N/2)); k[int(N/2)+1:] = arange(int(-N/2)+1,0,1)

    ik1 = 1j*k # 1j*k**3
    ik2 = 1j*k# 1j*k**3
    ik3 = 1j*k# 1j*k**3
    
    U = np.zeros((len(U0),len(U0),3))
    U[:,:,0]=H0#=swe(U0,V0,H0)
    U[:,:,1]=H0*U0
    U[:,:,2]=H0*V0

    
    n=0 
    t=0    
    #for n in range(0,nmax+1):
    while (t <tmax):    
        v = np.fft.fft2(U[:,:,0])
        v1 = np.fft.fft2(U[:,:,1])
        v2 = np.fft.fft2(U[:,:,2])
        t = n*dt+time; g = ik1*dt
       
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
        a = g * np.fft.fft2(matrix_antialiasing(suba,suba)/h)
        subb=matrix_antialiasing(E,(v1+a/2))
        subbb=real(np.fft.ifft2( subb       ))
        b = g * np.fft.fft2(matrix_antialiasing(subbb,subbb)/h)
        suzz=matrix_antialiasing(E,(v1+b/2))
        suzzz=real(np.fft.ifft2( suzz       ))
        c = g * np.fft.fft2(matrix_antialiasing(suzzz,suzzz)/h)
        subd=matrix_antialiasing(E2,v1)+matrix_antialiasing(E,c)#E2*v1+E*c
        subdd=real(np.fft.ifft2( subd       ))
        d = g * np.fft.fft2(matrix_antialiasing(subdd,subdd)/h)
        v1 = E2*v1 + (E2*a + 2*E*(b+c) + d)/6
        
        h1 = real(np.fft.ifft2(v1))
    ######################
        g = ik3*dt
        E = np.exp(-dt*ik3*U0); E2 = matrix_antialiasing(E,E)
        suba=real(np.fft.ifft2( v2         ))
        a = g * np.fft.fft2(matrix_antialiasing(suba,suba)/h)
        subb=matrix_antialiasing(E,(v2+a/2))
        subbb=real(np.fft.ifft2( subb       ))
        b = g * np.fft.fft2(matrix_antialiasing(subbb,subbb)/h)
        suzz=matrix_antialiasing(E,(v2+b/2))
        suzzz=real(np.fft.ifft2( suzz       ))
        c = g * np.fft.fft2(matrix_antialiasing(suzzz,suzzz)/h)
        subd=matrix_antialiasing(E2,v2)+matrix_antialiasing(E,c)#E2*v1+E*c
        subdd=real(np.fft.ifft2( subd       ))
        d = g * np.fft.fft2(matrix_antialiasing(subdd,subdd)/h)
        v2 = E2*v2 + (E2*a + 2*E*(b+c) + d)/6
        h2 = real(np.fft.ifft2(v2))
        
        U[:,:,0]=h
        U[:,:,1]=h1
        U[:,:,2]=h2
        
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
        dt = (Cour*dx)/(np.abs(max_valx+max_valy))
        
       
        print("t=")
        print(t)
        print("N=")
        print(N)
    
        print("dt=")
        print(dt)
        n=n+1
        print(n)
        print(nmax)
        
        #U[:,0,1] =  0#U[0,:,1]
        #U[:,-1,1] =  0#U[0,:,1] 
        ##U[-1,:,1] =  0#U[0,:,1]

        # on hv
          
        #U[0,:,2] =  0#U[-1,:,2]
        ##U[:,0,2] = 0# U[0,:,2
        ##U[:,-1,2] =  0#U[0,:,1] 
        #U[-1,:,2] =  0#U[0,:,1] 
    
            
       
        
        
    return U,dt
     

#ni=[70]
#ni=[24,26,28,30,32]
#ni=[32,34,36,38]
ni=[40,50,60,70,80,90]
linf=np.zeros(6)
NN=np.zeros(6)

for i in range(0,len(ni)):
    N=ni[i]

    ######################################################################################lin
    time=0.0
    #dt=.5/N**2
    t1 = .006#
    tmax = time + t1
    #nmax= int(round(t1/dt))  
    #print (nmax)
    Cour=.62
    M =.9#.1#.5
    g1 =.5 #between .1 and 1.0
    c1=14.0
    c2=11.0
    alpha= np.pi#
    x_o=-2.0#2
    y_o=-1.0#
    g=10.0
    x = (2*pi/N)*np.linspace(-N/2,N/2,N)#
#np.cos((pi*arange(0,N))/N); 
    dx=np.abs(x[1]-x[0])
    dy=dx
    y = x
    [xx,yy] = np.meshgrid(x,y)    
    f = lambda t,xx,yy: -c2*((xx-x_o-M*t*np.cos(alpha))**2+(yy-y_o-M*t*np.sin(alpha))**2)  
    H0 = 1 - (c1**2/(4*c2*g1))*np.exp(2*f(time,xx,yy))
    U0 = M*np.cos(alpha)+c1*(yy-y_o-M*time*np.sin(alpha))*np.exp(f(time,xx,yy))    # 12/8 i changed the tmax to t
    V0 = M*np.sin(alpha)-c1*(xx-x_o-M*time*np.cos(alpha))*np.exp(f(time,xx,yy))
    H0_max = 1 - (c1**2/(4*c2*g1))*np.exp(2*f(tmax,xx,yy))  
    U0_max = M*np.cos(alpha)+c1*(yy-y_o-M*tmax*np.sin(alpha))*np.exp(f(tmax,xx,yy))    # 12/8 i changed the tmax to t
    V0_max = M*np.sin(alpha)-c1*(xx-x_o-M*tmax*np.cos(alpha))*np.exp(f(tmax,xx,yy))  
    
    
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
    dt = (Cour*dx)/(np.abs(max_valx+max_valy))
    nmax = int(tmax/(dt))
    
    
       
    U, dt=spectral_del(nmax,dt, N, H0,U0,V0,time,tmax,Cour)
    linf[i]=np.linalg.norm(dx*np.abs((real(U[:,:,0]) - real(H0_max))), ord=np.inf)
    NN[i]=dx
    
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
sol_plot = axes.pcolor(real(H0), cmap=plt.get_cmap('RdBu_r'))
#ax.plot_surface(X,Y,real(H0_two), cmap=plt.get_cmap('RdBu_r'))
cbar = fig.colorbar(sol_plot)  
axes.set_title("h actual t = %f"%(time)) 
plt.show() 

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
sol_plot = axes.pcolor(real(H0_max), cmap=plt.get_cmap('RdBu_r'))
#ax.plot_surface(X,Y,real(H0_two), cmap=plt.get_cmap('RdBu_r'))
cbar = fig.colorbar(sol_plot)  
axes.set_title("h actual t = %f"%(tmax)) 
plt.show()
    
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
sol_plot = axes.pcolor(U[:,:,0], cmap=plt.get_cmap('RdBu_r'))
#ax.plot_surface(X,Y,real(H0_two), cmap=plt.get_cmap('RdBu_r'))
cbar = fig.colorbar(sol_plot)  
axes.set_title("h computed Runge-Kutta t = %f"%(tmax)) 
plt.show() 

fig = plt.figure()
ax = Axes3D(fig)
z = real(U[:,:,0])
y = x
ax.plot_surface(xx,yy,z)
ax.set_title('3D line plot')
plt.show()

             
U1=U[:,:,1]/U[:,:,0]
V1=U[:,:,2]/U[:,:,0]     
fig0, ax0 = plt.subplots()
strm = ax0.streamplot(xx, yy, U1, V1, color=U1, linewidth=2, cmap=plt.cm.autumn)
fig0.colorbar(strm.lines)
ax0.set_title("Computed Runge-Kutta Streamlines at t = %f"%(tmax))
plt.show()


     
fig0, ax0 = plt.subplots()
strm = ax0.streamplot(xx, yy, U0_max, V0_max, color=U0_max, linewidth=2, cmap=plt.cm.autumn)
fig0.colorbar(strm.lines)
ax0.set_title("Actual Streamlines at t = %f"%(tmax))
plt.show()

#fig0, ax0 = plt.subplots()
#strm = ax0.streamplot(xx, yy, U0, V0, color=U0, linewidth=2, cmap=plt.cm.autumn)
#fig0.colorbar(strm.lines)
#ax0.set_title("Actual Streamlines at t = 0")
#plt.show()


#sns.set(font_scale=1.6)
#fig = plt.figure()
#axes = fig.add_subplot(1, 1, 1) 
#order_C = lambda dx, error, order: np.exp(np.log(error) - order * np.log(dx))   
#axes.loglog(NN, order_C(NN[0], linf[0], 1.0) * NN**1.0, 'b--', label="1st Order")
#axes.loglog(NN, order_C(NN[0], linf[0], 2.0) * NN**2.0, 'r--', label="2nd Order")       
#axes.loglog(N, linf_euler, 'bs', label="Actual Euler")
#axes.loglog(NN, linf, 'rs', label="Fourth Order Runge-Kutta")
#plt.xlabel('dt')
#plt.ylabel('Infinity norm error')
#axes.set_title("Integrating Factor Method from t = %f to t = %f"%(time,tmax))
#axes.legend(loc=4)
#plt.show()    


ni=np.array(ni)

sns.set(font_scale=1.6)
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1) 
order_C = lambda ni, error, order: np.exp(np.log(error) + order * np.log(ni))    
axes.loglog(ni, order_C(ni[0], linf[0], 1.0) * ni**-1.0, 'b--', label="1st Order")
axes.loglog(ni, order_C(ni[0], linf[0], 2.0) * ni**-2.0, 'r--', label="2nd Order")
axes.loglog(ni, order_C(ni[0], linf[0], 3.0) * ni**-3.0, 'g--', label="3rd Order")        
axes.loglog(ni, linf, 'rs', label="4th Order Runge-Kutta")
plt.xlabel('N')
plt.ylabel('Infinity Norm error')
axes.set_title("Translating Vortex: Integrating Factor Method  from t = %f to t = %f"%(time,tmax))
axes.legend(loc=1)
axes.legend(loc='left')
plt.show()
