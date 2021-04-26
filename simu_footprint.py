import numpy as np
import matplotlib.pyplot as plt
import cPickle
import scipy.interpolate as intp
from matplotlib import cm
##################################################################################

def getxmax(E):
	xmax= 470 + 58*np.log10(E/1e15) # estimated for proton , heitlar's model
	return xmax 


def footprint(Xmax,E,zen,azimuth,x,y):
	c0 = 0.24
	c1 = 1.e-52
	c2= -7.88
	c3 = 28.58
	c4 = 1.98
	c5 = - 2.57
	c6 = -54.9
	c7 = 0.44
	c8 = -1.24 * 1e-4
	c9 = 20.4
	c10 = 0.006
	c11 = 9.7 * 1e-5
	c12=np.array([107,-0.94,1.94*1e-3,-1.5*1e-6,4.1e-10])
	#E = 7.0 * 1e17
	#Xmax= 590
	X_atm=1030
	X =0 #116.99
	Y = 0#-134.63
	antenna_power=[]

    	D_xmax=X_atm/np.cos(zen)-Xmax
    	f1=c1*E*E
    	f2=(x-(X+c2*np.sin(azimuth)+c3))**2 + (y - (Y+c4*np.sin(azimuth)+c5))**2
    	f3= (c6 + c7*D_xmax + c8 * D_xmax*D_xmax)**2
    	f_sum= 0
    	for i in np.arange(5):
        	f_sum =+ c12[i]*D_xmax**i
    	f4=(x-(X+f_sum))**2 + (y-Y)**2
    	f5 = ( c9 + c10 * D_xmax + c11 * D_xmax*D_xmax)**2
    	P = f1 * np.exp(-f2/f3) - c0 * f1 * np.exp(-f4/f5)
    	return P



def GetUVW(pos, cx, cy, zen, az):
   relpos = pos-np.array([cx,cy,7.6])
   inc=1.1837
   B = np.array([0,np.cos(inc),-np.sin(inc)])
   v = np.array([-np.cos(az)*np.sin(zen),-np.sin(az)*np.sin(zen),-np.cos(zen)])
   #print v
   vxB = np.array([v[1]*B[2]-v[2]*B[1],v[2]*B[0]-v[0]*B[2],v[0]*B[1]-v[1]*B[0]])
   vxB = vxB/np.linalg.norm(vxB)
   vxvxB = np.array([v[1]*vxB[2]-v[2]*vxB[1],v[2]*vxB[0]-v[0]*vxB[2],v[0]*vxB[1]-v[1]*vxB[0]])
   return np.array([np.inner(vxB,relpos),np.inner(vxvxB,relpos),np.inner(v,relpos)]).T


def showerplane(zen,az):
	inc = 67.8/180.*np.pi #angle of inclination wrt magnatic north?

	B = np.array([0,np.cos(inc),-np.sin(inc)]) #LOFAR coordinates
	v = np.array([-np.cos(az)*np.sin(zen),-np.sin(az)*np.sin(zen),-np.cos(zen)])
	vxB = np.array([v[1]*B[2]-v[2]*B[1],v[2]*B[0]-v[0]*B[2],v[0]*B[1]-v[1]*B[0]])
	vxB = vxB/np.linalg.norm(vxB)
	vxvxB = np.array([v[1]*vxB[2]-v[2]*vxB[1],v[2]*vxB[0]-v[0]*vxB[2],v[0]*vxB[1]-v[1]*vxB[0]])
	x_shower=[]
	y_shower=[]
	for i in np.arange(1,21):
   		for j in np.arange(8):
      			xyz=i*10*(np.cos(j/4.0*np.pi)*vxB+np.sin(j/4.0*np.pi)*vxvxB)
			c = xyz[2]/v[2]
	 	        xyz_ground=np.array([(xyz[0]-c*v[0]), (xyz[1]-c*v[1]),7.6])
			xyz_shower=GetUVW(xyz_ground,0,0,zen,az)
			#print xyz_ground
			x_shower.append(xyz_shower[0])
			y_shower.append(xyz_shower[1])
	return x_shower,y_shower
	
	




def plot_footprint(xmax,energy,zen,azimuth,runno):
	xyz= showerplane(zen,azimuth)
	antenna_power=[]		
	for i in np.arange(len(xyz[0])):
    		antenna_power.append(footprint(xmax,energy,zen,azimuth,xyz[0][i],xyz[1][i]))
	ti = np.linspace(-400, 400, 150)
	XI, YI = np.meshgrid(ti, ti)
	#print antenna_power
	rbf_ground = intp.Rbf(xyz[0], xyz[1], np.array(antenna_power), smooth =0, function='quintic')
	ZI = rbf_ground(XI, YI)
	plt.pcolor(XI, YI, ZI,vmax=np.max(ZI), vmin=0,cmap='gray',alpha=None)
	plt.colorbar()
	#plt.savefig('testgray.png')#('images/img_{0}.png'.format(runno))
	plt.show()
	#data=plt.imread('im_test.png')
	#print data.shape



def get_footprint(xmax,energy,zen,azimuth):
	xyz= showerplane(zen,azimuth)
	antenna_power=[]		
	for i in np.arange(len(xyz[0])):
    		antenna_power.append(footprint(xmax,energy,zen,azimuth,xyz[0][i],xyz[1][i]))
	ti = np.linspace(-200,200,80)
	XI, YI = np.meshgrid(ti, ti)
	#print antenna_power
	rbf_ground = intp.Rbf(xyz[0], xyz[1], np.array(antenna_power), smooth =0, function='quintic')
	ZI = rbf_ground(XI, YI)
	
	return ZI
	
	
	
	
'''
xmax=800
E=7*1e17
zen=np.pi/4
azimuth=3*np.pi/2-np.pi/6
power=get_footprint(xmax,E,zen,azimuth)
print power
'''
#power= plot_footprint(xmax,E,zen,azimuth,2)
#print power[0]#len(power[0][np.where(power[0]>0)])
#print len(power[1][np.where(power[1]>0)])


