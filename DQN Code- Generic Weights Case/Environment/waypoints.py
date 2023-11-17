import numpy as np
import matplotlib.pyplot as plt
##################################################
################ Sub Functions ###################
##################################################
def Distance(A,B):
    D = np.sqrt(np.square(A[0]-B[0]) + np.square(A[1]-B[1]))
    return D

def interpretval(P1,P2,intv):
    a,b    = np.array(P1), np.array(P2)
    DV_    = b-a
    DV     = DV_/np.linalg.norm(DV_)
    l,m    = DV[0],DV[1]
    newpt  = [intv*l+P1[0],intv*m+P1[1]]
    return newpt

def Eq_Distance_maker(wp,lbp):
    WP_es = [wp[0]]
    
    current_pnt = wp[0]
    index_first = 1
    
    k           = 0
    
    Tot_dis     = 0
    for ii in range(len(wp)-1):
        temp     = Distance(wp[ii],wp[ii+1])
        Tot_dis += temp
        
    Int_Ds      = 2*lbp 
    NP          = int(Tot_dis / Int_Ds)
    
    for k in range(NP):
        New_presence   = []
        dis_sum   = 0
        pt_now    = current_pnt
        kk        = 0
        pt_target = wp[index_first]
        remainder = Int_Ds
        while len(New_presence) == 0:
            dis_temp = Distance(pt_now,pt_target)
            dis_sum  += dis_temp
            if dis_sum >= Int_Ds:
                newpt  = interpretval(pt_now,pt_target,remainder)
                New_presence.append(newpt)
            else:
                remainder -= dis_temp
                pt_now     = pt_target
                kk        += 1
                if index_first + kk > len(wp):
                    newpt  = wp[-1]
                    New_presence.append(newpt)
                else:
                    pt_target = wp[index_first+kk]
        WP_es.append(newpt)
        current_pnt   = newpt
        index_first   += kk
    return WP_es

##################################################
##################################################
##################################################

##################################################
################### Spiral #######################
##################################################

def spiral(pivot):
    wp = list()
    X,Y = list(),list()
    ############################   
    ######### first arc ########
    ############################
    r1 = pivot #(pivot,0)
    for i in range(0,int(2 * pivot)):
        i = 0.5* i
        y = np.sqrt((r1**2)-((r1-i)**2))
        Y.append(i)
        X.append(round(y,1))
        wp.append([round(y,1),i])
        
    for i in range(int(pivot)+1):
        y = np.sqrt((r1**2)-((i)**2))
        Y.append(r1+i)
        X.append(round(y,1))
        wp.append([round(y,1),r1+i])
    ############################  
    ######### second arc #######
    ############################
    r2 = 2 * r1
    for i in range(1,r2+1):
        y = np.sqrt((r2**2)-((i)**2))
        Y.append(round(y,1))
        X.append(-i)
        wp.append([-i,round(y,1)])
        
    for i in range(r2+1):
        y = np.sqrt((r2**2)-((i)**2))
        Y.append(-i)
        X.append(-round(y,1))
        wp.append([-round(y,1),-i])
    #################################
    ########## third  arc ###########
    #################################
    r3 = 1.5 * r2
    for i in range(1,int(r3+1)):
        x = np.sqrt((r3**2)-((i)**2))
        Y.append(round(-x+r2-r1,1))
        X.append(i)
        wp.append([i,round(-x+r2-r1,1)])
        
    for i in range(int(r3+1)):
        y = np.sqrt((r3**2)-((i)**2))
        Y.append(i+r2-r1)
        X.append(round(y,1))
        wp.append([round(y,1),i+r2-r1])
        
    
    # #################################
    # ######### Arc Length  ###########
    # #################################
    L = 0
    L += np.pi * r1
    L +=  np.pi * r2
    L +=  np.pi * r3
    S_wp = Eq_Distance_maker(wp,7)
    return S_wp,X,Y,L

##################################################
################## Straight Line #################
##################################################

def straight_line(inertial_frame_limit,theta):
        """
        Parameters
        ----------
        inertial_frame_limit : required way points range
        
        Returns
        -------
        wp : waypoints
        -------
        Warning:
            Heading Angle should be in a range of (-90) to (90) degree
        """
        ### Assertion ###
        if theta > 180 :
            theta = theta - 360
        elif theta < -180:
            theta = theta + 360
        
        #################################
        #### path reward Declaration ####
        #################################
        a = (theta/180) * np.pi # radian mode
        wp = list() # path reward points
        # starting_point = [0,0] #starting point of the ship
        # prp.append(starting_point)
        
        if -45 <= theta <= 45:
            for e in range(inertial_frame_limit):
                y_t = e*(np.tan(a))
                if abs(y_t) < abs(inertial_frame_limit):
                    temp = [e,y_t]
                    wp.append(temp)
        elif -135 >= theta >= -180 or 135 <= theta <= 180:
            for e in range(inertial_frame_limit):
                y_t = -e*(np.tan(a))
                if abs(y_t) < inertial_frame_limit:
                    if e == 0:
                        temp = [e,-y_t]
                    else:
                        temp = [-e,y_t]
                    wp.append(temp)
                        
        elif 45 < theta < 135 :
            for e in range(inertial_frame_limit):
                x_t = -e/(np.tan(a))
                if abs(x_t) < inertial_frame_limit:
                    temp = [-x_t,e]
                    wp.append(temp)
        elif -45 > theta > -135 :
            for e in range(inertial_frame_limit):
                x_t = -e/(np.tan(a))
                if abs(x_t) < inertial_frame_limit:
                    temp = [x_t,-e]
                    wp.append(temp)
        
        
        ############################
        #### path reward end #######
        ############################ 
        x,y = list(),list()
        for i in range(len(wp)):
            x.append(wp[i][0])
            y.append(wp[i][1])
       
        x = x[::]
        y = y[::]
        
        ### Length of Trajectory ###
        L = np.sqrt((wp[-1][0]**2) + (wp[-1][1]**2))
        S_wp = Eq_Distance_maker(wp,7)
        return S_wp,x,y,L
    

##################################################
########### Fibbanaci_Trajectory #################
##################################################

def Fibbanaci_Trajectory(pivot):
    wp = list()
    X,Y = list(),list()
    ############################   
    ###### first quadrant ######
    ############################
    r1 = pivot #(pivot,0)
    for i in range(0,int(2 * pivot)):
        i = 0.5* i
        x = np.sqrt((r1**2)-((r1-i)**2))
        X.append(round(x,1))
        Y.append(i)
        wp.append([round(x,1),i])
    
    ############################  
    ####### second quadrant ####
    ############################
    r2 = 2*r1
    for i in range(r2):
        x = np.sqrt((r2**2)-((i)**2))
        X.append(round(x-r1,1))
        Y.append(r1+i)
        wp.append([round(x-r1,1),r1+i])
    ############################  
    ####### third quadrant ####
    ############################
    r3 = 2 * r2
    for i in range(r3):
        y = np.sqrt((r3**2)-((i)**2))
        X.append(-i-r1)
        Y.append(round(y-r1,1))
        wp.append([-i-r1,round(y-r1,1)])
    #################################
    #######  fourth quadrant ########
    #################################
    r4 = 2 * r3
    for i in range(r4):
        x = np.sqrt((r4**2)-((i)**2))
        X.append(round(-x+r2+r1,1))
        Y.append(-i-r1)
        wp.append([round(-x+r2+r1,1),-i-r1,])
    # #################################
    # #######  fifth quadrant ########
    # #################################
    r5 = 2 * r4
    for i in range(r5):
        y = np.sqrt((r5**2)-((i)**2))
        X.append(i+r2+r1)
        Y.append(round(-y+r4-r1,1))
        wp.append([i+r2+r1,round(-y+r4-r1,1)])
        
    #################################
    ######### Arc Length  ###########
    #################################
    L = 0
    L += 0.5*np.pi * r1
    L += 0.5 * np.pi * r2
    L += 0.5 * np.pi * r3
    L += 0.5 * np.pi * r4
    L += 0.5 * np.pi * r5
    S_wp = Eq_Distance_maker(wp,7)
    return S_wp,X,Y,L

##################################################
################ Cardioid ########################
##################################################

    
def cardioid(a):
    X,Y = [],[]
    wp = []
    for i in range(0,-180,-1):
        x = 2*a*(1-np.cos(np.deg2rad(i)))*np.cos(np.deg2rad(i))
        y = 2*a*(1-np.cos(np.deg2rad(i)))*np.sin(np.deg2rad(i))
        X.append(round(x,1))
        Y.append(round(y,1))
        wp.append([round(x,1),round(y,1)])
        
    for i in range(180,0,-1):
        x = 2*a*(1-np.cos(np.deg2rad(i)))*np.cos(np.deg2rad(i))
        y = 2*a*(1-np.cos(np.deg2rad(i)))*np.sin(np.deg2rad(i))
        X.append(round(x,1))
        Y.append(round(y,1))
        wp.append([round(x,1),round(y,1)])
        
    wp_new = [] # to change the clockwise / counter cloakwise
    for j in range(len(wp)):
        temp = wp[j]
        wp_new.append(temp)
    
    L = 8*a # length of the cardioid formula
    S_wp = Eq_Distance_maker(wp,7)
    return S_wp,X,Y,L

##################################################
################ Parametric ######################
##################################################

def parametric(a):
    X,Y = [],[]
    wp = []
    for i in range(0,-180,-1):
        x = 2*a*(1-np.cos(np.deg2rad(i)))*np.cos(np.deg2rad(i))*np.cos(np.deg2rad(i))
        y = 2*a*(1-np.cos(np.deg2rad(i)))*np.sin(np.deg2rad(i))
        X.append(round(x,1))
        Y.append(round(y,1))
        wp.append([round(x,1),round(y,1)])
        
    for i in range(180,0,-1):
        x = 2*a*(1-np.cos(np.deg2rad(i)))*np.cos(np.deg2rad(i))*np.cos(np.deg2rad(i))
        y = 2*a*(1-np.cos(np.deg2rad(i)))*np.sin(np.deg2rad(i))
        X.append(round(x,1))
        Y.append(round(y,1))
        wp.append([round(x,1),round(y,1)])
        
    wp_new = [] # to change to counter clockwise 
    for j in range(len(wp)):
        temp = wp[-j]
        wp_new.append(temp)
    
    L = 0
    S_wp = Eq_Distance_maker(wp,7)
    return S_wp,X,Y,L

##################################################
################### Curve ########################
##################################################

def curve(GFL):
    wp,x,y = [],[],[]
    for i in range(GFL):
        temp = 0.003 * (i**2)
        x.append(i)
        y.append(temp)
        wp.append([i,temp])
    from scipy.integrate import quad
    
    def integrand(x):
        return (1 + (0.006*x)**1.8)**0.5
    L = quad(integrand, 0, GFL)
    S_wp = Eq_Distance_maker(wp,7)
   
    return S_wp,x,y,L

##################################################
############### S-shaped Curve ###################
##################################################

def Arc_spline():
    wp        = []
    X,Y       = [],[]
    L         = 0 
        
    def f(x):
        num = 250
        den = 1 + np.exp(-0.03*x)
        return num/den
    
    for i in range(-200,200):
        temp  = f(i)
        wp.append([i+200,temp]) 
        X.append(i+200)
        Y.append(temp)
    S_wp = Eq_Distance_maker(wp,7)
    return S_wp,X,Y,L

##################################################
#################### Ellipse #####################
##################################################

def Ellipse(a,b):
    wp    = []
    X,Y   = [],[]
    L     = 0 
    def f(x,a,b):
        ia = b**2
        ib = (x**2)/(a**2)
        y2 = ia*(1-ib)
        y  = np.sqrt(y2)
        return y
    L = a
    for i in range(-L,L):
        temp = f(i,a,b)
        X.append(i+a)
        Y.append(temp)
        wp.append([i+a,temp])
    for i in range(-L,L):
        temp = f(-i,a,b)
        X.append(-i+a)
        Y.append(-temp)
        wp.append([-i+a,-temp])
    
    X.append(X[0])
    Y.append(Y[0])
    wp.append(wp[0])
    #####################
    #####################
    ec1 = 3*(a+b)
    ec2 = np.sqrt(((3*a) + b)*(a + (3*b)))
    L   = np.pi*(ec1 - ec2)
    S_wp = Eq_Distance_maker(wp,7)
    return S_wp,X,Y,L

##################################################
##################################################
##################################################
################################
###### To evaluate #############
################################
import wp_analysis

# wp,X,Y,L = straight_line(300,135)
# wp,X,Y,L = spiral(35)
# wp,X,Y,L = Fibbanaci_Trajectory(10)
# wp,X,Y,L = cardioid(40)
# wp,X,Y,L = parametric(30)
# wp,X,Y,L = curve(70)
# wp,X,Y,L = Arc_spline()

# gx,gy = [],[]
# A,B,C,D = wp_analysis.activate(wp)

# xx,yy = [],[]
# for i in range(len(wp)):
#     xx.append(wp[i][0])
#     yy.append(wp[i][1])

# plt.figure(figsize=(9,6))
# plt.plot(X,Y,'y',label = "Requied Trajectory")
# plt.scatter(gx,gy,marker="s",label= "Goals")
# plt.scatter(xx,yy,color = "purple",label = "waypoints")
# plt.axvline(x=0,color='green',alpha = 0.5)
# plt.axhline(y=0,color='green',alpha = 0.5)
# plt.title("Straight Line Trajectory")
# plt.title("Spiral Trajectory")
# plt.title("Fibbonacci Trajectory")
# plt.title("Cardioid Trajectory")
# plt.title("Parametric Curve")
# plt.ylabel("Transfer(in meters)")
# plt.xlabel("Advance (in meters)")
# # plt.xlim(-300,300)
# # plt.ylim(-300,300)
# plt.grid()
# plt.legend()
# plt.show()
# print("The lemgth of trajectory is :",  L)
##################################
###### evaluation end ############
##################################

