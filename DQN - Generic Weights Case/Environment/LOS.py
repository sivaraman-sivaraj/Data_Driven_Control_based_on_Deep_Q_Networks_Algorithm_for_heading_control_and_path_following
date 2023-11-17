import numpy as np

def los_mmg_normalizer(theta):
    
    pivot  = np.sign(theta)
    if pivot >= 0:
        theta  = theta % (2*np.pi)
    else:
        theta  = theta % (-2*np.pi)
    
    if theta > 0 :
        if 0 < theta <= np.pi:
            theta_new  = theta 
        elif theta > np.pi:
            theta_new  = theta - (2*np.pi) 
            
    elif theta < 0:
        if 0 > theta > -np.pi:
            theta_new = theta 
        elif theta < -np.pi:
            theta_new = theta + (2*np.pi) 
    elif theta == 0:
        theta_new = 0
    else : 
        theta_new = theta 
    return theta_new

def goal_setter(H,SP):
    """
    Parameters
    ----------
    H : [last quadrant, last waypoint, HE_old,Goal_old]
    SP : Separated waypoints

    Returns
    -------
    Goal point corresponding to lastly tracked waypoint

    """
    No_G_A  = 7             # Number of goal points ahead
    No_Q    = len(SP)-1     # Total number of Qudrants
    Qi      = H[0]          # Traced Quadrant Index
    Pi      = H[1]          # Traced Point Index
    l1      = len(SP[Qi])   # Total number of waypoints in current quadrant
    Rf_P    = l1 - (Pi+1)   # Remaining untraced points in the current quadrant
    
    if Rf_P >= No_G_A:
        Goal = SP[Qi][Pi+No_G_A]
        
    else:
        done = True
        Rf_P1  = No_G_A - Rf_P #it should range from 0 to 9
        while done:
            if Qi == No_Q:
                Goal = SP[-1][-1]
                done = False
            else:
                if len(SP[Qi+1]) >= Rf_P1:
                    Goal = SP[Qi+1][Rf_P1-1]
                    done = False
                else:
                    Rf_P1 = Rf_P1-len(SP[Qi+1])
                    Qi    += 1
    return Goal
            
def nearest_point(x,y,SP):
    """
    Parameters
    ----------
    x,y : spatial position of the agent
    SP  : separated points in the prior quadrant

    Returns
    -------
    nearest waypoints index

    """
    D              = dict()
    error_distance = list()                         # calculating the euclidian distance of all Separated Points
    for i in range(len(SP)):
       er_temp         = np.sqrt(((SP[i][0]-x)**2)+((SP[i][1]-y)**2))
       error_distance.append(er_temp)
       D[str(er_temp)] = i

    sorted_distance = sorted(error_distance)    # arranging the points in ascending order
    k               = D[str(sorted_distance[0])] 
    return k                                    # point index

def get_y_e_HE(ip,wp_k,wp_k_1):
    """
    Parameters
    ----------
    current_state :     [u,v,r,x,y,psi,delta,t]- position of ship
    wp_k          :     (x_k,y_k)              - K_th way point  
    wp_k_1        :     (x_k+1,y_k+1)          - K+1_th way point 
    
    Returns
    -------
    cross track error, Heading Angle Error, Desired Heading Angle

    """
    ###############################################
    ## Horizontal path tangential angle/ gamma  ###
    ###############################################
    del_x = wp_k_1[0]-wp_k[0]
    del_y = wp_k_1[1]-wp_k[1]
    g_p = np.arctan2(del_y, del_x)
    #########################################
    ###cross track error calculation (CTE) ##
    #########################################
    y_e     = -(ip[3]-wp_k[0])*np.sin(g_p) + (ip[4]-wp_k[1])*np.cos(g_p)  # Equation 24
    #############################
    ## finding the del_h value ##
    #############################
    lbp            = 7                  # Length between perpendicular
    delta_h        = 2*lbp              # look ahead distance
    ##########################################
    ## Calculation of desired heading angle ##
    ##########################################
    beta           = np.arctan2(-ip[1],ip[0])               # drift angle
    psi_d          = g_p + np.arctan2(-y_e,delta_h) - beta # Desired Heading angle # equation 29
    
    psi_a   = los_mmg_normalizer(ip[5])
    HE      = psi_d - psi_a
    
    if abs(HE) > np.pi:
        theta1  = np.pi  - abs(psi_a)
        theta2  = np.pi  - abs(psi_d)
        theta   = theta1 + theta2
        HE_     =  -np.sign(HE) * theta
        HE      = HE_
    
    return y_e, HE



def activate(ip,wpA,H):
    """
    Parameters
    ----------
    ip      : MMG model input state
    wpA     : waypoints Analysis report
                [separated path reward points in prior order B[1], Quadrant Sequence B[0],
                 Starting Quadrant A[0]]
    H       : History of the points already used [Quadrant,waypoint index,last heading error]

    Returns
    -------
    cross track error, Heading Angle Error,History

    """
    
    S_prp   = wpA[1][1]     # Separated waypoint
    QS      = wpA[1][0]     # Quadrant Sequence
    Goal    = goal_setter(H, S_prp)
    #############################################
    ######## Choosing the best way points #######
    #############################################
    SP        = S_prp[H[0]]
    wp_near   = nearest_point(ip[3],ip[4], SP) # nearest waypoint index
    
    End_flag = False                           # ensure that the last waypoint
    if H[0] == len(QS)-1 and wp_near == len(S_prp[-1]) -1:
        End_flag = True 
    
    if End_flag == True:
        try:
            wp_k, wp_k_1 =  S_prp[-1][-2], S_prp[-1][-1]
        except:
            wp_k, wp_k_1 =  S_prp[-2][-1], S_prp[-1][-1]
    
    elif End_flag == False:
        if wp_near == len(SP)-1:
            wp_k, wp_k_1 =  S_prp[H[0]][H[1]],S_prp[H[0]+1][0]
            
        elif wp_near >= H[1] and wp_near < len(SP)-1:
            wp_k, wp_k_1 =  S_prp[H[0]][wp_near],S_prp[H[0]][wp_near+1]
            
        elif wp_near < H[1] :
            wp_k, wp_k_1 =  S_prp[H[0]][wp_near],S_prp[H[0]][H[1]]
    
    
    ###########################################
    ##### Asserting the Final Point ###########
    ###########################################
    if H[0] >= len(QS) -1 and H[1] >= len(S_prp[-1]) - 1:
        wp_k        = S_prp[-1][-1]
        wp_k_1      = [wp_k[0]+0.001,wp_k[1]+0.001]
    
    #############################################
    ###### Calculating the CTE and HE ###########
    #############################################
    y_e, HE         =  get_y_e_HE(ip, wp_k, wp_k_1)
    #############################################
    ########## Updating  the Memory #############
    #############################################
    if End_flag == False:
            
        if wp_near == len(SP)-1:
            H       = [H[0]+1,0,Goal] 
       
        elif wp_near >= H[1] and wp_near < len(SP)-1:
            H       = [H[0],wp_near+1,Goal] 
        
        elif wp_near < H[1]:
            H       = [H[0],H[1],Goal]
            
    elif End_flag == True:
        H = [H[0],len(S_prp[-1])-1,Goal]
    
    return y_e, HE, H


#########################################
############## To Check #################
#########################################
# import matplotlib.pyplot as plt
# import waypoints
# import wp_analysis

# # wp,x,y,L = waypoints.straight_line(150,45)
# wp,x,y,L   = waypoints.Fibbanaci_Trajectory(25) #
# # wp,x,y,L   = waypoints.cardioid(35)
# wpA     = wp_analysis.activate(wp)
# H  = [0,0,0,0,[45,45]]

# R = []
# for i in range(100):
#     print(i)
#     ip   = [0,0,0,wp[i-2][0],wp[i-2][1],0,0,0]
#     op   = [0,0,0,wp[i-1][0],wp[i-1][1],0,0,0]
#     y_e, HE, H  = activate(ip,wpA,H)
#     print(H)
    
#     R.append(HE)
#     # R.append(HE)
    
# plt.plot(R)
# # plt.ylim(-10,110)
# print(len(wpA[1][1][-1]))
########################################
######### End ##########################
########################################










