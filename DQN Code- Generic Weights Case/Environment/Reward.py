import numpy as np

def Goal_monitor(ip,op,G):
    """
    Returns
    -------
    flag : when agent moves towards the goal, flag is True

    """
    Flag = False
    x_d0 = np.square(G[0] - ip[3])
    y_d0 = np.square(G[1] - ip[4])
    D0 = np.sqrt(x_d0+y_d0)
    
    x_d1 = np.square(G[0] - op[3])
    y_d1 = np.square(G[1] - op[4])
    D1 = np.sqrt(x_d1 + y_d1)
    
    if D0 >= D1 :
        Flag = True 
   
    return Flag

def CTE(y_e):
    if abs(y_e) <= 1:
                R =100
    elif 1 < abs(y_e) <= 2:
        R = 20
    else:
        c0 = abs(y_e)/49
        c1 = 1 - c0
        R  = 20 * c1 
    return R


def get(ip,op,HE,y_e,G):
    """
    Parameters
    ----------
    ip          : input state
    op          : output state
    y_e         : cross track error
    y_e_old     : previous cross track error
    HE          : heading error
    HE_old      : previous heading error
    G           : goal
    T_i         : Tolerence index for 
    i_episode   : episode number

    Returns
    -------
    Rf : Reward

    """
    Flag_G      = Goal_monitor(ip,op,G)
    
    if Flag_G == True:
        Id1 = 1- (abs(HE)/np.deg2rad(90))
        Id2 = CTE(y_e)
        R   = Id1 * Id2
        R   += 1
        
    else:
        theta = abs(HE) - np.deg2rad(90) 
        R     = 1 - (theta/np.deg2rad(90))
        
    
    # if abs(op[5]) >= (1.7*np.pi): #270 degree, for the case of heading action only
    #     R = -0.5
    if abs(y_e) > 56:
        R = -0.5
    ################################
    ########### Assertion ##########
    ################################
   
    Rf = np.array([R])
        
    return Rf


########################################
############# To check #################
########################################
# ip = [7.75,0,0,15,15,0]
# op = [7.75,0,0,16,16,0]
# G = [300,300]
# ss = get(ip,op,0,0,G)
# print(ss)
########################################
########################################
