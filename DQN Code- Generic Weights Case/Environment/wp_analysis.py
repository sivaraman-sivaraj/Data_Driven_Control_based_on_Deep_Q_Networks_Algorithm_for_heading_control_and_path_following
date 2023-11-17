import numpy as np

def Quadrant_position(A):
    """
    Parameters
    ----------
    A : (x,y) position of the ship

    Returns
    -------
    op : corresponding quadrant

    """
    x,y = A[0],A[1]
    op = 0
    if int(x)>0 and int(y)>0:
        op = 1
    elif int(x) < 0 and int(y) >0:
        op = 2
    elif int(x)<0 and int(y) < 0 :
        op = 3
    elif int(x)>0 and int(y)<0:
        op = 4
    ############################
    elif int(y) == 0 and int(x) > 0:
        op = 5
    elif int(y) == 0 and int(x)<0:
        op = 6
    elif int(x) == 0 and int(y) > 0:
        op = 7
    elif int(x) == 0 and int(y) < 0:
        op = 8
    else:
        op = 0
    return op

################################################
####### Step Counter algorithm #################
################################################

def Euclidean_Distance(A,B):
    X = np.square(A[0]-B[0])
    Y = np.square(A[1]-B[1])
    D = np.sqrt(X+Y)
    return D

def Tolerance_predicter(wp):
    """
    Parameters
    ----------
    wp : waypoints

    Returns
    -------
    tolerence steps
    
    Note:
    -------
    From the study on PID controller and based on starting  angle, we need to give tolerence
    steps. [maxium step for 180 degree heading is 110]

    """
    x        = wp[1][0] - wp[0][0]
    y        = wp[1][1] - wp[0][0]
    st_angle = np.arctan2(y,x)
    pivot    = abs(np.rad2deg(st_angle))/180.0
    T        = round(pivot*110)
    return int(T)

def Steps_Counter(wp):
    """
    Parameters
    ----------
    wp : waypoints

    Returns
    -------
    Horizon steps, Tolerence_index

    """
    L    = 0
    for i in range(len(wp)-1):
        temp = Euclidean_Distance(wp[i], wp[i+1])
        L    += temp 
    Tol  = Tolerance_predicter(wp)
    L    += Tol 
    return int(L),Tol+10


###########################################
############# Conglomeration ##############
###########################################
def activate(wp):
    """

    Parameters
    ----------
    wp :  waypoints
    N  :  No of episodes
    Returns
    -------
    Quadrant Analysis Report: (as list)
       A =  [1. Starting Quadrant 
        2. Major Quadrant 
        3. Quadrant 1 points 
        4. Quadrant 2 points
        5. Quadrant 3 points
        6. Quadrant 4 points 
        7. X_axis_positive points
        8. X_axis_negative points
        9. Y_axis_positive points
        10. Y_axis_negative points]
        
        B = [1.Quadrant sequence
         2.separated points by sequence wise]
        
        C = [goals,steps,number of episodes]
        D = reinforcement index for starting position

    """
    Starting_quadrant = 0
    ##########################################
    #### Finding the  Quadrant Sequence ######
    ##########################################
    Quad_sequence = [0]
    for j in range(len(wp)):
        pivot = Quadrant_position(wp[j])
        if pivot != Quad_sequence[-1]:
            Quad_sequence.append(pivot)
    ###########################################
    ########### Points Separation #############
    ###########################################
    separated_points = list()
    
    for k in range(len(Quad_sequence)):
        separated_points.append([])
    
    split_pivot     = 0
    quad_pivot      = 0  
    for ii in range(len(wp[split_pivot:])):
        pivot = Quadrant_position(wp[ii])
        
        if pivot == Quad_sequence[quad_pivot]:
            separated_points[quad_pivot].append(wp[ii])
        else:
            separated_points[quad_pivot+1].append(wp[ii])
            quad_pivot += 1
            split_pivot = ii
    
    ##########################
    Q1,Q2,Q3,Q4 = 0,0,0,0
    X_axis_positive, X_axis_negative = 0,0
    Y_axis_positive, Y_axis_negative = 0,0
    ##########################
    for i in range(len(wp)):
        
        x,y = wp[i][0],wp[i][1]
        
        ################################################
        #### Finding the inital waypoints's quadrant ###
        ################################################
        
        if i == 20: # As per need, we can change the value 20
            Starting_quadrant = np.argmax([Q1,Q2,Q3,Q4,
                                           X_axis_positive, X_axis_negative,
                                           Y_axis_positive, Y_axis_negative]) + 1
        ########################################
        ### Assigning Points to the Quadrant ###
        ########################################
        if int(x)>0 and int(y)>0:
            Q1 += 1
        elif int(x) < 0 and int(y) >0:
            Q2 += 1
        elif int(x)<0 and int(y) < 0 :
            Q3 += 1
        elif int(x)>0 and int(y)<0:
            Q4 += 1
        ############################
        elif int(y) == 0 and int(x) > 0:
            X_axis_positive += 1
        elif int(y) == 0 and int(x)<0:
            X_axis_negative += 1
        elif int(x) == 0 and int(y) > 0:
            Y_axis_positive += 1
        elif int(x) == 0 and int(y) < 0:
            Y_axis_negative += 1
        ##########################################
        ####### Waypoints major Quadrant #########
        ##########################################
        Major_Quadrant = np.argmax([Q1,Q2,Q3,Q4,X_axis_positive, X_axis_negative,
                                    Y_axis_positive, Y_axis_negative]) + 1
    ########################################
    ############ Step Counter ##############
    ########################################
    C,T_i = Steps_Counter(wp)
    
    ###### Gathering #######
    A = [Starting_quadrant,Major_Quadrant,Q1,Q2,Q3,Q4,
            X_axis_positive, X_axis_negative,
            Y_axis_positive, Y_axis_negative] 
    B = [Quad_sequence,separated_points]
    
    return [A,B,C,T_i]


###########################################
########### To Check ######################
###########################################
# import waypoints
# wp,x,y,L   = waypoints.Fibbanaci_Trajectory(25)
# wp,x,y,L   = waypoints.spiral(30)
# wp,x,y,L = waypoints.straight_line(200,0)
# wp,x,y,L   = waypoints.cardioid(25)

# A,B,C,D        = activate(wp)
# print("#############################")
# print("The Starting Quadrant        : ", A[0])
# print("The Major Quadrant           : ", A[1])
# print("Quadrant 1 points    :",A[2])
# print("Quadrant 2 points    :",A[3])
# print("Quadrant 3 points    :",A[4])
# print("Quadrant 4 points    :",A[5])
# print("#############################")
# print("X axis Points        : ",A[6]+A[7])
# print("Y axis Points        : ",A[8]+A[9])
# print("#############################")
# print("Quadrant Sequence    : ", B[0])
# print("#############################")
###########################################
################ End ######################
###########################################