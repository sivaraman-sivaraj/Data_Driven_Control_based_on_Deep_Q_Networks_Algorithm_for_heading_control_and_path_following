import numpy as np
import matplotlib.pyplot as plt
import os,shutil


def create_op_folder():
    """
    Returns
    -------
    Folder Creation for training plots 

    """
    R           = "Results"
    F1          = "0. Weights"
    F2          = "1. Reward_Plots"
    F3          = "2. Error_Plots"
    F4          = "3. Others"
    
    parent = os.getcwd()
    path0 = os.path.join(parent,R)
    path1 = os.path.join(path0, F1)
    path2 = os.path.join(path0, F2)
    path3 = os.path.join(path0, F3)
    path4 = os.path.join(path0, F4)
    
    try :
        os.mkdir(path0)
    except FileExistsError:
        shutil.rmtree(path0)
        os.mkdir(path0)
    
    try :
        os.mkdir(path1)
    except FileExistsError:
        os.rmdir(path1)
        os.mkdir(path1)
        
    try :
        os.mkdir(path2)
    except FileExistsError:
        os.rmdir(path2)
        os.mkdir(path2)
        
    try :
        os.mkdir(path3)
    except FileExistsError:
        os.rmdir(path3)
        os.mkdir(path3)
        
    try :
        os.mkdir(path4)
    except FileExistsError:
        os.rmdir(path4)
        os.mkdir(path4)

##########################################
########### Image Plotting ###############
##########################################

def plot_result1(NoEpi, Cumulative_reward,path,N="End"):
    plt.figure(figsize=(9,12))
    #############################
    plt.subplot(2,1,1)
    N = len(Cumulative_reward)
    running_avg1 = np.empty(N)
    for t in range(N):
            running_avg1[t] = np.mean(Cumulative_reward[max(0, t-100):(t+1)])
    plt.plot(Cumulative_reward,label="Cumulative Reward",color="r",alpha=0.2)
    plt.plot(running_avg1,color='r',label="Running Average")
    plt.title("DQN Training Resuts : Cumulative Rewards & Episode Durations ")
    plt.xlabel("No of Episode")
    plt.ylabel("Reward Unit")
    plt.legend(loc="best")
    plt.grid()
    ##############################
    plt.subplot(2,1,2)
    N = len(NoEpi)
    running_avg2 = np.empty(N)
    for t in range(N):
            running_avg2[t] = np.mean(NoEpi[max(0, t-100):(t+1)])
    plt.plot(NoEpi,color="g",label = "Episode Durations",alpha=0.2)
    plt.plot(running_avg2,color='g',label="Running Average" )
    plt.xlabel("No of Episode")
    plt.ylabel("Length of Episodes")
    plt.legend(loc="best")
    plt.grid()
    plt.legend(loc="best")
    plt.savefig(os.path.join(path,"pic1_epi_"+str(N)+".jpg" ),dpi=480)
    plt.close()
    
    
def plot_result2(HEs,MSE,path,N="End"):
    plt.figure(figsize=(9,12))
    ##############################
    plt.subplot(2,1,1)
    N = len(MSE)
    running_avg1 = np.empty(N)
    for t in range(N):
            running_avg1[t] = np.mean(MSE[max(0, t-100):(t+1)])
    plt.plot(MSE,color="m",label="Mean Square Error",alpha=0.2)
    plt.plot(running_avg1,color='m',label="Running Average")
    plt.title(" DQN Training Resuts : Mean Squared Loss , Heading Error ")
    plt.xlabel("No of Episode")
    plt.ylabel("MSE")
    plt.legend(loc="best")
    plt.grid()
    
    
    plt.subplot(2,1,2)
    N = len(HEs)
    running_avg2 = np.empty(N)
    for t in range(N):
            running_avg2[t] = np.mean(HEs[max(0, t-100):(t+1)])
    plt.plot(HEs,color="b",label = "Cumulative Heading Error",alpha=0.2)
    plt.plot(running_avg2,color='b',label="Running Average" )
    plt.xlabel("No of Episode")
    plt.ylabel("Heading Error in degree")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(os.path.join(path,"pic2_epi_"+str(N)+".jpg" ),dpi=480)
    plt.close()#plt.show()
