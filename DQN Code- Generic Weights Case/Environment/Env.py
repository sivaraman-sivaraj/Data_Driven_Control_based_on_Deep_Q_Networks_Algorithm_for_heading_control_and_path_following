import numpy as np
from   gym import Env
from   gym.utils import seeding
from gym.envs.classic_control import rendering
import MMG,LOS,Reward
import torch
import graph

colors ={"water"        : [0, 0.74, 0.8],
         "land"         : [0.3, 0, 0],
         "ship"         : [0.8,0,0],
         "trajectory"   : [0.94,0.94,0.94],
         "required_path": [0.192,1,0.192],
         "ship_engine"  : [0.17,0.17,0.17],
         "circle"       : [0.9,0.9,0.9],
         "axis"         : [0.5,0.5,0.7]}
######################################
###### MMG's Ship polygan point ######
######################################

def import_polygon(sr):
    pp = list() #Polygon points
    ########
    ########
    pp.append((29,0))
    pp.append((29,100))
    ########
    ########
    pp.append((25,115))#curve points
    pp.append((20,125))
    pp.append((15,135))
    pp.append((10,145))
    pp.append((3,158))
    ########
    ########
    pp.append((0,160))
    ########
    ########
    pp.append((-3,158))#curve points
    pp.append((-10,145))
    pp.append((-15,135))
    pp.append((-20,125))
    pp.append((-25,115))
    ########
    ########
    pp.append((-29,100))
    pp.append((-29,0))
    ########
    ########
    pp.append((-29,-80))
    pp.append((-35,-80))
    pp.append((-35,-100))
    ########
    ########
    pp.append((-25,-100))
    pp.append((-25,-110))
    pp.append((-15,-110))
    pp.append((-15,-120))
    ########
    ########
    pp.append((15,-120))
    pp.append((15,-110))
    pp.append((25,-110))
    pp.append((25,-100))
    ########
    ########
    pp.append((35,-100))
    pp.append((35,-80))
    pp.append((29,-80))
    ########
    ########
    pp.append((29,0))
    pp.append((29,0))
    
    pp_temp = np.array(pp)
    pp_rt = pp_temp/sr
    pp_rt.tolist()
    return pp_rt

#####################################
##### Environment Construction ######
##################################### 

class load(Env): # NavigationRL-v0
     metadata = {'render.modes': ['human', 'rgb_array'],
                 'video.frames_per_second': 2 }
     
     def __init__(self,wp,wpA,u,rps,W_Flag):
         
        '''
        Establishing the Environment.

        Parameters
        ----------
        wpA                 : waypoints Analysis Report
        wp                  : waypoints
        u                   : intial velocity 
        rps                 : revolution per second
        W_Flag              : Flag for Calm water or Wave
        Returns
        -------
        environment 

        '''

        self.wp             = wp
        self.wpA            = wpA
        self.u              = u
        self.rps            = rps
        self.W_Flag         = W_Flag
        ### grid size ###
        self.grid_size      = 600
        #### separated points ####
        self.S_wp           = self.wpA[1][1]
        self.T_i            = wpA[3]
        #### initial conditions ####
        self.done,self.viewer   = False, None           # see here
        self.st_x, self.st_y    = 0,0
        self.actions_set        = {'0':-35,'1':-20,'2':0,'3':20,'4':35}
        self.St_angle           = np.arctan2((self.wp[2][1]-self.wp[0][1]),(self.wp[2][0]-self.wp[0][0]))
        
        
     def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
     def reset(self):
        #######################
        ## H = [last quadrant, last waypoint,previous heading error, previous CTE,goal]
        ########################
        self.goal                             = LOS.goal_setter([0,0], self.S_wp)
        self.H                                = [0,0,self.goal]
        self.done                             = False
        self.current_state                    = torch.tensor([self.u,0,0,self.st_x, self.st_y,0,-self.St_angle,0])
        return self.current_state,self.H
    
     def step(self,C):
         self.action,self.H,self.WHA          = C[0],C[1],C[2]
         self.goal                            = self.H[2]
         self.ip                              = self.current_state.clone().detach()
         self.op                              = MMG.activate(self.ip,np.deg2rad(self.actions_set[str(self.action)]),self.rps,
                                                             self.WHA,self.W_Flag)
         self.y_e,self.HE,self.H              = LOS.activate(self.op,self.wpA,self.H)
         self.reward_a                        = Reward.get(self.ip,self.op,self.HE,self.y_e,self.goal)
         self.op[6]              = self.HE
         self.op[7]              = self.y_e
         #################################
         ###### Next State Update ########
         #################################
         self.current_state      = torch.tensor(self.op)
         #################################
         ### Epi Termination Creteria ####
         #################################
         ### @1 ### by  reward
         if abs(self.y_e) > 70 :
             self.done = True
         if abs(self.HE) > np.deg2rad(175):
             self.done = True
         ### @2 ### by final point
         self.x_d0 = np.square(self.goal[0] - self.op[3])
         self.y_d0 = np.square(self.goal[1] - self.op[4])
         self.D0 = np.sqrt(self.x_d0+ self.y_d0)
         
         if self.D0 < 2 :
             self.done     = True
         
         return self.current_state, [self.reward_a,self.HE], self.done,self.H
             
         

     def action_space_sample(self):
        n = np.random.randint(0,5)
        return n
    
     def action_space(self):
        return np.arange(0,5,1)
     
     def render(self,mode='human'):
         """
         Parameters
         ----------
         mode : The default is 'human' : .
         tp      : Trajectory Points
         
         ----------------
         description
         ----------------
         Scale : (grid_size/600) = scaling ratio
         
         """
         screen_width     = 600
         screen_height    = 600
         self.sr          = 1 # scaling coefficient
         self.tp          = [[0,0]]# if you need ship trajectory please make the travelled points at here
         self.rtn         = -self.current_state[5].item()+(np.pi/2)
         if self.viewer is None:
            
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            ###################################################
            ############## Water Surface ######################
            ###################################################
            #water surface
            water_surface = rendering.FilledPolygon([(20,20),(580,20),(580,580),(20,580)])
            self.wstrans = rendering.Transform() 
            water_surface.set_color(colors["water"][0],colors["water"][1],colors["water"][2])
            water_surface.add_attr(self.wstrans)
            self.viewer.add_geom(water_surface)
            
            ###################################################
            ############### Land Surface ######################
            ###################################################
            #left land rendering
            left_land = rendering.FilledPolygon([(0,0),
                                                 (20,0),
                                                 (20,600),
                                                 (0,600)])
            self.lltrans = rendering.Transform()#left land transform
            left_land.set_color(colors["land"][0],colors["land"][1],colors["land"][2])
            left_land.add_attr(self.lltrans)
            self.viewer.add_geom(left_land)
                
            #right land rendering
            right_land = rendering.FilledPolygon([(580,0),
                                                  (600,0),
                                                  (600,600),
                                                  (580,600)])
            self.rltrans = rendering.Transform()#right land transform
            right_land.set_color(colors["land"][0],colors["land"][1],colors["land"][2])
            right_land.add_attr(self.rltrans)
            self.viewer.add_geom(right_land)
            
            #top land rendering
            top_land = rendering.FilledPolygon([(20,580),(580,580),(580,600),(20,600)])
            self.tptrans = rendering.Transform()#top land transform
            top_land.set_color(colors["land"][0],colors["land"][1],colors["land"][2])
            top_land.add_attr(self.tptrans)
            self.viewer.add_geom(top_land)
            
            #bottom land rendering
            bottom_land = rendering.FilledPolygon([(20,0),(580,0),(580,20),(20,20)])
            self.bttrans = rendering.Transform()#right land transform
            bottom_land.set_color(colors["land"][0],colors["land"][1],colors["land"][2])
            bottom_land.add_attr(self.bttrans)
            self.viewer.add_geom(bottom_land)
            
            ###################################################
            ############### Ship Rendering ####################
            ###################################################            
            ship            = rendering.FilledPolygon(import_polygon(15)) #
            ship.set_color(colors["ship_engine"][0],colors["ship_engine"][1],colors["ship_engine"][2])
            self.shiptrans  = rendering.Transform(translation=(0,0),rotation= self.rtn)#
            ship.add_attr(self.shiptrans)
            self.viewer.add_geom(ship)
            
            self.axle = rendering.make_circle(1.5) #will chnage to 3.8
            self.axle.add_attr(self.shiptrans)
            self.axle.set_color(colors["circle"][0],colors["circle"][1],colors["circle"][2])
            self.viewer.add_geom(self.axle)
            
            self._ship_geom = ship
            
            ###################################################
            ############### Axis Rendering ####################
            ###################################################
            self.x_axis = rendering.Line((300,0), (300,600))
            self.x_axis.set_color(colors["axis"][0],colors["axis"][1],colors["axis"][2])
            self.viewer.add_geom(self.x_axis)
            ###########################################
            self.y_axis = rendering.Line((0,300), (600,300))
            self.y_axis.set_color(colors["axis"][0],colors["axis"][1],colors["axis"][2])
            self.viewer.add_geom(self.y_axis)
            ###################################################
            ############### Trajectory Rendering ##############
            ###################################################
            self.traject = rendering.make_polyline(self.tp) # for every time step, position should be updated
            self.traject.set_color(colors["trajectory"][0],colors["trajectory"][1],colors["trajectory"][2])
            self.viewer.add_geom(self.traject)
            ###################################################
            ############### Target Path Rendering #############
            ###################################################
            TPR = []
            for i in range(len(self.wp)):
                temp = (self.wp[i][0]+(self.grid_size/2)/self.sr, (1+self.wp[i][1]+(self.grid_size/2))/self.sr)
                TPR.append(temp)
            self.TPR     = TPR
            self.traject = rendering.make_polyline(self.TPR) # for every time step, position should be updated
            self.traject.set_color(colors["required_path"][0],colors["required_path"][1],colors["required_path"][2])
            self.viewer.add_geom(self.traject)
           
         shipx = (self.current_state[3].item()+(self.grid_size/2))/self.sr
         shipy = (self.current_state[4].item()+(self.grid_size/2))/self.sr
         ship  = self._ship_geom
                   
         self.shiptrans.set_translation(shipx,shipy)
         self.shiptrans.set_rotation(-self.rtn)
         return self.viewer.render(return_rgb_array=mode == 'rgb_array')
     
        
     def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

            
            
################################
########## To Check ############
################################
# # Ship boundary points ##
# A = import_polygon(7)
# print(A)
# import matplotlib.pyplot as plt
# plt.figure(figsize=(6,6))
# plt.xlim(-35,35)
# plt.ylim(-35,35)
# for i in range(len(A)):
#     plt.scatter(A[i][0],A[i][1],color="g")
################################
############# End ##############
################################
        
        
        
        
     

