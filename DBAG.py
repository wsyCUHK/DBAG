import argparse
import datetime
import numpy as np
import itertools
import torch
import random
import math
from sac_framework import SAC
import torch
from PIL import Image
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import pickle

#############################################################################
#########################Function Define##################################################
##################################################################################
class GELU(torch.nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def project_label(xxyy):
    xyz=np.zeros(xxyy.shape)
    xyz[:,0]=xxyy[:,0]
    xyz[:,1]=xxyy[:,1]
    xyz[:,2]=xxyy[:,0]+xxyy[:,2]
    xyz[:,3]=xxyy[:,1]+xxyy[:,3]
    return xyz

def NMS(boxes,Boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    boxes = boxes.astype("float32")
    Boxes = Boxes.astype("float32")
    # initialize the list of picked indexes 
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    x1B = Boxes[:,0]
    y1B = Boxes[:,1]
    x2B = Boxes[:,2]
    y2B = Boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2B - x1B + 1) * (y2B - y1B + 1)
    #print(area)
    idxs = np.argsort(y2)
    idxs2=np.argsort(y2B)
    # keep looping while some indexes still remain in the indexes
    start_size=len(idxs2)
    for i in range(len(idxs)):
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1B[idxs2[:]])
        yy1 = np.maximum(y1[i], y1B[idxs2[:]])
        xx2 = np.minimum(x2[i], x2B[idxs2[:]])
        yy2 = np.minimum(y2[i], y2B[idxs2[:]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs2[:]]
        # delete all indexes from the index list that have
        idxs2 = np.delete(idxs2, np.where(overlap > overlapThresh)[0])
    # return only the bounding boxes that were picked using the
    # integer data type
    return (start_size-len(idxs2))/(start_size)
def pj1(df1,sizex,sizey):
    nparray=df1.values[:,1:]
    output=np.zeros(nparray.shape)
    output[:,0]=sizex*(nparray[:,0]-nparray[:,2]/2)
    output[:,1]=sizey*(nparray[:,1]-nparray[:,3]/2)
    output[:,2]=sizex*(nparray[:,0]+nparray[:,2]/2)
    output[:,3]=sizey*(nparray[:,1]+nparray[:,3]/2)
    return output

##################################################################################
##################argparse for easy parameter tunning#######################################
##################################################################################
parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient (default: 0.005)')
parser.add_argument('--alr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--clr', type=float, default=0.005, metavar='G',
                    help='learning rate (default: 0.005)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=20000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', type=bool, default=True,
                    help='run on CUDA (default: False)')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='run on CUDA (default: False)')
parser.add_argument('--history_length', type=int, default=10, metavar='N',
                    help='the state length')
parser.add_argument('--filename', type=str, default='test', help='save_file_name')
parser.add_argument('--num_header', type=int, default=4)
parser.add_argument('--num_layer', type=int, default=1)
parser.add_argument('--dataset', type=str, default='SelfDriving')
parser.add_argument('--user', type=int, default=2)
parser.add_argument('--maxep', type=int, default=32)
parser.add_argument('--eta', type=float, default=5)
parser.add_argument('--embedding_size', type=int, default=64)
parser.add_argument('--drop', type=float, default=0.15)
parser.add_argument('--rat', type=float, default=0.01)
parser.add_argument('--nstep', type=int, default=3)
parser.add_argument('--alpha_lr', type=float, default=0.01)
parser.add_argument('--layers', type=int, default=2)
args = parser.parse_args()

##############################################################################
##################Hyper Parameters############################################
##############################################################################
random.seed(args.seed)
np.random.seed(args.seed)

F_c = 915*1e6  # carrier bandwidth
A_d = 4.11  # antenna gain
degree = 3  # path loss value
light = 3*1e8  # speed of light
v0=1e-3 * np.power(10, 0.1 * -174)*10**6#4*10**(-15)
eta=args.eta
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP_STEP = args.maxep
bandw_ttl=2
P=0.1
N0=10**(-10)
scale_reward=100
num_of_user=args.user
history_length=args.history_length
beta=2
gamma=args.gamma
args.gamma=np.power(gamma,args.nstep)


#############################################################################################################################
###Load the channel, object detection result, and resource allocation datasets############################################
##########################################################################################################################
with open('./preprocessed_data/distance_channel_'+str(args.seed)+'.pickle', 'rb') as f:
        dist, channel_hist=pickle.load(f)
distv = dist[:num_of_user]
bandwidth=2*10**6
if args.dataset=='SelfDriving':
    def user_location_reset():
        user_screen_index=np.zeros((num_of_user,))
        for i in range(num_of_user):
            user_screen_index[i]=np.random.randint(10000)
        user_screen_index=user_screen_index.astype('int')
        return user_screen_index
    def user_location_reset_test():
        user_screen_index=np.zeros((num_of_user,))
        for i in range(num_of_user):
            user_screen_index[i]=np.random.randint(9000,10000)
        user_screen_index=user_screen_index.astype('int')
        return user_screen_index    
    test_screen_index=10000
    with open('./preprocessed_data/SelfDriving_TrainingData0729.pickle','rb') as f:
        i_time,d_time,c_c,A5,AP7=pickle.load(f)
    resolution=np.array([1920*1200,1920*1200/2,1920*1200/4,1920*1200/8])
    width_detect=np.array([1920,960,480,240])
    test_length=3000
elif args.dataset=='Drone':
    def user_location_reset():
        user_screen_index=np.zeros((num_of_user,))
        for i in range(num_of_user):
            user_screen_index[i]=np.random.randint(800)
        user_screen_index=user_screen_index.astype('int')
        return user_screen_index
    def user_location_reset_test():
        user_screen_index=np.zeros((num_of_user,))
        for i in range(num_of_user):
            user_screen_index[i]=np.random.randint(700,800)
        user_screen_index=user_screen_index.astype('int')
        return user_screen_index 
    test_screen_index=800
    with open('./preprocessed_data/Drone_TrainingDatav2.pickle','rb') as f:
        i_time,d_time,c_c,A5,AP7=pickle.load(f)
    resolution=np.array([1400*788,1400*788/2,1400*788/4,1400*788/8])
    width_detect=np.array([1400,700,350,175])
    test_length=800
else:
    print('There is no dataset named ',args.dataset)
    exit()
AP5=[]
AP75=[]
CC=[]
length_ratio=7


#####################################################################################################
########Pre-Train the DNN for Bandwidth Allocation###################################################
#####################################################################################################
from scipy import io
import torch.nn as nn
import torch.nn.functional as F
file_list=['./preprocessed_data/training_N1_0805.mat','./preprocessed_data/training_N2_0805.mat','./preprocessed_data/training_N3_0805.mat','./preprocessed_data/training_N4_0805.mat','./preprocessed_data/training_N5_0805.mat']
data=io.loadmat(file_list[num_of_user-1])
inputs=np.transpose(data['conditions'][:,:39000])
labels=np.transpose(data['optimal_a'][:,:39000])
inputs2=np.transpose(data['conditions'][:,39000:40000])
labels2=np.transpose(data['optimal_a'][:,39000:40000])
from sklearn.preprocessing import StandardScaler
scale= StandardScaler()
inputs = scale.fit_transform(inputs)
inputs_val=scale.transform(inputs2)
from torch.utils.data import TensorDataset, DataLoader
tensor_x = torch.Tensor(inputs)#.transpose(0,1) # transform to torch tensor
tensor_y = torch.Tensor(labels)#.transpose(0,1)
my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
my_dataloader = DataLoader(my_dataset, batch_size=128, shuffle=True) # create your dataloader
tensor_xv = torch.Tensor(inputs_val)#.transpose(0,1) # transform to torch tensor
tensor_yv = torch.Tensor(labels2)#.transpose(0,1)
val_dataset = TensorDataset(tensor_xv,tensor_yv) # create your datset
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True) # create your dataloader
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        #self.conv1 = nn.Conv2d(1, 6, 5)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(2*num_of_user, 128)  # 5*5 from image dimension
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, num_of_user)
        #self.relu = GELU()
        self.active=GELU()
        self.m = nn.Softmax(dim=1)
        self.apply(weights_init_)
        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        #x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        #print(x.shape)
        x = self.active(self.fc1(x))
        x = self.active(self.fc2(x))
        x = self.m(self.fc3(x))
        return x

bandwidth_allocator = Net()
import torch.optim as optim
use_cuda = torch.cuda.is_available()
device = torch.device(args.device)
bandwidth_allocator.to(device)
criterion=torch.nn.SmoothL1Loss()
optimizer=optim.Adam(bandwidth_allocator.parameters(), lr=0.001)

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(my_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = bandwidth_allocator(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 10 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
            with torch.no_grad():
                val_loss=0.0
                for j, data_val in enumerate(val_dataloader, 0):
                    inputs_val, labels_val = data_val
                    outputs = bandwidth_allocator(inputs_val.to(device))
                    loss = criterion(outputs, labels_val.to(device))
                    val_loss+=loss.item()
            print('[%d, %5d] val_loss: %.3f' %
                  (epoch + 1, i + 1, val_loss/j))

print('Finished Pre-Training')

####################################################################################################
##############################Here Comes the main story#############################################
####################################################################################################


def env(a,s,clock):
    ##############Image Offloading################################################
    a=np.floor(a*0.999999)
    channel_c=s[np.arange(0,(num_of_user)*length_ratio,length_ratio),-1]*10**(-7)
    Trans_size=resolution[a.astype('int')] #in mb
    if num_of_user==1:
        allocation=np.ones((1,))
    else:
        cvxopt_para=scale.transform((np.concatenate((channel_c,Trans_size),axis=0)).reshape((1,2*num_of_user)))
        with torch.no_grad():
            allocation=bandwidth_allocator(torch.tensor(cvxopt_para,device=device).float()).cpu().numpy().reshape((num_of_user,))
        if np.sum(allocation)==0:
            allocation[:]=1.0/num_of_user
        else:
            allocation=allocation/np.sum(allocation)

    Uplink_rate=allocation*bandw_ttl*np.log(1+P*channel_c/(v0*allocation*bandw_ttl))/np.log(2)
    Uplink_rate=np.nan_to_num(Uplink_rate)

    if np.min(Uplink_rate)<0.0001:
         print(Uplink_rate)
         print(allocation)
         print(s)
    Trans_time=(Trans_size/(10**6))/(Uplink_rate)
    Trans_time[np.where(Trans_time>1)]=1

    ##############Server Inference################################################
    Inf_time=np.zeros((num_of_user,))
    Inf_cc=np.zeros((num_of_user,1))
    deg_time=np.zeros((num_of_user,))
    acc=0
    for i in range(num_of_user):
        deg_time[i]=d_time[user_screen_index[i],a[i].astype('int')]
        Inf_time[i]=d_time[user_screen_index[i],a[i].astype('int')]
        Inf_cc[i]= c_c[user_screen_index[i],a[i].astype('int')]
        acc+=AP7[user_screen_index[i],a[i].astype('int')]

    ##############Wireless Channel Model##########################################
    

    total_latency=Inf_time+Trans_time+deg_time
    total_latency=total_latency.reshape(num_of_user,1)
    channel_d=channel_hist[:num_of_user,clock]
    new_current_state=np.zeros((num_of_user*length_ratio,1))
    for i in range(num_of_user):
        new_current_state[i*length_ratio,0]=channel_d[i]*10**(7)
        new_current_state[i*length_ratio+1,0]=a[i]
        new_current_state[i*length_ratio+2,0]=Inf_cc[i]
        new_current_state[i*length_ratio+3,0]=deg_time[i]
        new_current_state[i*length_ratio+4,0]=Inf_time[i]
        new_current_state[i*length_ratio+5,0]=Inf_cc[i]
        new_current_state[i*length_ratio+6,0]=allocation[i]
    new_state=np.concatenate((s[:,1:],new_current_state),axis=1)
    
    return scale_reward*(np.mean(Inf_cc)-eta*(np.mean(Inf_time)+np.mean(Trans_time)+np.mean(deg_time))), new_state,Inf_time,Trans_time,Inf_cc,deg_time,acc/num_of_user





#Initialization
action_space=np.ones((num_of_user,))
for i in range(num_of_user):
    action_space[i]=resolution.shape[0]
agent = SAC(num_of_user*history_length*length_ratio,action_space , args)



def random_actor(num_of_user):
    action=np.zeros((num_of_user,))
    for j in range(num_of_user):
        action[j]=random.randint(0,resolution.shape[0]-1)
    return action

def env_reset(user_screen_index):
    # Generate the channel gain for next time slot
    s=np.zeros((num_of_user*length_ratio,history_length))
    for i in range(num_of_user):
        s[i*length_ratio,-1]= 10**(7)*np.random.exponential() * A_d * (light / (4.0 * 3.141592653589793 * F_c * distv[i]))**degree
     
   
    for i in range(history_length):
        action=random_actor(num_of_user)
        _,s_,_,_,_,_,_= env(action,s,np.random.randint(19999))
        user_screen_index+=1
        s=s_
    return action,s





# Training Loop
total_numsteps = 0
updates = 0
plot_log=[]
plot_test=[]
step_log=[]
energy_con_log=[]
gain_log=[]
bitrate_request_log=[]
cclog=[]
ttlog=[]
itlog=[]
dtlog=[]
action_log=[]
ap_log=[]
state_log=[]
nstep=args.nstep

for i_episode in itertools.count(1):
    ###Training Initialization
    episode_reward = 0
    episode_reward_with_explore=0
    episode_steps = 0
    last_print_explore=0
    last_print=0
    done = False
    user_screen_index=user_location_reset()
    store_index=user_screen_index
    action, state=env_reset(user_screen_index)
    reset_flag=1
    ###############################

    while not done:
        if args.start_steps > total_numsteps:
            #action = env.action_space.sample()  # Sample random action
            if random.random()<1.0/np.sqrt(np.sqrt(total_numsteps+1)):
                action =random_actor(num_of_user)
            else:
                action = agent.select_action(state)
        else:
            action = agent.select_action(state)  # Sample action from policy
            if reset_flag==1:
                agent.alpha_reset(args)
                reset_flag=0
            if total_numsteps==args.num_steps-2000:
                agent.alpha_zeros(args)
        if total_numsteps> args.batch_size:  
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(args.batch_size, updates,num_of_user,length_ratio,history_length)
            updates += 1

        reward, next_state,itime,ttime,icc,dtime,ap_temp= env(action,state,total_numsteps) # Step
        episode_steps += 1
        total_numsteps += 1
        user_screen_index+=1
        episode_reward += reward
        state_log.append(state)
        step_log.append(reward)
        itlog.append(itime)
        ttlog.append(ttime)
        cclog.append(icc.squeeze())
        action_log.append(action)
        dtlog.append(dtime)
        ap_log.append(ap_temp)
        if episode_steps == MAX_EP_STEP:
            mask = 1
            done =True
        else:
            mask=float(not done)
        episode_reward_with_explore+= reward

        try:
            ###n-step return###
            acm_reward=0
            for jj in range(nstep):
                acm_reward+=np.power(gamma,jj)*step_log[total_numsteps-nstep+jj]

            agent.append_sample(state_log[total_numsteps-nstep], action_log[total_numsteps-nstep], acm_reward, next_state, mask)
        except:
            pass
        state = next_state



        if total_numsteps%99==0:
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, reward+: {}".format(i_episode, total_numsteps, episode_steps, round((episode_reward-last_print)/99, 2),round((episode_reward_with_explore-last_print_explore)/99, 2)))
            last_print_explore=episode_reward_with_explore
            last_print=episode_reward



    if total_numsteps > args.num_steps:
        break

    plot_log.append(round(episode_reward, 2))
    


file_id=0
file_exists=True
while file_exists:
    if os.path.isfile('./results/'+args.filename+str(file_id)+str(eta)+'.pickle'):
        print('File Exists')
        file_id+=1
    else:
        print('Save_model: ',args.filename+str(file_id))
        agent.save_model(args.filename+str(file_id))
        with open('./results/'+args.filename+str(file_id)+str(eta)+'.pickle', 'wb') as f:
            pickle.dump([ap_log, step_log, itlog,ttlog,cclog,dtlog], f)
        file_exists=False







total_numsteps = 0
updates = 0
plot_log=[]
plot_test=[]
step_log=[]
energy_con_log=[]
gain_log=[]
bitrate_request_log=[]
cclog=[]
ttlog=[]
itlog=[]
dtlog=[]
action_log=[]
ap_log=[]
state_log=[]
nstep=args.nstep


episode_reward = 0
episode_reward_with_explore=0
episode_steps = 0
last_print_explore=0
last_print=0
done = False
user_screen_index=user_location_reset_test()
action, state=env_reset(user_screen_index)
reset_flag=1
###############################

while not done:
    action = agent.select_action(state,evaluate=True)  # Sample action from policy
    reward, next_state,itime,ttime,icc,dtime,ap_temp= env(action,state,total_numsteps) # Step
    episode_steps += 1
    total_numsteps += 1
    user_screen_index+=1
    episode_reward += reward
    state_log.append(state)
    step_log.append(reward)
    itlog.append(itime)
    ttlog.append(ttime)
    cclog.append(icc.squeeze())
    action_log.append(action)
    dtlog.append(dtime)
    ap_log.append(ap_temp)
    if episode_steps == test_length:
        mask = 1
        done =True
    else:
        mask=float(not done)
    episode_reward_with_explore+= reward
    state = next_state


file_id=0
file_exists=True
while file_exists:
    if os.path.isfile('./results/'+args.filename+str(file_id)+str(eta)+'test.pickle'):
        print('File Exists')
        file_id+=1
    else:
        print('Save_model: ',args.filename+str(file_id))
        agent.save_model(args.filename+str(file_id))
        with open('./results/'+args.filename+str(file_id)+str(eta)+'test.pickle', 'wb') as f:
            pickle.dump([ap_log, step_log, itlog,ttlog,cclog,dtlog], f)
        file_exists=False