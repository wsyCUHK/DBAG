import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW as Adam
from lazy_adam import LazyAdam
from utils_drl import soft_update, hard_update
from models import GaussianPolicy, QNetwork, DeterministicPolicy
import numpy as np
from prioritized_memory import Memory
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as func

def score_to_weight(x):
    return min(np.exp(-(x-1))**0.5,1)

class SAC(object):
    def __init__(self, num_inputs, action_space, args):
        self.memory=Memory(args.replay_size)
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = args.device

        #Similar to Double-QNetwork
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size,args.user,args.embedding_size,args.num_header,args.drop,args.device,args.layers).to(device=self.device)
        self.critic_optim =Adam(self.critic.parameters(), lr=args.clr)
        self.critic_optim_scheduler= CosineAnnealingLR(self.critic_optim,T_max=args.num_steps)
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size,args.user,args.embedding_size,args.num_header,args.drop,args.device,args.layers).to(self.device)
        hard_update(self.critic_target, self.critic)
        #The two networks are with the same initialization

        #Two option policy, stochastic(Gaussian) or Deterministic 
        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.alpha_lr)
                self.alpha_optim_scheduler= CosineAnnealingLR(self.alpha_optim,T_max=args.num_steps)
            

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size,args.user, action_space,args.embedding_size,args.num_header).to(self.device)
            self.policy_optim =Adam(self.policy.parameters(), lr=args.alr)
            self.policy_optim_scheduler= CosineAnnealingLR(self.policy_optim,T_max=args.num_steps)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim =Adam(self.policy.parameters(), lr=args.alr)
            self.policy_optim_scheduler= CosineAnnealingLR(self.policy_optim,T_max=args.num_steps)
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        #input is the state, output is the action
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def alpha_reset(self,args):
        if self.log_alpha>1:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=args.alpha_lr)
            self.alpha_optim_scheduler= CosineAnnealingLR(self.alpha_optim,T_max=args.num_steps)
            
        return 

    def alpha_zeros(self,args):
        self.alpha = 0
        self.automatic_entropy_tuning = False
        return 

    def append_sample(self, state, action, reward, next_state, done):
        state_batch = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        next_state_batch = torch.FloatTensor(next_state).to(self.device).unsqueeze(0)
        action_batch = torch.FloatTensor(action).to(self.device).unsqueeze(0)
        reward_batch = torch.FloatTensor([reward]).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor([done]).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            #Under current \theta, generate the action and probability
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            #Under current \theta, Q value
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            #TD error of instances
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            #TD error of the batch
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            
        qf1, qf2 = self.critic(state_batch, action_batch)

        error = abs(qf1- next_q_value)+abs(qf2- next_q_value)
        self.memory.add(error.detach().cpu().numpy()+1e-5, (state, action, reward, next_state, done))

        return 


    def update_parameters(self, batch_size, updates,num_users,per_frame_length,history_length):
        # Sample a batch from memory
        #state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        
        


        mini_batch1,mini_batch2,mini_batch3,mini_batch4,mini_batch5, idxs, is_weights = self.memory.sample(batch_size)
        state_batch = torch.FloatTensor(np.array(mini_batch1)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(mini_batch4)).to(self.device)
        action_batch = torch.FloatTensor(np.array(mini_batch2)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(mini_batch3)).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(np.array(mini_batch5)).to(self.device).unsqueeze(1)

        
        with torch.no_grad():
            #Under current \theta, generate the action and probability
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            #Under current \theta, Q value
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            #TD error of instances
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            #TD error of the batch
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            
        qf1, qf2 = self.critic(state_batch, action_batch)
        
        # qf1_loss = F.mse_loss(weight*qf1, weight*next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # qf2_loss = F.mse_loss(weight*qf2, weight*next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf1_loss =  (torch.FloatTensor(is_weights).to(self.device) * F.mse_loss(qf1, next_q_value,reduction='none')).mean()# JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss =  (torch.FloatTensor(is_weights).to(self.device) * F.mse_loss(qf2, next_q_value,reduction='none')).mean() # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss
        

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        self.critic_optim_scheduler.step()
        

        pi, log_pi, _ = self.policy.sample(state_batch)

       
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
            
        
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = (torch.FloatTensor(is_weights).to(self.device)*((self.alpha * log_pi) - min_qf_pi)).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        errors = abs(qf1- next_q_value)+abs(qf2- next_q_value)+1e-5

        # update priority
        for i in range(batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i].detach().cpu().numpy())


        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        self.policy_optim_scheduler.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (torch.FloatTensor(is_weights).to(self.device) *log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha_optim_scheduler.step()


            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

