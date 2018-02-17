# -*- coding: utf-8 -*-
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os





class Basic_Agent(nn.Module):
    
    def __init__(self,args,DQN_model):
        super().__init__()
        
        self.batch_size = args.batch_size 
        self.discount = args.discount
        self.max_gradient_norm = args.max_gradient_norm
        self.epsilon = args.epsilon
        self.action_space = args.action_space
        self.hidden_size = args.hidden_size
        self.state_space = args.state_space
        
        self.main_dqn= DQN_model(args)
        self.target_dqn = DQN_model(args)
        
        if args.cuda:
            self.main_dqn.cuda()
            self.target_dqn.cuda()
        
        self.target_dqn_update()
        #target_param=list(target_dqn.parameters())
        #print("target update done ",main_param[0][0] , target_param[0][0])
        self.optimizer = optim.Adam(self.main_dqn.parameters(), lr=args.lr)
    
    def save(self,path ='./param.p'):
        torch.save(self.main_dqn.state_dict(),path)
        
    def load(self,path ='./param.p'):
        if os.path.exists(path):
            self.main_dqn.load_state_dict(torch.load(path))
        else :
            print("file not exist")
    
    def target_dqn_update(self):
        self.target_dqn.parameter_update(self.main_dqn)
        
    def get_action(self,state):
        ret = self.main_dqn(Variable(state,volatile=True).type(torch.FloatTensor).view(1,-1))
        action_value = ret.max(1)[0].data[0]
        action = ret.max(1)[1].data[0] #return max index call [1] 
        return action, action_value
        
    
    def learn(self,memory):
        random.seed(time.time())
        
        batch = memory.sample(self.batch_size)
        [states, actions, rewards, next_states, dones] = zip(*batch)
        state_batch = Variable( torch.stack(states,0).type(torch.FloatTensor))
        action_batch = Variable(torch.LongTensor(actions))
        reward_batch = Variable(torch.FloatTensor(rewards))
        next_states_batch = Variable(torch.stack(next_states,0).type(torch.FloatTensor))
        done_batch = Variable(torch.FloatTensor(dones))
        done_batch = -done_batch +1

        
        state_action_values = self.main_dqn(state_batch.view(self.batch_size,-1)).gather(1, action_batch.view(-1,1)).view(-1)
        next_states_batch.volatile = True
        next_state_values = self.target_dqn(next_states_batch.view(self.batch_size,-1)).max(1)[0]
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount * done_batch) + reward_batch
        expected_state_action_values.volatile = False
        loss = F.mse_loss(state_action_values, expected_state_action_values)        
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.main_dqn.parameters(), self.max_gradient_norm)  # Clip gradients (normalising by max value of gradient L2 norm)
        self.optimizer.step()
          
        
class Agent_conv2d(Basic_Agent):
    def __init__(self,args,DQN_model):
        super().__init__(args,DQN_model)
        
    
    def get_action(self,state):
        ret = self.main_dqn(Variable(state.unsqueeze(0).unsqueeze(0),volatile=True).type(torch.FloatTensor))
        action_value = ret.max(1)[0].data[0]
        action = ret.max(1)[1].data[0] #return max index call [1] 
        return action, action_value
        
    
    def learn(self,memory):
        random.seed(time.time())
        
        batch = memory.sample(self.batch_size)
        [states, actions, rewards, next_states, dones] = zip(*batch)
        state_batch = Variable( torch.stack(states,0).type(torch.FloatTensor),volatile = True).unsqueeze(1)
        action_batch = Variable(torch.LongTensor(actions),volatile = True)
        reward_batch = Variable(torch.FloatTensor(rewards),volatile = True)
        next_states_batch = Variable(torch.stack(next_states,0).type(torch.FloatTensor),volatile = True).unsqueeze(1)
        done_batch = Variable(torch.FloatTensor(dones),volatile = True)
        done_batch = -done_batch +1


                
        state_action_values = self.main_dqn(state_batch).gather(1, action_batch.view(-1,1)).view(-1)
        state_action_values.volatile = False
        state_action_values.requires_grad = True
        
        next_state_values = self.target_dqn(next_states_batch).max(1)[0]
        next_state_values.volatile = False
        next_state_values.requires_grad = True
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount * done_batch) + reward_batch
        expected_state_action_values.volatile = False
        expected_state_action_values.requires_grad = True
        
        loss = F.mse_loss(state_action_values, expected_state_action_values)        
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.main_dqn.parameters(), self.max_gradient_norm)  # Clip gradients (normalising by max value of gradient L2 norm)
        self.optimizer.step()
    
    
    
from model import DQN_rainbow

class Agent_rainbow(nn.Module):
    def __init__(self,args):
        super().__init__()
        
        self.batch_size = args.batch_size 
        self.discount = args.discount
        self.epsilon = args.epsilon
        self.action_space = args.action_space
        self.hidden_size = args.hidden_size
        self.state_space = args.state_space
        
#        self.main_dqn= DQN_model(args)
        self.main_dqn = 0
#        self.main_dqn.train()
        self.target_dqn = DQN_rainbow(args)
#        self.target_dqn = target_model
#        self.target_dqn_update()
#        self.target_dqn.eval()
        
        self.atoms = args.atoms
        self.v_min = args.V_min
        self.v_max = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, args.atoms)  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (args.atoms - 1)
        self.m = torch.zeros(args.batch_size, self.atoms).type(torch.FloatTensor)
        self.discount = args.discount
        self.priority_exponent = args.priority_exponent
        self.max_gradient_norm = args.max_gradient_norm
    
#        self.optimizer = optim.Adam(self.main_dqn.parameters(), lr=args.lr, eps=args.adam_eps)
        self.optimizer  = 0
      
        if args.cuda:
            self.main_dqn.cuda()
            self.target_dqn.cuda()
            self.support = self.support.cuda()
        
      
     # Resets noisy weights in all linear layers (of policy and target nets)
    def reset_noise(self):
        self.main_dqn.reset_noise()
        self.target_dqn.reset_noise()

    
    def save(self,path ='./param.p'):
        torch.save(self.main_dqn.state_dict(),path)
        
    def load(self,path ='./param.p'):
        if os.path.exists(path):
            self.main_dqn.load_state_dict(torch.load(path))
        else :
            print("file not exist")
    
    def target_dqn_update(self):
        self.target_dqn.parameter_update(self.main_dqn)
    
    
    def get_action(self, state):
        ret = (self.main_dqn(Variable(state, volatile=True).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)).data * self.support).sum(2)
        action = ret.max(1)[1][0]
        action_value = ret.max(1)[0][0]
        return action, action_value
    
    def prev_action(self,state):
        ret = self.main_dqn(Variable(state,volatile=True).type(torch.FloatTensor).view(1,-1))
        action_value = ret.max(1)[0].data[0]
        action = ret.max(1)[1].data[0] #return max index call [1] 
        return action, action_value
    
    def train(self):
        self.main_dqn.train()
    def eval(self):
        self.main_dqn.eval()
        
    
    def _get_categorical(self, next_states, rewards, mask):
        # input dim = [batch , channel , h, w ]
        batch_sz = next_states.size(0)
        gamma = self.discount

        # Compute probabilities p(x, a)
        probs = self.target_dqn(next_states).data
        #[batch x action x atoms] 
        qs = torch.mul(probs, self.support.expand_as(probs))
        
        
        argmax_a = qs.sum(2).max(1)[1].unsqueeze(1).unsqueeze(1)
        action_mask = argmax_a.expand(batch_sz, 1, self.atoms)
        #action mask  [batch x 1 x atoms]
        qa_probs = probs.gather(1, action_mask).view(batch_sz,self.atoms)
        #qa_probs [32 x 21 ]
        # Mask gamma and reshape it torgether with rewards to fit p(x,a).
        rewards = rewards.expand_as(qa_probs)
        gamma = (mask * gamma).expand_as(qa_probs)

        # Compute projection of the application of the Bellman operator.
#        bellman_op = rewards + gamma * self.support.unsqueeze(0).expand_as(rewards)
        bellman_op = rewards + gamma * self.support.unsqueeze(0).expand_as(rewards)
        bellman_op = torch.clamp(bellman_op, self.v_min, self.v_max)

        # Compute categorical indices for distributing the probability
        m = self.m.fill_(0)
        b = (bellman_op - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # Distribute probability
        """
        for i in range(batch_sz):
            for j in range(self.atoms):
                uidx = u[i][j]
                lidx = l[i][j]
                m[i][lidx] = m[i][lidx] + qa_probs[i][j] * (uidx - b[i][j])
                m[i][uidx] = m[i][uidx] + qa_probs[i][j] * (b[i][j] - lidx)
        for i in range(batch_sz):
            m[i].index_add_(0, l[i], qa_probs[i] * (u[i].float() - b[i]))
            m[i].index_add_(0, u[i], qa_probs[i] * (b[i] - l[i].float()))

        """
        # Optimized by https://github.com/tudor-berariu
        offset = torch.linspace(0, ((batch_sz - 1) * self.atoms), batch_sz)\
            .type(torch.LongTensor)\
            .unsqueeze(1).expand(batch_sz, self.atoms)

        m.view(-1).index_add_(0, (l + offset).view(-1),
                              (qa_probs * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1),
                              (qa_probs * (b - l.float())).view(-1))
        return Variable(m)



    def get_td_error(self,re,st,ac,st_,dd):
        re = torch.FloatTensor([re])
        st = Variable(st.type(torch.FloatTensor)).unsqueeze(0).unsqueeze(0)
        st_ = Variable(st_.type(torch.FloatTensor)).unsqueeze(0).unsqueeze(0)
        ac = torch.LongTensor([ac]).unsqueeze(0)
        dd = -torch.FloatTensor([dd])+1
        
        # input dim = [batch , channel , h, w ]
        target_qa_probs = self._get_categorical(st_,re,dd)[0]
        qa_probs = self.main_dqn(st)[0,ac[0,0],:]
        
        td_error = target_qa_probs - qa_probs
#        td_error = re + self.discount * self.target_dqn(st_).max(1)[0] - self.main_dqn(st).gather(1,ac)
        return abs(td_error.sum().data[0])
    
    
    def learn(self,memory):
        
        #distributional 
        batch, batch_idx  = memory.sample(self.batch_size)
       
        [states, actions, rewards, next_states, dones] = zip(*batch)
        
        mask = (-torch.FloatTensor(dones)+1).view(-1,1)
        rewards = torch.FloatTensor(rewards).view(-1,1)
        
        states = Variable(torch.stack(states).type(torch.FloatTensor)).unsqueeze(1)
        actions = Variable(torch.LongTensor(actions))
        rewards = (torch.FloatTensor(rewards))
        next_states = Variable(torch.stack(next_states).type(torch.FloatTensor)).unsqueeze(1)
        
        
        # Compute probabilities of Q(s,a*)
        q_probs = self.main_dqn(states)
        actions = actions.view(self.batch_size, 1, 1)
        action_mask = actions.expand(self.batch_size, 1, self.atoms)
        qa_probs = q_probs.gather(1, action_mask).squeeze()

        # Compute distribution of Q(s_,a)
        target_qa_probs = self._get_categorical(next_states, rewards, mask)

        # Compute the cross-entropy of phi(TZ(x_,a)) || Z(x,a)
        qa_probs.data.clamp_(0.01, 0.99)   # Tudor's trick for avoiding nans
        
        
        loss = (target_qa_probs - qa_probs).sum(1).mean()
#        loss = - torch.sum(target_qa_probs * torch.log(qa_probs))
        # kl_div 는 항상 >0 큼으로 기본적으로 -1 시스템에서 사용할 수 없다. 
#        loss = F.kl_div(qa_probs,target_qa_probs)  
        
        
        
        td_error = target_qa_probs - qa_probs
        for i in range(self.batch_size):
            val = abs(td_error[i].data[0])
            memory.update(batch_idx[i],val)
            
        print('loss : ',loss.data[0])
        # Accumulate gradients
        self.main_dqn.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.main_dqn.parameters(), self.max_gradient_norm)  # Clip gradients (normalising by max value of gradient L2 norm)
        self.optimizer.step()
        
        
    
    
    
    
    
    
    