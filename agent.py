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

import numpy as np
import copy 


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors -- output from policy function - a list of tuples of actions
            and their prior probability according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value, Q plus bonus u(P).
        Returns:
        A tuple of (action, next_node)
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        Arguments:
        leaf_value -- the value of subtree evaluation from the current player's perspective.        
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value) 

    def get_value(self, c_puct):
        """Calculate and return the value for this node: a combination of leaf evaluations, Q, and
        this node's prior adjusted for its visit count, u
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.
        """
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search.
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """Arguments:
        policy_value_fn -- a function that takes in a board state and outputs a list of (action, probability)
            tuples and also a score in [-1, 1] (i.e. the expected value of the end game score from 
            the current player's perspective) for the current player.
        c_puct -- a number in (0, inf) that controls how quickly exploration converges to the
            maximum-value policy, where a higher value means relying on the prior more
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at the leaf and
        propagating it back through its parents. State is modified in-place, so a copy must be
        provided.
        Arguments:
        state -- a copy of the state.
        """
        node = self._root
        while(1):            
            if node.is_leaf():
                break                
            # Greedily select next move.
            action, node = node.select(self._c_puct)  
            state.step(action)
#            state.do_move(action)

        # Evaluate the leaf using a network which outputs a list of (action, probability)
        # tuples p and also a score v in [-1, 1] for the current player.
        action_probs, leaf_value = self._policy(state)
        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player() else -1.0

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Runs all playouts sequentially and returns the available actions and their corresponding probabilities 
        Arguments:
        state -- the current state, including both game state and the current player.
        temp -- temperature parameter in (0, 1] that controls the level of exploration
        Returns:
        the available actions and the corresponding probabilities 
        """        
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
  
        # calc the move probabilities based on the visit counts at the root node
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))       
         
        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"
        

class Agent_MCTS(nn.Module):
    def __init__(self,args,share_model,opti):
        super().__init__()
        c_puct=5, n_playout=2000, is_selfplay=1
        
        self.mcts = MCTS(self.main_dqn, c_puct, n_playout)
        
        self.batch_size = args.batch_size 
        self.discount = args.discount
        self.epsilon = args.epsilon
        self.action_space = args.action_space
        self.hidden_size = args.hidden_size
        self.state_space = args.state_space
        
#        self.main_dqn= DQN_model(args)
        self.main_dqn = share_model
#        self.main_dqn.train()
#        self.target_dqn = DQN_rainbow(args)
#        self.target_dqn = target_model
#        self.target_dqn_update()
#        self.target_dqn.eval()
        
#        self.optimizer = optim.Adam(self.main_dqn.parameters(), lr=args.lr, eps=args.adam_eps)
        self.optimizer  = opti
      
    def reset_player(self):
        self.mcts.update_with_move(-1) 
        
        
    def save(self,path ='./param.p'):
        torch.save(self.main_dqn.state_dict(),path)
        
    def load(self,path ='./param.p'):
        if os.path.exists(path):
            self.main_dqn.load_state_dict(torch.load(path))
        else :
            print("file not exist")
    
    def target_dqn_update(self):
        self.target_dqn.parameter_update(self.main_dqn)
    
    
    def get_action(self, env):
        sensible_moves = env.availables
        move_probs = np.zeros(env.width*env.height)
        
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(env, temp)
            
            move_probs[list(acts)] = probs         
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for self-play training)
                move = np.random.choice(acts, p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))    
                self.mcts.update_with_move(move) # update the root node and reuse the search tree
            else:
                # with the default temp=1e-3, thisZ is almost equivalent to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)       
                # reset the root node
                self.mcts.update_with_move(-1)             
#                location = env.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))
                
            return move, move_probs
        else:            
            print("WARNING: the board is full")
        
    
    def train(self):
        self.main_dqn.train()
    def eval(self):
        self.main_dqn.eval()
        
    
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
        
        
    
    
    
    
    
    
    