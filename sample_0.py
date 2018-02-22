# -*- coding: utf-8 -*-
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
#import torchvision.transforms as T
from collections import defaultdict, deque
import sys
import os 

import argparse
"""
X - MCTS
"""
#"""
#define test function
#"""
from plot import _plot_line

def run_process(args,share_model,board_max,rank):
    
    from checkerboard import Checkerboard
    env = Checkerboard(board_max, args.render)
    
    from agent import Agent_rainbow
    B_Agent = Agent_rainbow(args)
    W_Agent = Agent_rainbow(args)
    B_Agent.main_dqn = B_share_model
    W_Agent.main_dqn = W_share_model
    B_Agent.optimizer = optim.Adam(B_share_model.parameters(), lr=args.lr, eps=args.adam_eps)
    W_Agent.optimizer = optim.Adam(W_share_model.parameters(), lr=args.lr, eps=args.adam_eps)
    
#    from memory import PER_Memory
#    memory = PER_Memory(args)
    data_buffer = deque(maxlen=args.memory_capacity)     
    
    """
    main loop
    """
    global_count = 0
    episode = 0

    W_Agent.target_dqn_update()
    B_Agent.target_dqn_update()
    W_Agent.train()
    B_Agent.train()

    Ts =[]
    Trewards =[]
    TQmax = []
    while episode < args.max_episode_length:
        random.seed(time.time())
        T=0
        turn = 0
        max_action_value = -999999999999999
        state = env.reset()
        evaluation = False
        total_reward = 0
        if episode % args.evaluation_interval == 0 :
            evaluation = True
    #    args.epsilon -= 0.8/args.max_episode_length
        while T < args.max_step:
            
            action_value = -999999999999999
            if T%2 == 0 :
                Agent_ptr = B_Agent
                turn = env.black
            else:
                Agent_ptr = W_Agent
                turn = env.white
            
            if not evaluation and (random.random() <= args.epsilon or global_count < args.learn_start ):
                action = env.get_random_xy_flat()
            else:
                action, action_value = Agent_ptr.get_action(state)
            
            max_action_value = max(max_action_value,action_value)
            
            next_state , reward , done, _ = env.step_flat(action,turn)

            total_reward += reward
            memory.push(td_error,[state, action, reward, next_state, done])
            state = next_state
            
            
            
            # replay_interval, target_update_interval  only used  odd number 
            if not evaluation and global_count % args.replay_interval == 0 and global_count > args.learn_start:
                Agent_ptr.learn(memory)
                Agent_ptr.reset_noise()

            if not evaluation and global_count % args.target_update_interval == 0 :
                Agent_ptr.target_dqn_update()
                
                
            T += 1
            global_count += 1
            
            if done :
                B_Agent.reset_noise()
                W_Agent.reset_noise()
                
                if args.render:
                    env.render()
                break
        
        if evaluation :
            print('episode : ', episode, '  step : ',T, ' max_action ',max_action_value, 'total_reward : ' , total_reward)
            Ts.append(episode)
            Trewards.append([total_reward])
            TQmax.append([max_action_value])
            _plot_line(Ts, Trewards, 'rewards_'+args.name+'_'+str(rank), path='results')
            _plot_line(Ts, TQmax, 'Q_'+args.name+'_'+str(rank), path='results')
        if episode % args.save_interval ==0 :
            print('save')
            B_Agent.save('B'+args.name)
            W_Agent.save('W'+args.name)

        episode += 1
        




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='DQN')
    parser.add_argument('--name', type=str, default='main_rainbow_multi.p', help='stored name')
    parser.add_argument('--epsilon', type=float, default=0.05, help='random action select probability')
    #parser.add_argument('--render', type=bool, default=True, help='enable rendering')
    parser.add_argument('--render', type=bool, default=False, help='enable rendering')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    #parser.add_argument('--game', type=str, default='CartPole-v1', help='gym game')
    #parser.add_argument('--game', type=str, default='Acrobot-v1', help='gym game')
    #parser.add_argument('--game', type=str, default='MountainCar-v0', help='gym game')
    #parser.add_argument('--max-step', type=int, default=500, metavar='STEPS', help='Number of training steps (4x number of frames)')
    parser.add_argument('--action-space', type=int, default=2 ,help='game action space')
    parser.add_argument('--state-space', type=int, default=4 ,help='game action space')
    parser.add_argument('--max-episode-length', type=int, default=100000, metavar='LENGTH', help='Max episode length (0 to disable)')
#    parser.add_argument('--history-length', type=int, default=1, metavar='T', help='Number of consecutive states processed')
    parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
#    parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
#    parser.add_argument('--atoms', type=int, default=11, metavar='C', help='Discretised size of value distribution')
#    parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
#    parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
    #parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
    parser.add_argument('--memory-capacity', type=int, default=1000000, metavar='CAPACITY', help='Experience replay memory capacity')
    parser.add_argument('--learn-start', type=int, default=1 , metavar='STEPS', help='Number of steps before starting training')
    parser.add_argument('--replay-interval', type=int, default=1, metavar='k', help='Frequency of sampling from memory')
#    parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent')
#    parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
    #parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
    parser.add_argument('--target-update-interval', type=int, default=1, metavar='τ', help='Number of steps after which to update target network')
#    parser.add_argument('--reward-clip', type=int, default=10, metavar='VALUE', help='Reward clipping (0 to disable)')
    parser.add_argument('--lr', type=float, default=0.0000625, metavar='η', help='Learning rate')
#    parser.add_argument('--lr', type=float, default=0.0000625, metavar='η', help='Learning rate')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
    parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
    parser.add_argument('--max-gradient-norm', type=float, default=10, metavar='VALUE', help='Max value of gradient L2 norm for gradient clipping')
    parser.add_argument('--save-interval', type=int, default=1000, metavar='SAVE', help='Save interval')
    #parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--evaluation-interval', type=int, default=20, metavar='STEPS', help='Number of training steps between evaluations')
    #parser.add_argument('--evaluation-episodes', type=int, default=1, metavar='N', help='Number of evaluation episodes to average over')
    #parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
    #parser.add_argument('--log-interval', type=int, default=25000, metavar='STEPS', help='Number of training steps between logging status')
    
    # Setup
    args = parser.parse_args()
    " disable cuda "
    args.disable_cuda = True
        
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
      print(' ' * 26 + k + ': ' + str(v))
    args.cuda = torch.cuda.is_available() and not args.disable_cuda
    torch.manual_seed(random.randint(1, 10000))
    if args.cuda:
      torch.cuda.manual_seed(random.randint(1, 10000))
    
    
    
    #board setup 
    board_max = 7
    args.max_step = board_max*board_max
    args.action_space = board_max*board_max
    args.state_space = board_max*board_max
    args.memory_capacity = 100000
    args.learn_start = 1
    args.max_episode_length = 100000
#    args.render = True
    
    
    
    
#    if args.replay_interval % 2 ==0:
#        args.replay_interval += 1 
#    if args.target_update_interval % 2 == 0:
#        args.target_update_interval += 1
#    

    
    from model import PV_NET
    share_model = PV_NET(args)
    share_model.share_memory()
#    if os.path.exists('B'+args.name):
#        print('load')
#        B_share_model.load_state_dict(torch.load('B'+args.name))
#        W_share_model.load_state_dict(torch.load('W'+args.name))
    
    run_process(args,share_model,board_max,999)
    
#    num_processes = 8
#    processes = []
#    for rank in range(num_processes):
#        p = mp.Process(target=run_process, args=(args,B_share_model,W_share_model,board_max,rank))
#        p.start()
#        processes.append(p)
#    for p in processes:
#        p.join()