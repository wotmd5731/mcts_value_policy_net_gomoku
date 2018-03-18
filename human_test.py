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
    
def run_process(args,share_model,board_max,n_rows,rank):
    
    from checkerboard import Checkerboard, BoardRender
    board = Checkerboard(board_max,n_rows)
    board_render = BoardRender(board_max,render_off=False,inline_draw=True)
    board_render.clear()
    board_render.draw(board.states)
    
    for episode in range(1):
        random.seed(time.time())
        board.reset()
        board_render.clear()
        
        """ start a self-play game using a MCTS player, reuse the search tree
        store the self-play data: (state, mcts_probs, z)
        """
        p1, p2 = board.players
        player = input('select player 1: balck , 2 : white')
        if player=='1':
            play_step = 0
        else:
            play_step = 1
        for step in range(10000):
            if step %2 == play_step:
                ss =input('input x,y:')
                pos = ss.split(',')
                if pos == 'q' :
                    return
                move = int(pos[0])+int(pos[1])*board_max
                print('movd ',move)
            else:            
                move, move_probs = agent.get_action(board, temp=1.0, return_prob=1)
            board.step(move)
            board_render.draw(board.states)
            end, winner = board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                agent.reset_player() 
                if winner != -1:
                    print("Game end. Winner is player:", winner)
                else:
                    print("Game end. Tie")
#                return winner, zip(states, mcts_probs, winners_z)
                break

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

#    
#    from model import PV_NET
#    share_model = PV_NET(args)
#    share_model.share_memory()
#    if os.path.exists('B'+args.name):
#        print('load')
#        B_share_model.load_state_dict(torch.load('B'+args.name))
#        W_share_model.load_state_dict(torch.load('W'+args.name))
    n_rows = 4
    from agent import Agent_MCTS
    agent = Agent_MCTS(args,0,0,board_max,param='./net_param',is_selfplay=False)
    try:
        run_process(args,agent,board_max,n_rows,999)
    except:
        print('except save')
#        agent.save()
        
#    num_processes = 8
#    processes = []
#    for rank in range(num_processes):
#        p = mp.Process(target=run_process, args=(args,B_share_model,W_share_model,board_max,rank))
#        p.start()
#        processes.append(p)
#    for p in processes:
#        p.join()