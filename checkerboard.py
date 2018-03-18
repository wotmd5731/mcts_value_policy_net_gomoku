# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 20:29:26 2018

@author: JAE
"""
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display


class Checkerboard():
    empty = 0
    black = 1
    white = 2
    block = 3
    
    def __init__(self,max_size,n_in_row):
        self.width = max_size
        self.height = max_size
        
        self.max_size = max_size
#        self.board = [[ self.empty ]*max_size for i in range(max_size)]  
        self.states= {}
        self.players = [1, 2]
        self.current_player = self.players[0]
        self.availables = list(range(self.max_size * self.max_size))
        self.last_move = -1
        self.n_in_row =n_in_row # need how many pieces in a row to win
        
        
    def reset(self, start_player=0):
        self.states= {}
        self.current_player = self.players[start_player]  # start player        
        self.availables = list(range(self.max_size * self.max_size))
        self.last_move = -1
#        for y in range(self.max_size):
#            for x in range(self.max_size):
#                self.board[y][x] = 0
#        ss = torch.LongTensor(self.board)
    
    
    def __repr__(self):
        return "info  max_size %d " % (self.max_size)

    def __str__(self):
#        for i in reversed(range(self.max_size)):
        for i in range(self.max_size):
            print(i,self.board[i])
        return '-----end-----'
    
    def current_state(self): 
        """return the board state from the perspective of the current player
        shape: 4*width*height"""
        
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]                           
            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0   
            square_state[2][self.last_move //self.width, self.last_move % self.height] = 1.0 # last move indication   
        
        square_state[3][:,:] = self.current_player
            
        return square_state[:,::-1,:]
    

    def _check_rec(self, x, y, dx, dy , stone):
        if x<0 or x>=self.max_size or y<0 or y>=self.max_size or self.get_xy(x,y)!=stone:
            return 0
        #board data == stone   . next rec 
        return self._check_rec(x+dx,y+dy,dx,dy,stone) + 1
    
    def _check_done(self,x,y,stone):
        'stone으로 들어온게 5개 만들면 끝 과 리워드 +1 '
        '아니면 리턴 0 '
        delta = [[1,0],[0,1],[1,1],[1,-1]]
        max_ret = 0
        reward = 0
        done= 0
        for dx,dy in delta:
            ret = self._check_rec(x+dx,y+dy,dx,dy,stone) + self._check_rec(x-dx,y-dy,-dx,-dy,stone) + 1
            max_ret = max(max_ret,ret)
        
        # 5개 완성 시 리턴 1
        if max_ret == 5:
#            reward = 1
            reward = 1
            done = 1
#            self.next_done_flag = 1
#            print("create 5connection by me ! **** you win ****")
        #6개 완성시 자동 패배
        elif max_ret == 6:
            reward = -1
            done = 1
#            self.next_done_flag = 2
        return reward , done

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if(len(moved) < self.n_in_row + 2):
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):#            
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player



    def step(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        
        self.last_move = move
        x = int(move%self.max_size)
        y = int(move/self.max_size)

        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1] 
#         = torch.LongTensor(self.board)
#        rr,dd = self._check_done(x,y,stone)
            
        
    def get_xy(self, x ,y ):
        return self.board[y][x]
    
    def get_random_xy_flat(self):
        return random.choice(self.availables)
        
#        x, y =self.get_random_xy()
#        return x+y*self.max_size
        
    def get_random_xy(self):
        x = random.choice(self.availables)%self.max_size
        y = random.choice(self.availables)//self.max_size
        
        return x,y
#        x,y = random.randint(0,self.max_size-1),random.randint(0,self.max_size-1)
#        while not self.get_xy(x,y)==self.empty:
#            x,y = random.randint(0,self.max_size-1),random.randint(0,self.max_size-1)
#        return x,y
    

    

class BoardRender():

    def __init__(self, max_size,render_off=False,inline_draw = False):
        self.render_off = render_off
        if self.render_off:return
        self.inline_draw = inline_draw
        self.max_size = max_size
#        if inline_draw:
        self.fig = plt.figure(figsize=(2,2))
        self.ax = self.fig.add_subplot(1,1,1)
        self.ax.set(xlim=[-1, max_size], ylim=[-1, max_size], title='Example', xlabel='xAxis', ylabel='yAxis')
        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(0, max_size+1, 1)
#        minor_ticks = np.arange(0, max_size+1, 1)
        self.ax.set_xticks(major_ticks)
#        self.ax.set_xticks(minor_ticks, minor=True)
        self.ax.set_yticks(major_ticks)
#        self.ax.set_yticks(minor_ticks, minor=True)
        # And a corresponding grid
        self.ax.grid(which='both')
        # Or if you want different settings for the grids:
#        self.ax.grid(which='minor', alpha=0.5)
        self.ax.grid(which='major', alpha=0.5)
        self.ax.grid(color='black', linestyle='-', linewidth=0.5)
#        plt.xticks(range(11))
#        plt.show()
    def clear(self):
        if self.render_off:return
        self.ax.patches.clear()
        
    def draw(self,board_states):
        if self.render_off:return
        for key , val in board_states.items():
            self.set_xy_draw(int(key%self.max_size),int(key//self.max_size),val)
#        plt.clf()
#        self.ax.draw()
#        self.ax.figure.canvas.draw()
#        self.fig.canvas.draw()
#        self.fig.update()
#        self.ax.update()
#        plt.draw()
#        self.fig.clf()
#        plt.show()
        # ipython command 
        if self.inline_draw:
            display(self.fig)
        else :
            plt.pause(0.001)            
        print(' ')
        
        pass
    


    def set_xy_draw(self, x, y, stone):
#        self.board[y][x] = stone
        if stone == 1:
            ax_stone = patches.Circle((x, y), 0.5, facecolor='black', edgecolor='black',linewidth=1)
            self.ax.add_patch(ax_stone)
        elif stone == 2:
            ax_stone = patches.Circle((x, y), 0.5, facecolor='white', edgecolor='black',linewidth=1)
            self.ax.add_patch(ax_stone)
        elif stone == 3:
            ax_stone = patches.Circle((x, y), 0.5, facecolor='blue', edgecolor='black',linewidth=1)
            self.ax.add_patch(ax_stone)
        pass
        

if __name__=="__main__":
    board = Checkerboard(10,inline_draw = True)
#    board = Checkerboard(10)
    board.reset()
    for i in range(100):
        "x,y = black agent .get_action(state)"
        mv = board.get_random_xy_flat()
        board.step(mv)
        print(board.current_state(board.players[0]))
        board.draw()
        print(board.game_end())
        
#        "x,y = white agent .get_action(state)"
#        mv = board.get_random_xy_flat()
#        ss_ , rr, dd,winner = board.step(mv)
#        board.draw()
#        if dd:
#            print("winner : ",winner)
#            break
    


    
    
    
    