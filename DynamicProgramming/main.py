from numpy.core.numeric import Inf
from numpy.lib.shape_base import tile
import pygame
import value_iteration
import time 
import numpy as np

WHITE = (240, 240, 240)
BLACK = (0, 0, 0)
border_color = 'black'

tile_size = 75
triangular_size = 7
BOARD_SIZE_PX = 400
BOARD_X_OFFSET_PX = 50
BOARD_Y_OFFSET_PX = 50
TEXT_X_OFFSET_PX = 19
TEXT_Y_OFFSET_PX = 26
LABEL_POS = (10, 10)

Methods = ['Value Iteration', 'Policy Iteration']

class FindPath:
    def __init__(self, rows, columns, obstacles, credits):
        pygame.init()
        pygame.display.set_caption("FindPath")
        self.rows = rows
        self.cols  = columns
        self.SURFACE_WIDTH = columns*tile_size+BOARD_X_OFFSET_PX*2
        self.SURFACE_HEIGHT = rows*tile_size+BOARD_Y_OFFSET_PX*2
        self.surface = pygame.display.set_mode((self.SURFACE_WIDTH, self.SURFACE_HEIGHT))
        self.myfont = pygame.font.SysFont("arial", 15)
        self.labelfont = pygame.font.SysFont("arial", 20)
        self.ob = obstacles
        self.credit = credits
    
    def search(self):
        '''
        Using Value iteration or Policy Iteration to update value and policy
        '''
        solver = value_iteration.value_iteration(self.rows, self.cols, self.ob, self.credit)
        utility = np.zeros((self.rows,self.cols))
        condition = True
        changeCredit = True
        while condition:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        changeCredit = not changeCredit
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    j = (x - BOARD_X_OFFSET_PX)//tile_size
                    i = (y - BOARD_Y_OFFSET_PX)//tile_size
                    if i < 0 or i > self.rows or j < 0 or j > self.cols:
                        break
                    if changeCredit:
                        if (i,j) in self.credit:
                            self.credit.remove((i,j))
                        else:
                            self.credit.append((i,j))
                    else:
                        if (i,j) in self.ob:
                            self.ob.remove((i,j))
                        else:
                            self.ob.append((i,j))
                    solver = value_iteration.value_iteration(self.rows, self.cols, self.ob, self.credit)
            next_utility = solver.get_next_utility(utility)
            self.print_matrix(utility = next_utility, wait = 0.05, changecredit=changeCredit)
            utility = next_utility
            

    def print_matrix(self, utility=None, wait=False, changecredit=True, method = 0):
        self.surface.fill(WHITE)
        label = self.labelfont.render(Methods[method], 1, BLACK)
        self.surface.blit(label, LABEL_POS)
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                ul_x_px = j*tile_size + BOARD_X_OFFSET_PX
                ul_y_px = i*tile_size + BOARD_Y_OFFSET_PX
                #print(ul_x_px,ul_y_px)
                if (i,j) in self.ob:
                    pygame.draw.rect(self.surface, 'gray', (ul_x_px, ul_y_px, tile_size, tile_size))
                elif (i,j) in self.credit:
                    pygame.draw.rect(self.surface, 'pink', (ul_x_px, ul_y_px, tile_size, tile_size))
                else:
                    pygame.draw.rect(self.surface, 'white', (ul_x_px, ul_y_px, tile_size, tile_size))
                pygame.draw.rect(self.surface, border_color, (ul_x_px, ul_y_px, tile_size, tile_size),1)
                if changecredit:
                    text = self.myfont.render('[credit]', 1, BLACK)
                else:
                    text = self.myfont.render('[obstacle]', 1, BLACK)
                self.surface.blit(text, (self.SURFACE_WIDTH-80, 15))
                if utility is not None:
                    tile_utility = self.myfont.render(str(round(utility[i][j],2)), 1, BLACK)
                    tile_x = ul_x_px + TEXT_X_OFFSET_PX
                    tile_y = ul_y_px + TEXT_Y_OFFSET_PX
                    pos = (tile_x, tile_y)
                    self.surface.blit(tile_utility, pos)
        self.print_direction(utility)
        pygame.display.flip()
        if wait:
            time.sleep(wait)
    
    def print_direction(self,utility):
        '''
        print direction of policy for each tile according to the utility
        '''
        for i in range(self.rows):
            for j in range(self.cols):
                ul_x_px = j*tile_size + BOARD_X_OFFSET_PX
                ul_y_px = i*tile_size + BOARD_Y_OFFSET_PX
                U = np.zeros(4)
                states = [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
                for n in range(len(states)):
                    (i_, j_) = states[n]
                    if 0 <= i_ < self.rows and 0 <= j_ < self.cols:
                        U[n] = utility[i_][j_]
                maxu = max(U)
                idxs = np.where(np.array(U)==maxu)[0]
                POS = self.get_triangular_pos(ul_x_px,ul_y_px)
                for idx in idxs:
                    pygame.draw.polygon(self.surface, BLACK, points=POS[idx])

    def get_triangular_pos(self,x,y):
        #up
        pos_up = [(x+tile_size//2-triangular_size,y+5+triangular_size*1.7), (x+tile_size//2, y+5), (x+tile_size//2+triangular_size,y+5+triangular_size*1.7)]
        #down
        pos_down = [(x+tile_size//2-triangular_size,y+tile_size-5-triangular_size*1.7), (x+tile_size//2, y+tile_size-5), (x+tile_size//2+triangular_size,y+tile_size-5-triangular_size*1.7)]
        #left
        pos_left = [(x+5+triangular_size*1.7,y+tile_size//2-triangular_size), (x+5, y+tile_size//2), (x+5+triangular_size*1.7,y+tile_size//2+triangular_size)]
        #right
        pos_right = [(x+tile_size-5-triangular_size*1.7,y+tile_size//2-triangular_size), (x+tile_size-5, y+tile_size//2), (x+tile_size-5-triangular_size*1.7,y+tile_size//2+triangular_size)]
        Pos = [pos_up, pos_down, pos_left, pos_right]
        return Pos
    


    


if __name__ == '__main__':
    solver = FindPath(6,5,[(1,1),(1,3)],[(1,2)])
    solver.search()

