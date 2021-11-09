from collections import deque
import queue
import heapq
from networkx.readwrite import edgelist
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
'''
simple case
given adjacent list and heuristic function, find the optimal path.
'''


class PriorityQueue(object):
    def __init__(self):
        self._queue = []        #创建一个空列表用于存放队列
        self._index = 0        #指针用于记录push的次序
    
    def push(self, node):
        """队列由（priority, index, item)形式的元祖构成"""
        heapq.heappush(self._queue, (node.cost, self._index, node.name)) 
        self._index += 1
        
    def pop(self):
        return heapq.heappop(self._queue)[-1]    #返回拥有最高优先级的项
    
    def is_empty(self):
        if len(self._queue) == 0:
            return True
        return False

    def have(self, node):
        q = [i[2] for i in self._queue]
        if node in q:
            return True
        return False
    
    def get_score(self, node):
        q = [i[2] for i in self._queue]
        return self._queue[q.index(node)][0]
    
    def change_score(self, node, cost):
        q = [i[2] for i in self._queue]
        idx = self._queue[q.index(node)][1]
        self._queue[q.index(node)] = (cost,idx,node)


class node(object):
    def __init__(self, name, cost):
        self.name = name
        self.cost = cost

    def __repr__(self):
        return f'node_name={self.name}, node_cost={self.cost}'


class Astar:
    def __init__(self,adjacent_list, heuristic_function,show_animation=True):
        self.adj = adjacent_list
        self.h = heuristic_function
        self.show_animation = show_animation
    
    def search(self,s,e):
        came_from = dict()

        explored = set([])
        frontier = PriorityQueue()
        n = node(s, self.h[s])
        frontier.push(n)
        if self.show_animation:
            self.init_graph(s,e)
            plt.pause(0.5)
        while True:
            if frontier.is_empty():
                print('failed')
                return None
            n = frontier.pop()
            if n == e:
                path = self.reconstruct_path(came_from, s, e)
                print('find path: ', path)
                if self.show_animation:
                    plt.clf()
                    self.init_graph(s,e)
                    self.update_graph(path)
                    plt.annotate(f'Find the optimal path: {path}', xy=(0.35, 1.05), xycoords='axes fraction',color='black')
                    plt.pause(0)
                return self.reconstruct_path(came_from, s, e)
            explored.add(n)
            for neighbor, cost in self.adj[n]:
                total_cost = self.h[neighbor]+cost #G+H
                if neighbor not in explored and frontier.have(neighbor) == False:
                    frontier.push(node(neighbor,total_cost))
                    came_from[neighbor] = n
                    if self.show_animation:
                        plt.clf()
                        self.init_graph(s,e)
                        self.update_graph([(n,neighbor)])
                        plt.annotate(f'The state of {neighbor}: G + H = {cost}+{self.h[neighbor]} = {total_cost}', xy=(0.35, 1.05), xycoords='axes fraction',color='black')
                        plt.pause(0.5)
                elif frontier.have(neighbor):
                    if frontier.get_score(neighbor) > total_cost:
                        frontier.change_score(neighbor,total_cost)
                        came_from[neighbor] = n
                        if self.show_animation:
                            plt.clf()
                            self.init_graph(s,e)
                            self.update_graph([(n,neighbor)])
                            plt.annotate(f'update {neighbor} state: G + H = {cost}+{self.h[neighbor]} = {total_cost}', xy=(0.35, 1.05), xycoords='axes fraction',color='black')
                            plt.pause(0.5)
                        
    
    def init_graph(self,s,e):
        self.G = nx.Graph()
        self.nodes = list(self.adj.keys())
        self.labels = {}
        for node in self.nodes:
            self.labels[node] = node
            for neighbor, cost in self.adj[node]:
                self.G.add_edge(node,neighbor,weight=cost)
        self.pos = nx.spring_layout(self.G,seed=0)
        nx.draw_networkx_nodes(self.G,self.pos,nodelist=[s,e],node_color="tab:red")
        nx.draw_networkx_nodes(self.G,self.pos,nodelist=[v for v in self.nodes if v != s and v!= e],node_color="tab:gray")
        nx.draw_networkx_labels(self.G,self.pos,self.labels,font_size=12,font_color='whitesmoke')
        nx.draw_networkx_edges(self.G,self.pos,width=1.0,alpha=0.5)
    
    def update_graph(self,lst):
        edgelst = []
        if len(lst)!=1:
            for i in range(len(lst)-1):
                edgelst.append((lst[i],lst[i+1]))
        else:
            edgelst = lst
        nx.draw_networkx_edges(
            self.G,
            self.pos,
            edgelist=edgelst,
            width=8,
            alpha=0.5,
            edge_color="tab:red",
        )


    def reconstruct_path(self,came_from, s, e):
        path = [e]
        before = came_from[e]
        path.append(before)
        while before != s:
            before = came_from[before]
            path.append(before)
        return path[::-1]
        


if __name__ == '__main__':
    # frontier = PriorityQueue()
    # a = node('a',1)
    # s = node('s',5)
    # v = node('a',10)
    # frontier.push(s)
    # frontier.push(a)
    # frontier.push(v)
    # print(frontier.get_score('s'))
    adjacent_list = {
        'A': [('B', 3),('C', 5),('D', 2)],
        'B': [('A', 3),('C', 4),('E', 6)],
        'C': [('B', 4),('A', 5),('D', 2),('F',7)],
        'D': [('A', 2),('C', 2),('G', 8)],
        'E': [('B', 6),('F', 1),('H', 5)],
        'F': [('C', 7),('E', 1),('H', 1),('G',5)],
        'G': [('D', 8),('F', 5),('H', 6)],
        'H': [('E', 5),('F', 1),('G', 6)],
    }
    heuristic_function = {
        'A': 6,
        'B': 4,
        'C': 4,
        'D': 4,
        'E': 1,
        'F': 1,
        'G': 1,
        'H': 0
    }
    solver = Astar(adjacent_list, heuristic_function)
    solver.search('A','H')
