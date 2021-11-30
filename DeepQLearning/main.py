from typing import AsyncGenerator
import pygame 
import random
from car_simulation import Car
from dqn import Agent
import numpy as np

WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 60    
CAR_SIZE_Y = 60

BORDER_COLOR = (255, 255, 255, 255) # Color To Crash on Hit

current_generation = 0 # Generation counter


def train():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    game_map = pygame.image.load('map.png').convert() # Convert Speeds Up A Lot
    clock = pygame.time.Clock()

    agent = Agent(gamma=0.99, epsilson=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[5], lr = 0.01)
    scores, eps_history = [], []
    n_games = 1000

    for i in range(n_games):
        car = Car()
        done = False
        score = 0
        observation = car.get_data()
        while not done:
            action = agent.choose_action(observation)
            if action == 0:
                car.angle += 10 # Left
            elif action == 1:
                car.angle -= 10 # Right
            elif action == 2:
                if(car.speed - 2 >= 12):
                    car.speed -= 2 # Slow Down
            else:
                car.speed += 2 # Speed Up

            screen.blit(game_map, (0, 0))
            car.update(game_map)
            car.draw(screen)
            pygame.display.flip()
            clock.tick(30)

            observation_, reward, done = car.get_data(), car.get_reward(), not car.is_alive()
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])

        print(f'episode: {i}, score = {round(score,2)}, epsilon= {round(agent.epsilon,3)}, avg_score = {round(avg_score,2)}')


def random_simulation():

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    game_map = pygame.image.load('map.png').convert() # Convert Speeds Up A Lot
    clock = pygame.time.Clock()

    n_games = 1000

    for i in range(n_games):
        car = Car()
        done = False
        score = 0
        observation = car.get_data()
        while not done:
            action = random.choice([0,1,2,3])
            if action == 0:
                car.angle += 10 # Left
            elif action == 1:
                car.angle -= 10 # Right
            elif action == 2:
                if(car.speed - 2 >= 12):
                    car.speed -= 2 # Slow Down
            else:
                car.speed += 2 # Speed Up

            screen.blit(game_map, (0, 0))
            car.update(game_map)
            car.draw(screen)
            pygame.display.flip()
            clock.tick(30)
            done = not car.is_alive()

if __name__ == '__main__':
    train()
    # random_simulation()
