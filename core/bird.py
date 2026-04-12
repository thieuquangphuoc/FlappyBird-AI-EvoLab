import numpy as np
import random
import tkinter as tk
import config

from ai.neural_network import NeuralNetwork
from typing import List, Tuple, Optional, Any

class BirdData:
    def __init__(self, brain: Optional['NeuralNetwork'] = None) -> None:
        self.x = config.WIDTH_GAME_SCREEN // 4
        self.y = config.HEIGHT_GAME_SCREEN // 2
        
        self.velocity = 0.0
        self.width = config.BIRD_SIZE
        self.height = config.BIRD_SIZE
        
        self.alive = True
        self.fitness = 0
        self.closest_pipe: Optional[Any] = None 
        
        # brain of agent using neural network
        if brain is not None:
            self.brain = brain.copy()
        else:
            self.brain = NeuralNetwork()
        
        # function that randomly generates colors for 100 agents
        self.agent_color = generate_random_agent_color()

def jump_bird(bird: BirdData) -> None:
    bird.velocity = config.JUMP_STRENGTH

def update_bird(bird: BirdData) -> None:
    bird.velocity += config.GRAVITY
    bird.y += bird.velocity
    bird.fitness += 1

def think_bird(bird: BirdData, pipes: List[Any]) -> None:
    if not bird.alive:
        return
        
    closest_dist = float('inf')
    bird.closest_pipe = None

    # find the closest pipe (bird has not passed through)
    for pipe in pipes:
        if pipe.x + pipe.width > bird.x:
            dist = pipe.x - bird.x
            if dist < closest_dist:
                closest_dist = dist
                bird.closest_pipe = pipe

    if bird.closest_pipe is None:
        return

    # 4 inputs: horziontal, vertical top, vertical bottom, and velocity respectively
    # compressing the 4 inputs to [0,1] to save it to array later
    input_1 = closest_dist / float(config.WIDTH_GAME_SCREEN)
    input_2 = (bird.closest_pipe.top_height - bird.y) / float(config.HEIGHT_GAME_SCREEN)
    input_3 = (bird.closest_pipe.bottom_y - bird.y) / float(config.HEIGHT_GAME_SCREEN)
    input_4 = bird.velocity / 10.0

    # matrix 1x4 
    inputs = np.array([input_1, input_2, input_3, input_4])

    # this is the weight layer 1, and later will go through ReLU activation function
    output = bird.brain.feed_forward(inputs)

    decision = output[0]
    if decision > 0.5:
        jump_bird(bird)

def get_bird_bbox(bird: BirdData) -> Tuple[float, float, float, float]:
    return (bird.x, bird.y, bird.width, bird.height)

def draw_bird(bird: BirdData, canvas: tk.Canvas) -> None:
    # draw agents' vision
    if bird.alive and bird.closest_pipe is not None:
        eye_x = bird.x + bird.width
        eye_y = bird.y + bird.height / 2.0
        target_x = bird.closest_pipe.x

        canvas.create_line(eye_x, eye_y, target_x, bird.closest_pipe.top_height, fill=config.COLOR_LASER, dash=(2,2))
        canvas.create_line(eye_x, eye_y, target_x, eye_y, fill=config.COLOR_LASER, dash=(2,2))
        canvas.create_line(eye_x, eye_y, target_x, bird.closest_pipe.bottom_y, fill=config.COLOR_LASER, dash=(2,2))

    # draw body of the agents
    canvas.create_rectangle(
        bird.x, bird.y, 
        bird.x + bird.width, bird.y + bird.height,
        fill=bird.agent_color, outline="black", width=2
    )

def generate_random_agent_color() -> str:
    r_agent = lambda: random.randint(0, 255)
    return "#%02x%02x%02x" % (r_agent(), r_agent(), r_agent())
        


    

