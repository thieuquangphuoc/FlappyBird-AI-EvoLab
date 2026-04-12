import config
import random
import tkinter as tk
from typing import Tuple


class PipeData:
    def __init__(self):
        self.x = 0.0
        self.width = config.PIPE_WIDTH
        self.passed = False
        self.top_height = 0
        self.bottom_y = 0
        self.bottom_height = 0

def setup_pipe(start_x: float) -> PipeData:
    pipe = PipeData()
    pipe.x = start_x
    pipe.passed = False

    available_height = config.HEIGHT_GAME_SCREEN - config.GROUND_HEIGHT
    
    min_top_height = 50
    max_top_height = available_height - config.PIPE_GAP - 50

    # random height of top pipe
    pipe.top_height = random.randint(min_top_height, max_top_height)

    # calculate height of bottom pipe
    pipe.bottom_y = config.PIPE_GAP + pipe.top_height
    pipe.bottom_height = available_height - pipe.bottom_y
    
    return pipe

def update_pipe(pipe: PipeData) -> None:
    # pipe moving to the left 
    pipe.x -= config.PIPE_SPEED

def get_pipe_bboxes(pipe: PipeData) -> Tuple[Tuple[float, float, float, float], 
                                             Tuple[float, float, float, float]]:
    # get bouding box of both top and bottom pipe
    top_box = (pipe.x, 0, pipe.width, pipe.top_height)
    bottom_box = (pipe.x, pipe.bottom_y, pipe.width, pipe.bottom_height)

    return top_box, bottom_box

def draw_pipe(pipe: PipeData, canvas: tk.Canvas) -> None:
    # draw top pipe
    canvas.create_rectangle(pipe.x, 0, 
                            pipe.x + pipe.width, pipe.top_height, 
                            fill=config.COLOR_PIPE_FILL, outline=config.COLOR_PIPE_OUTLINE, width=2)
    # draw bottom pipe
    canvas.create_rectangle(pipe.x, pipe.bottom_y, 
                            pipe.x + pipe.width, pipe.bottom_y + pipe.bottom_height,
                            fill=config.COLOR_PIPE_FILL, outline=config.COLOR_PIPE_OUTLINE, width=2)