import config
import tkinter as tk
from typing import Tuple


class GroundData:
    def __init__(self):
        self.x = 0
        self.y = config.HEIGHT_GAME_SCREEN - config.GROUND_HEIGHT
        self.width = config.WIDTH_GAME_SCREEN
        self.height = config.GROUND_HEIGHT

def get_ground_bbox(ground: GroundData) -> Tuple[float, float, float, float]:
    # return co-ordinates of collision with ground
    return (ground.x, ground.y, ground.width, ground.height)

def draw_ground(ground: GroundData, canvas: tk.Canvas) -> None:
    # draw ground
    canvas.create_rectangle(ground.x, ground.y,
                            ground.x + ground.width, ground.y + ground.height, 
                            fill=config.COLOR_GROUND, 
                            outline=config.COLOR_PIPE_OUTLINE, width=2)