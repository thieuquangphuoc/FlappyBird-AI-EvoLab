import tkinter as tk
import time
import config

from core.bird import get_bird_bbox, draw_bird
from core.collision import check_aabb
from core.ground import GroundData, get_ground_bbox, draw_ground
from core.pipe import setup_pipe, update_pipe, get_pipe_bboxes, draw_pipe

from ai.population import Population
from ai.neural_network import NeuralNetwork

from typing import List, Dict
from datetime import datetime, timezone

class GameEngine:
    def __init__(self, canvas: tk.Canvas) -> None:
        self.canvas = canvas
        
        # include solely data, not object
        self.ground_data = GroundData()
        self.population = Population()
        self.pipes_data = [] 

        # game state
        self.score = 0
        self.max_score = 0
        self.generation = 1

        # save data to export to csv file
        self.history_data: List[Dict[str, float]] = []

        # save elite brain
        self.best_all_time_brain = None
        self.demo_brain = None
        self.last_gen_best_brain = None

        # tracking real-time
        self.app_start_time = time.time()
        self.total_time_to_max_score = 0.0
        self.best_generation = 1
        self.best_gen_score = 0
        self.record_achieved_time = None
        self.record_achieved_utc = None
        self.generation_start_time = time.time()

        self.pending_draws = []

    def reset_training(self) -> None:
        # restore everything to start training AI from gen 1
        self.population = Population() 

        self.best_all_time_brain = None
        self.best_generation = 1
        self.max_score = 0
        self.history_data = []

        self.reset_game(do_evolution = False)

    def reset_game(self, do_evolution = True) -> None:
        # reset game environment when every agents died
        if self.demo_brain is not None:
            for bird_data in self.population.birds:
                bird_data.x = config.WIDTH_GAME_SCREEN // 4
                bird_data.y = config.HEIGHT_GAME_SCREEN // 2
                bird_data.velocity = 0.0
                bird_data.closest_pipe = None
                bird_data.brain = self.demo_brain.copy()
                bird_data.alive = True

        # if it is in training mode, crossover with new generation
        elif do_evolution:
            self.population.next_generation()

        # clear score and pipe
        self.pipes_data.clear()
        self.score = 0
        self.best_gen_score = 0
        self.generation_start_time = time.time()

    def check_collisions(self):
        # get bounding box of the ground
        ground_box = get_ground_bbox(self.ground_data)

        for bird_data in self.population.birds:
            if not bird_data.alive:
                continue

            # get bounding box of the bird
            bird_box = get_bird_bbox(bird_data)

            # case 1: check whether it touches the ground or go beyond the height of screen
            if check_aabb(bird_box, ground_box) or bird_data.y < 0:
                bird_data.alive = False
                continue

            # case 2: check whether it collides with the pipes
            for p_data in self.pipes_data:
                top_box, bottom_box = get_pipe_bboxes(p_data)
                if check_aabb(bird_box, top_box) or check_aabb(bird_box, bottom_box):
                    bird_data.alive = False
                    break
                   
    # game logic: rendering 60fps, seperating totally with UI rendering
    def update_logic(self) -> None:
        self.population.update(self.pipes_data)
        for p_data in self.pipes_data:
            update_pipe(p_data)
        
        # spawn new pipe if the old one goes beyond the width
        if len(self.pipes_data) == 0 or self.pipes_data[-1].x < config.WIDTH_GAME_SCREEN - 200:
            new_pipe_data = setup_pipe(config.WIDTH_GAME_SCREEN)
            self.pipes_data.append(new_pipe_data)

        # delete the pipe that goes beyond the width
        if len(self.pipes_data) > 0 and self.pipes_data[0].x + self.pipes_data[0].width < 0:
            self.pipes_data.pop(0)

        # caluclation part
        bird_x_ref = config.WIDTH_GAME_SCREEN // 4
        for p_data in self.pipes_data:
            # if bird passed through the co-ordiante x of pipe
            if not p_data.passed and p_data.x + p_data.width < bird_x_ref:
                self.score += 1
                p_data.passed = True

                # update record
                if self.score > self.best_gen_score:
                    self.best_gen_score = self.score
                    
                if self.score > self.max_score:
                    self.max_score = self.score
                    self.best_generation = self.population.generation
                    elapsed = time.time() - self.app_start_time
                    self.total_time_to_max_score = round(elapsed, 2)

                    self.record_achieved_time = datetime.now()
                    self.record_achieved_utc = datetime.now(timezone.utc)

                    # save data (weights) of the best agent
                    for b_data in self.population.birds:
                        if b_data.alive:
                            self.best_all_time_brain = b_data.brain.copy()
                            break
        
        self.check_collisions()

        if self.population.is_extinct():
            # tracking time
            lifespan_seconds = round(time.time() - self.generation_start_time, 2)
            lifespan_minutes = round(lifespan_seconds / 60, 2)

            now = datetime.now()
            utc_now = datetime.now(timezone.utc)

            # calculate fitness for data visualization
            fitness_list = [b.fitness for b in self.population.birds]
            max_fit = max(fitness_list) if fitness_list else 0
            avg_fit = sum(fitness_list) / len(fitness_list) if fitness_list else 0

            # export data to csv file
            generation_info = {
                "Generation": self.population.generation,
                "Gen Best Score": self.best_gen_score,
                "All Time Max": self.max_score,
                "Max Fitness": max_fit, 
                "Avg Fitness": avg_fit,
                "Lifespan (Seconds)": lifespan_seconds,
                "Lifespan (Minutes)": lifespan_minutes,
                "DOB": now.strftime("%d/%m/%Y"),
                "UTC-TimeStamp": utc_now.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            if self.population.birds:
                best_bird_this_gen = max(self.population.birds, key=lambda b: getattr(b, 'fitness', 0))
                self.last_gen_best_brain = best_bird_this_gen.brain.copy()

            self.history_data.append(generation_info)
            self.reset_game()

    # UI rendering
    def draw_ui(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, config.WIDTH_GAME_SCREEN, config.HEIGHT_GAME_SCREEN, fill=config.BG_GAME)
        
        # spawn pipes
        for p_data in self.pipes_data:
            draw_pipe(p_data, self.canvas)
        
        # spawn birds (only alive ones to optimize)
        alive_birds = [bird_data for bird_data in self.population.birds if bird_data.alive]
        for bird_data in alive_birds:
            draw_bird(bird_data, self.canvas)

        # spawn ground
        draw_ground(self.ground_data, self.canvas)

        # spawn real-time score
        self.canvas.create_text(
            config.WIDTH_GAME_SCREEN // 2, 50,
            text = f"{self.score}", font=("Helvetica", 40, "bold"),
            fill=config.COLOR_INFO_UI_GAME
        )

        self.canvas.create_text(
            config.WIDTH_GAME_SCREEN // 2, 100,
            text = (f"Gen: {self.population.generation} | Alive: {self.population.get_alive_count()}"),
            font=("Helvetica", 20, "bold"),
            fill=config.COLOR_INFO_UI_GAME
        )

        self.canvas.create_text(
            config.WIDTH_GAME_SCREEN // 2, 130,
            text = (f"Max Score: {self.max_score}"),
            font=("Helvetica", 17, "bold"),
            fill=config.COLOR_INFO_UI_GAME
        )