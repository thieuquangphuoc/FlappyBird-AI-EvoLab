import tkinter as tk
import tkinter.font as tkFont
import config

import time
import threading

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from engine.game_engine import GameEngine
from data_manager import DataManager
from controller import UIController
from tkinter import ttk


class SimulationAILab:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Flappy Bird AI Evolution Lab")
        self.root.geometry(f"{config.WIDTH_MAIN_SCREEN}x{config.HEIGHT_MAIN_SCREEN}")
        self.root.resizable(False, False)
        self.root.configure(bg="#0d1117")
        root.attributes('-topmost', True)
        
        self.ctrl = UIController(self)
        self.data_mgr = DataManager()

        self.speed_multiplier = 1
        self.is_paused = False
        self.last_chart_gen_count = 0

        self.ui_frame_skip = 0  
        self.chart_update_counter = 0

        self.dialog_open = False

        self.font_title = tkFont.Font(family="Segoe UI", size=13, weight="bold")
        self.font_label = tkFont.Font(family="Segoe UI", size=10)
        self.font_value = tkFont.Font(family="Consolas", size=14, weight="bold")
        self.font_button = tkFont.Font(family="Segoe UI", size=10, weight="bold")
        self.font_small = tkFont.Font(family="Consolas", size=9)

        # spawn lock to control data
        # UI and game cannot run simultaneous or else crash
        self.game_thread_running = True
        self.game_lock = threading.Lock()

        self.setup_styles()
        self.build_ui()

        # daemon thread -> running in background
        self.game_thread = threading.Thread(target=self.game_loop_thread, daemon=True)
        self.game_thread.start()

        # main thread
        self.ui_loop()
    
    
    def setup_styles(self) -> None:
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Dark.TNotebook', background='#0d1117', borderwidth=0)
        style.configure('Dark.TFrame', background='#0d1117')

        style.configure('Dark.TNotebook.Tab', 
                        background='#161b22', 
                        foreground='#8b949e',
                        padding=[16, 6], 
                        font=('Segoe UI', 10, 'bold'),
                        borderwidth=0)

        style.map('Dark.TNotebook.Tab',
                  background=[('selected', '#e94560'), ('!selected', '#161b22')],
                  foreground=[('selected', '#ffffff'), ('!selected', '#8b949e')],
                  expand=[('selected', [2, 6, 2, 0]), ('!selected', [0, 0, 0, 0])]
                 )

    def build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1) 
        self.root.rowconfigure(0, weight=1)   
        self.root.rowconfigure(1, weight=0)   

        game_outer = tk.Frame(self.root, bg="#30363d", padx=2, pady=2) 
        game_outer.grid(row=0, column=0, padx=(8, 4), pady=(8, 4), sticky="ns")

        self.canvas = tk.Canvas(game_outer, width=config.WIDTH_GAME_SCREEN, 
                                height=config.HEIGHT_GAME_SCREEN, highlightthickness=0, bg="#0d1117")
        self.canvas.pack()
        self.engine = GameEngine(self.canvas)

        viz_outer = tk.Frame(self.root, bg="#30363d", padx=2, pady=2)
        viz_outer.grid(row=0, column=1, padx=(4, 8), pady=(8, 4), sticky="nsew")

        self.notebook = ttk.Notebook(viz_outer, style='Dark.TNotebook')
        self.notebook.pack(expand=True, fill="both")

        self.build_charts_tab()
        self.build_research_tab()
        
        stats_outer = tk.Frame(self.root, bg="#30363d", padx=2, pady=2)
        stats_outer.grid(row=1, column=0, padx=(8, 4), pady=(4, 8), sticky="nsew")
        self.build_stats_panel(stats_outer) 

        ctrl_outer = tk.Frame(self.root, bg="#30363d", padx=2, pady=2)
        ctrl_outer.grid(row=1, column=1, padx=(4, 8), pady=(4, 8), sticky="nsew")
        self.build_control_panel(ctrl_outer) 

    def style_axes(self, ax, title) -> None:
        ax.set_facecolor('#161b22') 
        ax.set_title(title, color='#c9d1d9', fontsize=11, fontweight='bold', pad=8)
        ax.tick_params(colors='#8b949e', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.grid(True, alpha=0.15, color='#8b949e')

    def build_charts_tab(self) -> None:
        tab = tk.Frame(self.notebook, bg="#0d1117")
        self.notebook.add(tab, text="  Live Simulation  ")

        self.fig = Figure(figsize=(7.6, 5.4), dpi=100, facecolor='#0d1117')
        self.fig.subplots_adjust(hspace=0.45, left=0.1, right=0.95, top=0.93, bottom=0.08)

        self.ax_fitness = self.fig.add_subplot(211)
        self.ax_score = self.fig.add_subplot(212)
        self.style_axes(self.ax_fitness, "Evolution History (Fitness)")
        self.style_axes(self.ax_score, "Score per Generation")

        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=tab)
        self.chart_canvas.get_tk_widget().pack(expand=True, fill="both")
        self.chart_canvas.draw()

    def build_research_tab(self) -> None:
        tab = tk.Frame(self.notebook, bg="#0d1117")
        self.notebook.add(tab, text="  Research Lab  ")

        self.fig2 = Figure(figsize=(7.6, 5.4), dpi=100, facecolor='#0d1117')
        self.fig2.subplots_adjust(hspace=0.5, wspace=0.35, left=0.08, right=0.95, top=0.93, bottom=0.08)

        self.ax_w1 = self.fig2.add_subplot(221)
        self.ax_w2 = self.fig2.add_subplot(222)
        self.ax_b1 = self.fig2.add_subplot(223)
        self.ax_dist = self.fig2.add_subplot(224)

        for ax in [self.ax_w1, self.ax_w2, self.ax_b1, self.ax_dist]:
            self.style_axes(ax, "")

        self.ax_w1.set_title("Weights Layer 1 (Input→Hidden)", color='#c9d1d9', fontsize=9, fontweight='bold')
        self.ax_w2.set_title("Weights Layer 2 (Hidden→Output)", color='#c9d1d9', fontsize=9, fontweight='bold')
        self.ax_b1.set_title("Biases (Hidden Layer)", color='#c9d1d9', fontsize=9, fontweight='bold')
        self.ax_dist.set_title("Fitness Distribution", color='#c9d1d9', fontsize=9, fontweight='bold')

        self.chart_canvas2 = FigureCanvasTkAgg(self.fig2, master=tab)
        self.chart_canvas2.get_tk_widget().pack(expand=True, fill="both")
        self.chart_canvas2.draw()

    def build_stats_panel(self, parent) -> None:
        inner = tk.Frame(parent, bg="#0d1117")
        inner.pack(expand=True, fill="both", padx=8, pady=8)

        tk.Label(inner, text="LIVE STATISTICS", font=self.font_title,
                 fg="#e94560", bg="#0d1117").pack(pady=(5, 8))

        grid = tk.Frame(inner, bg="#0d1117")
        grid.pack(expand=True, fill="both")

        self.stat_labels = {}
        items = [
            ("Mode", "TRAINING", "#2ecc71", 0, 0),
            ("Generation", "1", "#c9d1d9", 0, 1),
            ("Alive", "100", "#c9d1d9", 1, 0),
            ("Score", "0", "#c9d1d9", 1, 1),
            ("Max Score", "0", "#f0883e", 2, 0),
            ("Speed", "1x", "#58a6ff", 2, 1),
        ]
        for name, default, color, row, col in items:
            cell = tk.Frame(grid, bg="#161b22", padx=10, pady=5)
            cell.grid(row=row, column=col, padx=3, pady=3, sticky="ew")
            grid.columnconfigure(col, weight=1)

            tk.Label(cell, text=name.upper(), font=("Segoe UI", 8), fg="#8b949e", bg="#161b22").pack(anchor="w")
            lbl = tk.Label(cell, text=default, font=self.font_value, fg=color, width=14, bg="#161b22")
            lbl.pack(anchor="w")
            self.stat_labels[name] = lbl

    def build_control_panel(self, parent) -> None:
        inner = tk.Frame(parent, bg="#0d1117")
        inner.pack(expand=True, fill="both", padx=8, pady=8)

        tk.Label(inner, text="CONTROL PANEL", font=self.font_title,
                 fg="#e94560", bg="#0d1117").pack(pady=(5, 6))

        container = tk.Frame(inner, bg="#0d1117")
        container.pack(expand=True, fill="both")

        left = tk.Frame(container, bg="#0d1117")
        left.pack(side="left", expand=True, fill="both", padx=(7, 5))

        spd_frame = tk.Frame(left, bg="#161b22", padx=8, pady=6)
        spd_frame.pack(fill="x", pady=3)
        tk.Label(spd_frame, text="SPEED", font=("Segoe UI", 8), fg="#8b949e", bg="#161b22").pack(anchor="w")

        spd_btns = tk.Frame(spd_frame, bg="#161b22")
        spd_btns.pack(fill="x", pady=3)

        self.speed_buttons = {}
        for spd in [1, 2, 5, 10, 50]:
            bg_color = "#e94560" if spd == 1 else "#21262d"
            btn = tk.Label(spd_btns, text=f"{spd}x", font=("Consolas", 10, "bold"),
                           bg=bg_color, fg="#ffffff", padx=6, pady=4, cursor="hand2", width=4)
            btn.pack(side="left", expand=True, fill="x", padx=2)
            
            btn.bind("<Button-1>", lambda e, s=spd: self.set_speed(s))
            self.speed_buttons[spd] = btn

        self.pause_btn = tk.Label(spd_btns, text="||", font=("Consolas", 10, "bold"),
                                  bg="#21262d", fg="#ffffff", padx=6, pady=4,
                                  cursor="hand2", width=3)
        self.pause_btn.pack(side="left", padx=2)
        self.pause_btn.bind("<Button-1>", lambda e: self.toggle_pause())

        mut_frame = tk.Frame(left, bg="#161b22", padx=8, pady=6)
        mut_frame.pack(fill="x", pady=3)
        tk.Label(mut_frame, text="MUTATION RATE", font=("Segoe UI", 8), fg="#8b949e", bg="#161b22").pack(anchor="w")

        self.mutation_var = tk.DoubleVar(value=config.MUTATION_RATE)
        scale = tk.Scale(mut_frame, from_=0.01, to=0.5, resolution=0.01,
                         orient="horizontal", variable=self.mutation_var,
                         bg="#161b22", fg="#c9d1d9", troughcolor="#21262d",
                         highlightthickness=0, font=("Consolas", 8),
                         activebackground="#e94560", command=self.on_mutation_change)
        scale.pack(fill="x")

        right = tk.Frame(container, bg="#0d1117")
        right.pack(side="right", expand=True, fill="both", padx=(5, 5))

        actions_frame = tk.Frame(right, bg="#161b22", padx=8, pady=6)
        actions_frame.pack(fill="both", expand=True, pady=3)
        tk.Label(actions_frame, text="ACTIONS", font=("Segoe UI", 8), fg="#8b949e", bg="#161b22").pack(anchor="w", pady=(0, 4))

        buttons = [
            ("Save Best Model", "#238636", self.ctrl.action_save),
            ("Load Model & Play", "#1f6feb", self.ctrl.action_load),
            ("Reset Training", "#da3633", self.ctrl.action_reset),
            ("Export CSV", "#8957e5", self.ctrl.action_export),
        ]
        
        for text, color, cmd in buttons:
            btn = tk.Label(actions_frame, text=text, font=("Segoe UI", 10, "bold"),
                           bg=color, fg="#ffffff", padx=10, pady=7, cursor="hand2")
            btn.pack(fill="x", pady=3)
            btn.bind("<Button-1>", lambda e, c=cmd: c())
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg="#c9d1d9", fg="#0d1117"))
            btn.bind("<Leave>", lambda e, b=btn, c=color: b.config(bg=c, fg="#ffffff"))
        
    def set_speed(self, speed) -> None:
        self.speed_multiplier = speed
        for s, btn in self.speed_buttons.items():
            btn.configure(bg="#e94560" if s == speed else "#21262d")
            
        self.stat_labels["Speed"].configure(text=f"{speed}x")

    def on_mutation_change(self, value) -> None:
        config.MUTATION_RATE = float(value)

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.pause_btn.configure(text=">" if self.is_paused else "||",
                                 bg="#238636" if self.is_paused else "#21262d")

    def update_charts(self) -> None:
        history = self.engine.history_data
        
        if not history or len(history) == self.last_chart_gen_count:
            return
            
        self.last_chart_gen_count = len(history)

        gens = [h["Generation"] for h in history]
        max_fit = [h.get("Max Fitness", 0) for h in history]
        avg_fit = [h.get("Avg Fitness", 0) for h in history]
        scores = [h.get("Gen Best Score", 0) for h in history]

        self.ax_fitness.clear()
        self.style_axes(self.ax_fitness, "Evolution History (Fitness)")
        
        self.ax_fitness.plot(gens, max_fit, color='#e94560', linewidth=2, label='Max Fitness', marker='o', markersize=5)
        self.ax_fitness.plot(gens, avg_fit, color='#58a6ff', linewidth=2, label='Avg Fitness', marker='s', markersize=5)
        
        self.ax_fitness.fill_between(gens, avg_fit, alpha=0.15, color='#58a6ff')
        self.ax_fitness.fill_between(gens, max_fit, alpha=0.11, color='#e94560')
        
        self.ax_fitness.legend(loc='upper left', fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
        self.ax_fitness.set_ylabel('Fitness', color='#8b949e', fontsize=8)

        self.ax_score.clear()
        self.style_axes(self.ax_score, "Score per Generation")
        
        colors = ['#9d00ff' if s > 0 else '#21262d' for s in scores]
        self.ax_score.bar(gens, scores, color=colors, alpha=0.3, edgecolor=colors, linewidth=1.5)
        
        if len(gens) > 1:
            z = np.polyfit(gens, scores, min(3, len(gens) - 1))
            p = np.poly1d(z)
            x_smooth = np.linspace(min(gens), max(gens), 100)
            
            self.ax_score.plot(x_smooth, p(x_smooth), color='#ffd700', linewidth=2.5, linestyle='--', label='Trend Vector')
            self.ax_score.legend(loc='upper left', fontsize=8, facecolor='#0d1117', edgecolor='#30363d', labelcolor='#c9d1d9')
            
        self.ax_score.set_xlabel('Generation', color='#8b949e', fontsize=8)
        self.ax_score.set_ylabel('Score', color='#8b949e', fontsize=8)
        self.chart_canvas.draw_idle()

    def update_nn_viz(self) -> None:
        best_brain = getattr(self.engine, 'last_gen_best_brain', None)
        if best_brain is None:
            if self.engine.best_all_time_brain:
                best_brain = self.engine.best_all_time_brain
            else:
                return

        self.ax_w1.clear()
        self.style_axes(self.ax_w1, "Weights L1 (Input→Hidden)")
        im1 = self.ax_w1.imshow(best_brain.w1, cmap='coolwarm', aspect='auto', interpolation='nearest')
        self.ax_w1.set_xlabel('Hidden Neurons', color='#8b949e', fontsize=7)
        self.ax_w1.set_ylabel('Input Sensors', color='#8b949e', fontsize=7)
        
        labels_in = ['Dist X', 'Pipe Top', 'Pipe Bot', 'Velocity']
        if best_brain.w1.shape[0] == 4:
            self.ax_w1.set_yticks(range(4))
            self.ax_w1.set_yticklabels(labels_in, fontsize=7)

        self.ax_w2.clear()
        self.style_axes(self.ax_w2, "Weights L2 (Hidden→Output)")
        self.ax_w2.imshow(best_brain.w2, cmap='coolwarm', aspect='auto', interpolation='nearest')
        self.ax_w2.set_xlabel('Output', color='#8b949e', fontsize=7)
        self.ax_w2.set_ylabel('Hidden Neurons', color='#8b949e', fontsize=7)

        self.ax_b1.clear()
        self.style_axes(self.ax_b1, "Biases (Hidden Layer)")
        b1_vals = best_brain.b1.flatten() 
        bar_colors = ['#e94560' if v < 0 else '#238636' for v in b1_vals]
        self.ax_b1.bar(range(len(b1_vals)), b1_vals, color=bar_colors, alpha=0.85)
        self.ax_b1.set_xlabel('Neuron Index', color='#8b949e', fontsize=7)

        self.ax_dist.clear()
        self.style_axes(self.ax_dist, "Fitness Distribution")
        fitnesses = [getattr(b, 'fitness', 0) for b in self.engine.population.birds]
        self.ax_dist.hist(fitnesses, bins=20, color='#8957e5', alpha=0.8, edgecolor='#6e40c9')
        self.ax_dist.set_xlabel('Fitness Score', color='#8b949e', fontsize=7)
        self.ax_dist.set_ylabel('Number of Birds', color='#8b949e', fontsize=7)

        self.chart_canvas2.draw_idle()
    
    def smart_chart_updater(self):
        try:
            active_tab = self.notebook.index("current")
        except Exception:
            return

        current_gen = self.engine.population.generation

        if active_tab == 0:
            if getattr(self, 'last_drawn_tab0', 0) != current_gen:
                self.update_charts() 
                self.last_drawn_tab0 = current_gen 
        elif active_tab == 1:
            if getattr(self, 'last_drawn_tab1', 0) != current_gen:
                self.update_nn_viz()
                self.last_drawn_tab1 = current_gen 
    
    def update_live_stats(self) -> None:
        e = self.engine
        
        def smart_update(name, new_text, new_fg=None):
            lbl = self.stat_labels[name]
            if lbl.cget("text") != new_text:
                lbl.configure(text=new_text)
            if new_fg and lbl.cget("fg") != new_fg:
                lbl.configure(fg=new_fg)

        if getattr(e, 'demo_brain', None) is not None: 
            smart_update("Mode", "PLAY", "#FFD700") 
        else:
            smart_update("Mode", "TRAINING", "#2ecc71") 

        smart_update("Generation", str(e.population.generation))
        smart_update("Alive", f"{e.population.get_alive_count()} / {e.population.size}")
        smart_update("Score", str(e.score))
        smart_update("Max Score", str(e.max_score))

    def show_toast(self, message: str, duration_ms: int = 3000, color: str = "#238636") -> None:
        toast = tk.Label(
            self.root, text=message,
            bg=color, fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padx=14, pady=8
        )
        toast.place(relx=0.5, rely=0.96, anchor="center")
        self.root.after(duration_ms, toast.destroy)

    # 1ST THREAD: BACKGROUND -> CALCULATING
    def game_loop_thread(self) -> None:
        frame_time = config.FRAME_TIME / 1000.0
        while self.game_thread_running:
            if not self.is_paused:
                # when game holds lock, UI cannot run
                with self.game_lock:
                    # 1x 2x 5x 10x ...
                    for _ in range(self.speed_multiplier):
                        self.engine.update_logic()
            # release key for the UI
            time.sleep(frame_time) 
    
    # 2ND THREAD: MAIN -> TKINTER UI
    def ui_loop(self) -> None:
        if not self.is_paused:
            # when game holds key (blocking = false) -> UI wont wait
            if self.game_lock.acquire(blocking=False):
                try:
                    # draw everything into UI
                    self.engine.draw_ui()
                finally:
                    # when finishing drawing, release key again
                    self.game_lock.release()

        self.update_live_stats()
        if not self.dialog_open:
            self.smart_chart_updater()

        # 1/60s will call -> 60FPS
        self.root.after(config.FRAME_TIME, self.ui_loop)
    
    def on_closing(self):
        self.game_thread_running = False
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SimulationAILab(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()