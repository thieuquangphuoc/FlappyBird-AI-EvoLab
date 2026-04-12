# FLAPPY BIRD AI EVOLUTION LAB
### Neural Networks & Genetic Algorithms Simulation

> An advanced **simulation environment** designed to explore the self-learning capabilities of Artificial Intelligence within a dynamic 2D obstacle course.  
> Watch as the AI agents teach themselves to master jump timings and navigate through pipes using the power of **natural selection and neuroevolution**—achieving mastery entirely without hard-coded rules.
>
<p align="center">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python">
  <img src="https://img.shields.io/badge/tkinter-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Tkinter">
  <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/matplotlib-%23ffffff.svg?style=for-the-badge&logo=matplotlib&logoColor=black" alt="Matplotlib">
</p>

---
![preview](./assets/preview_demo.gif)

## Table of contents
1. [Project Overview](#1-project-overview)
2. [Technology Stacks](#2-technology-stack)
3. [Souce Code Architecture](#3-source-code-architecture)
4. [AI Core](#4-ai-core)
5. [Data Visualization & Chart Analysis](#5-data-visualization--chart-analysis)
6. [Usage Instructions](#6-usage-instructions)
7. [Execution Guide](#7-execution-guide)
---

## 1. Project Overview

| Information | Details |
|-------------|---------|
| **Objective** | Enable an AI to autonomously master Flappy Bird via evolutionary mechanisms |
| **Methodology** | Neural Network + Genetic Algorithm (NEAT-lite approach) |
| **GUI Framework** | Tkinter (excluding Pygame) |
| **Rationale for Tkinter** | Supports Grid/Frame layout partitioning, enabling seamless embedding of Matplotlib charts |

### Why exclude Pygame?

Pygame renders on a single surface, lacking native support for structural partitioning (Grid/Frame), and exhibits severe latency when embedding analytical charts. Tkinter facilitates the division of the graphical interface into **three independent zones**: Simulation, Control Panel, and Data Visualization—making it highly optimal for research-oriented software.

---
## 2. Technology Stack

| Technology | Role |
|-----------|---------|
| **Python 3.x** | Core Object-Oriented Programming language |
| **Tkinter** | GUI Framework — Canvas for game rendering, Frame for Dashboard layouts |
| **NumPy** | High-performance matrix computations for Neural Network propagation |
| **Matplotlib** | Real-time chart visualization embedded via `FigureCanvasTkAgg` |
| **JSON** | Data persistence for saving/loading neural network weights |
| **CSV** | Experimental logging (Generations, Fitness, Score) |
| **Threading** | Python's built-in library for concurrent execution (Daemon Thread) to prevent UI freezing |

---
## 3. Source Code Architecture

```text
## 3. Source Code Architecture

```text
FlappyBird-AI-EvoLab/
├── main.py                  # Entry point — GUI initialization & Multithreading setup
├── config.py                # Global system constants & algorithmic hyperparameters
├── controller.py            # UI Controller — Manages user interactions & state transitions
├── data_manager.py          # Data I/O — Handles CSV training logs & JSON model persistence
├── requirements.txt         # Project dependencies list (NumPy, Matplotlib)
│
├── ai/                      # ARTIFICIAL INTELLIGENCE CORE
│   ├── neural_network.py    # Multi-Layer Perceptron (ReLU/Sigmoid activations)
│   └── population.py        # Genetic Algorithm — Selection, Crossover, & Mutation
│
├── analysis/                # DATA VISUALIZATION & RESEARCH OUTPUTS
│   ├── data_visualization.ipynb # Jupyter Notebook for generating IEEE research charts
│   └── *.pdf                # Exported analytical plots (Convergence, Heatmaps)
│
├── assets/                  # STATIC ASSETS
│   └── preview_demo.gif     # Simulation demonstration for documentation
│
├── core/                    # PHYSICS & ENVIRONMENT ENGINE
│   ├── bird.py              # Agent class — Kinematics & real-time neural inference
│   ├── pipe.py              # Obstacle class — Dynamic generation & translation
│   ├── ground.py            # Boundary class — Collision floor rendering
│   └── collision.py         # AABB (Axis-Aligned Bounding Box) collision logic
│
├── data/                    # GENERATED DATA & PERSISTENCE
│   ├── logs/                # Exported CSV training histories (Fitness, Score tracking)
│   └── models/              # Saved JSON neural network weights (Champion Brains)
│
└── engine/                  # SIMULATION ENGINE
    └── game_engine.py       # Core loop decoupling physics steps from UI rendering
```
## Module Specifications

### `config.py` — System Configuration
Contains all system constants, categorized as follows:

| Category | Constants | Description |
|------|---------|-------|
| Window | `WIDTH_MAIN_SCREEN`, `HEIGHT_MAIN_SCREEN` | Main window dimensions (1200×800) |
| Physics | `GRAVITY = 0.5`, `JUMP_STRENGTH = -8` | Gravitational pull and jump impulse |
| Pipe | `PIPE_SPEED = 4`, `PIPE_GAP = 150` | Obstacle translation speed and vertical gap |
| GA | `POPULATION_SIZE = 100`, `NUM_ELITISM = 10` | Total population per generation and elite count |
| NN | `INPUT_NODE = 4`, `HIDDEN_NODE = 5`, `OUTPUT_NODE = 1` | Neural network topology |
| Mutation | `MUTATION_RATE = 0.12` | Default mutation probability |

---
### `core/bird.py` — Class Bird (Agent)

```
Bird
├── Physical Attributes:  x, y, velocity, width, height
├── AI attributes:         brain (NeuralNetwork), fitness, alive
├── jump()                 Jump up (velocity = JUMP_STRENGTH)
├── update()               Update physics (gravity + velocity) + increase fitness
├── think(pipes)           Make decision based on Neural Network
├── get_bbox()             Return bounding box for collision
├── vision(canvas)         Draw debug line from bird to nearest pipe
└── draw(canvas)           Draw rectangle representing the bird
```

**The function `think()` — Agent's Brain**
1. Find the nearest pipe ahead
2. Calculate 4 normalized inputs:
   - `input_1`: Horizontal distance to pipe / screen width
   - `input_2`: (Bird position − Top pipe) / screen height
   - `input_3`: (Bottom pipe − Bird position) / screen height
   - `input_4`: Bird velocity / 10
3. Feed into Neural Network → if output > 0.5 → **JUMP**

---

### `core/pipe.py` — Class Pipe (Obstacle)

Each pipe contains:
**Top pipe**: from y=0 to `top_height` (random)
**Gap**: empty space `PIPE_GAP = 150px`
**Bottom pipe**: from `bottom_y to ground`
Moves left at `PIPE_SPEED`

---

### `core/collision.py` — AABB Collision Algorithm

```python
def check_aabb(box1, box2) -> bool:
    # AABB = Axis-Aligned Bounding Box
    # Check if two rectangles overlap
    # box = (x, y, width, height)
```

**Custom implementation** instead of `Pygame's colliderect` — demonstrates algorithmic understanding.

---

### `ai/neural_network.py` — Artificial Neural Networks

```
INPUT (4 nodes)          HIDDEN (5 nodes)         OUTPUT (1 node)
┌─────────────┐        ┌──────────────┐        ┌──────────────┐
│ Distance    │───W1──▶│              │───W2──▶│              │
│ Dist Top    │        │  ReLU(x)     │        │  Sigmoid(x)  │
│ Dist Bottom │        │              │        │  > 0.5? JUMP │
│ Velocity    │        │  5 neurons   │        │              │
└─────────────┘        └──────────────┘        └──────────────┘
      +B1 (bias)             +B2 (bias)
```
| Components | Size | Description |
|-----------|-----------|-------|
| **W1** | 4×5 = 20 weights | Weight Matrix (Input -> Hidden) |
| **B1** | 1×5 = 5 biases | Hidden Layer Biases |
| **W2** | 5×1 = 5 weights | Weight Matrix (Hidden -> Output) |
| **B2** | 1×1 = 1 bias | Output Layer Bias |
| **Tổng** | **31 parameters** | Optimized via Genetic Algorithm |

**Activation Function:**
- **ReLU** (Hidden): `f(x) = max(0, x)` — Eliminates negative values, helping the network learn faster
- **Sigmoid** (Output): `f(x) = 1/(1+e^(-x))` — Compresses the output to a [0,1] range to make the jump/no-jump decision

**Feedforward Operation:**

```
hidden = ReLU(inputs × W1 + B1)
output = Sigmoid(hidden × W2 + B2)

```

**Save/Load Model:**
- `save_to_json(filepath)`: Converts weights/biases to lists → dumps to JSON
- `load_from_json(filepath)`: Reads JSON → reconstructs the NeuralNetwork

---
### `ai/population.py` — Genetic Algorithm

```
Generation N (100 birds)
         │
         ▼
   ┌─────────────┐
   │  All Die    │
   └─────┬───────┘
         ▼
   ┌─────────────────┐
   │Calculate Fitness│  fitness = frames survived
   └─────┬───────────┘
         ▼
   ┌─────────────────┐
   │ Elitism (Top 10)│  Clone top 10 birds 
   └─────┬───────────┘
         ▼
   ┌─────────────────┐
   │ Selection       │  Roulette Wheel (fitness² → probability)
   │ Crossover       │  Uniform Crossover (inherit weights)
   │ Mutation        │  5% chance each weight ± random * 0.3
   └─────┬───────────┘
         ▼
   Generation N+1 (100 new birds)
```

**Detail Steps**

| Step | Name | Description |
|------|-----|-------|
| 1 | **Fitness** | `fitness = frames_survived` → square it → calculate probability |
| 2 | **Elitism** | Keep 10 birds with highest fitness (no mutation) |
| 3 | **Selection** | Roulette Wheel — Higher fitness equals higher chance of being parents |
| 4 | **Crossover** | Uniform — Each weight inherited from parent A OR parent B (50/50)|
| 5 | **Mutation** | 5% each can change by ± `randn() * 0.3` |

**Why Uniform Crossover Instead of Blend??**
- Blend averages values ⇒ destroys learned strategies (catastrophic averaging)
- Uniform preserves weight values ⇒ retains good strategies

---

### `engine/game_engine.py` — Game Engine

```
GameEngine
├── reset_training()     Wipes evolutionary history and re-initializes a fresh population
├── reset_game()         Resets environment & triggers next generation (or Play mode)
├── check_collisions()   Evaluates AABB intersections between agents and obstacles
├── update_logic()       Updates physics, AI feedforward, and scoring (no rendering)
└── draw_ui()            Renders all graphical entities and HUD to the Tkinter Canvas
```
---
### `data_manager.py` - Data Manager
```
DataManager
├── save_brain()         Serializes NeuralNetwork weights/biases & metadata to JSON
├── load_brain()         Deserializes JSON data back into a working NeuralNetwork
└── export_history()     Writes training generation metrics to a CSV file
```
---
### `controller.py` - UI Controller

```
UIController
├── action_save()        Opens dialog to save the champion brain (.json) with metadata
├── action_load()        Opens dialog to load a brain and trigger Play mode
├── action_export()      Opens dialog to export training logs to a CSV file
└── action_reset()       Wipes training data and re-initializes the simulation
```

---
### `main.py` - Main Interface
```
+--------------------+----------------------------------------------+
|                    | [Live Simulation]  [Research Lab]    ← Tabs  |
|    GAME CANVAS     |                                              |
|    (Flappy Bird    |  Tab 1: Fitness + Score charts               |
|     Simulation)    |  Tab 2: NN Heatmaps + Fitness Distribution   |
|                    |                                              |
+--------------------+----------------------------------------------+
|  LIVE STATISTICS   |  CONTROL PANEL                               |
|  Mode | Gen        |  Speed [1x][2x][5x][10x][50x]                |
|  Alive | Score     |  Mutation Rate slider                        |
|  Max Score | Speed |  [Save] [Load & Play] [Reset] [Export CSV]   |
+--------------------+----------------------------------------------+
```

---
## 4. AI Core
### Neural Network (Agent's Brain)
Each bird has **1 unique brain** — _a 3-layer neural network (Input → Hidden → Output)._

Decision-making Process (each frame):
1. Collect 4 environmental signals (pipe distance, gap position, velocity)
2. Multiply through Hidden Layer (5 neurons) with **ReLU activation**
3. Multiply through Output Layer (1 neuron) with **Sigmoid activation**
4. If output > 0.5 → **JUMP**, otherwise → don't jump

### Genetic Algorithm (Agent's Evolutionary Progress)
Instead of Backpropagation (gradient descent), we use **Natural Selection**:

1. `Population`: 100 birds flying simultaneously, each with different brain
2. `Survival`: Bird avoids more pipes → higher fitness
3. `Reproduction`: High fitness birds more likely chosen as parents
4. `Inheritance`: Offspring inherit weights from both parents (Uniform Crossover)
5. `Mutation`: Some weights randomly change → genetic diversity
6. `Iteration`: Over many generations, population becomes increasingly skilled

---
## 5. Data Visualization & Chart Analysis

### Tab "Live Simulation"

#### 1. Evolution History (Fitness) — Line Chart

```
Fitness ▲
        │      ╱‾‾‾‾‾‾‾‾‾‾ Max Fitness (đỏ)
        │    ╱
        │  ╱  ╱‾‾‾‾‾‾‾‾‾‾ Avg Fitness (xanh)
        │╱  ╱
        └──────────────────▶ Generation
```

| Line | Color | Meaning |
|-------|-----|---------|
| **Max Fitness** | Red | Fitness of the best bird in that generation|
| **Avg Fitness** | Blue | Average fitness of the entire population  |

**How to Read:**

- Both lines _going up_ → AI is **learning well**, evolution successful 
- Max increases but Avg doesn't → Only few birds good, majority still weak
- Both lines _flat_ → AI has **converged** (no more improvement)
- Lines going down → AI is **degenerating** (mutation too high) 
- **Fitness** = `frames survived (not score)`. Longer survival = higher fitness.

#### 2. Score per Generation — Bar Chart

```
Score ▲
      │  ▓▓
      │  ▓▓  ▓▓      ▓▓▓▓
      │  ▓▓  ▓▓  ▓▓  ▓▓▓▓  ▓▓▓▓
      └─────────────────────────▶ Generation
```

| Line | Color | Meaning |
|-----------|-----|---------|
| **Purple Chars** | Purple | Highest score in that generation |
| **Trend Line** | Yellow _(dashed)_ | Trend line (polynomial fit) |

**How to read**
- Score = **how many pipes have passed** in that generation
- Trend line _going up_ → AI has passed more pipes over time
- _Unexpected/sudden tall bar_ -> a **genius agent** has appeared!

---

#### 3. Weights Layer 1 (Input→Hidden) — Heatmap

```
                          Weights Layer 1
Input 0 (DistX)    [  0.3  -1.2  0.8  0.1 -0.5 ]
Input 1 (PipeTop)  [ -0.7   0.9  0.2 -0.3  1.1 ]
Input 2 (PipeBot)  [  1.5  -0.4  0.6  0.7 -0.8 ]
Input 3 (Velocity) [  0.2   0.3 -1.0  0.5  0.4 ]
```

| Color | Meaning |
|-----|---------|
| Dark Red | Large positive weight → Input strongly **stimulates** this hidden node |
| Dark Blue | Large negative weight → Input strongly **inhibits** this hidden node |
| White | Near-zero weight → Input has little effect on this hidden node|

**How to red:**
- If "DistX" row is all red → pipe distance is very important for decisions
- If "Velocity" row is all white → AI ignores velocity (may be suboptimal)

#### 4. Weights Layer 2 (Hidden→Output) — Heatmap

A 5×1 matrix — shows how much each hidden neuron contributes to final decision (jump/don't jump)

| Weight | Meaningful |
|--------|---------|
| Large Positive | This hidden neuron **encourages** jumping |
| Large Negative | This hidden neuron **discourages** jumping |

#### 5. Biases (Hidden Layer) — Bar Chart

```
Bias ▲
     │  ▓▓
     │  ▓▓      ▓▓
  0 ─┤──────▓▓──────
     │          ▓▓
     └──────────────▶ Neuron
```

Bias = **activation threshold** for each neuron:
- `Positive bias (green):` Neuron easier to activate (leans toward "jump")
- `Negative bias (red):` Neuron harder to activate (leans toward "don't jump")

#### 6. Fitness Distribution — Histogram

```
Count ▲
      │     ▓▓▓▓
      │     ▓▓▓▓▓▓
      │  ▓▓▓▓▓▓▓▓▓▓  ▓▓
      └──────────────────▶ Fitness
```

|   Shape   | Meaning |
|-----------|---------|
| Concentrated **left**	  | Most birds die early → training incomplete |
| Concentrated **right**  | Most birds survive long → population skilled |
| **Two peaks** (bimodal) |	Skilled and unskilled groups present → diversification happening |
| **Spread evenly**	      | High genetic diversity → exploration phase |

---
## 6. Usage Instructions
### Training Mode (Default)
1. Run app → 100 birds fly automatically
2. Press **5x** or **50x** to speed up training
3. Watch Fitness chart — if going up → AI learning well
4. When score reaches high (e.g., > 50) → press Save Best Model

### Play Mode
1. Press Load Model & Play
2. Select a saved .json file
3. 1 golden bird will play with trained brain
4. Press _Reset Training_ to return to training

### Save Data & Export Data
- `Save Best Model:` Export best bird's weights/biases → .json
- `Export CSV:` Export training log → .csv (Generation, Max Fitness, Avg Fitness, Max Score)

### Control Panel


| Control | Functions |
|---------|-----------|
| **Speed 1x~50x** | Physics steps per frame (speed multiplier)) |
| **Mutation Rate** | Mutation probability (recommended 0.03–0.10) |
| **Pause (‖)** |Pause/resume simulation |

---

## 7. Execution Guide

### Requirements

```bash
pip install numpy matplotlib
```

```bash
pip install numpy numpy
```

> **Tkinter** is a built-in function in python (do not need to install)

### Execute

```bash
cd FlappyBird-AI-EvoLab
python main.py
```
---

## Technical Notes

**1. Why Use fitness² for Selection?**
- Squaring fitness creates _stronger selection pressure_: a bird with fitness 200 is 4x more likely to be chosen than one with fitness 100 (vs. 2x without squaring). Helps good genes spread faster.

**2. Why Need Elitism?**
- Without keeping the best bird, crossover + mutation could destroy the best brain. Elitism = 10 ensures top 10 birds are copied unchanged to the next generation, preserving progress.

**3. Why Multithreading is Implemented?**
> Standard Python GUI frameworks (like Tkinter) run on a single Main Thread. Computing matrix multiplications for 100 Neural Networks at 50x speed and visualize data in six different charts simultaneously would instantly freeze the UI. This project resolves this by implementing a **Concurrent Architecture**:
- **Main Thread:** Handles the Tkinter mainloop, listening to button clicks and rendering the 60 FPS Canvas.
- **Daemon Thread:** Runs the `GameEngine` in the background, crunching physics and AI Feedforward logic silently without blocking the visual interface.
