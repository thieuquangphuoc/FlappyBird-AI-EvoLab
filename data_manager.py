import os
import json
import csv
import numpy as np

from ai.neural_network import NeuralNetwork
from typing import List, Dict, Optional

class DataManager:
    def __init__(self) -> None:
        pass
    
    # save best brain to file json
    def save_brain(self, brain:'NeuralNetwork', metadata: dict, filepath: str) -> None:
        # dictionary
        data = {
            "metadata": metadata,
            "weights": {
                # using tolist(), as python cannot save np.ndarray
                "w1": brain.w1.tolist(),
                "w2": brain.w2.tolist(),
                "b1": brain.b1.tolist(),
                "b2": brain.b2.tolist()
            }
        }
        
        with open(filepath, "w") as file:
            # write data to file, with indentation for easier reading
            json.dump(data, file, indent = 4)

        print(f"Successfully saved best brain to the {filepath}")
    
    # load json of best brain 
    def load_brain(self, filepath: str) -> Optional['NeuralNetwork']:
        # avoid crashing
        if not os.path.exists(filepath):
            print(f"Cannot file the {filepath}. Please save the best brain!")
            return None
        
        with open(filepath, "r") as file:
            data = json.load(file)

        # spawn new brain
        loaded_brain = NeuralNetwork()
        weights_data = data.get("weights", data)

        # turn list into array by using np.array -> feedforward function
        loaded_brain.w1 = np.array(weights_data["w1"])
        loaded_brain.w2 = np.array(weights_data["w2"])
        loaded_brain.b1 = np.array(weights_data["b1"])
        loaded_brain.b2 = np.array(weights_data["b2"])

        print(f"Successfully loaded the brain from {filepath}")

        return loaded_brain
    
    # export training history
    def export_history(self, history_data: List[Dict[str, float]], filepath: str) -> None:
        if not history_data:
            print(f"Do not have any data to export")
            return
        headers = history_data[0].keys()

        with open(filepath, "w", newline= '') as file:
            # using DictWriter to turn Dictionary into file csv
            writer = csv.DictWriter(file, fieldnames = headers) 
            writer.writeheader() # write header on 1st line
            writer.writerows(history_data) # other lines will be data
        
        print(f"Successfully export data to {filepath}")

