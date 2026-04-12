import config
import os
import json
import csv
import threading

from datetime import datetime, timezone
from tkinter import filedialog, messagebox

class UIController:
    def __init__(self, ui_app) -> None:
        self.ui = ui_app
        os.makedirs(config.MODEL_PATH, exist_ok=True)
        os.makedirs(config.EXPORT_PATH, exist_ok=True)

    # wrapper for multi-threading
    def with_game_paused(self, fn):
        # game is paused or is running?
        was_paused = self.ui.is_paused
        # force game to be paused
        self.ui.is_paused = True

        # get the key, maximum waiting time is 5s
        acquired = self.ui.game_lock.acquire(timeout=5.0)
        if not acquired:
            self.ui.is_paused = was_paused
            raise RuntimeError("Không thể lấy lock sau 5 giây.")

        try:
            # when game stops calculating, do the fn() (reset or save)
            result = fn()
        finally:
            # release key 
            self.ui.game_lock.release()

            # return to origin state
            self.ui.is_paused = was_paused

        return result

    def action_save(self) -> None:
        if self.ui.engine.best_all_time_brain is None:
            messagebox.showwarning(parent=self.ui.root, title="Warning", message="Chưa có model nào để save")
            return

        was_paused = self.ui.is_paused
        self.ui.is_paused = True
        
        # have to beg for the key, before touching UI
        with self.ui.game_lock:
            self.ui.root.update() #

            gen   = self.ui.engine.best_generation
            score = self.ui.engine.max_score
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"model_gen_{gen}_score_{score}_{ts}.json"

            # call dialog -> UI is blocked until user press OK/Cancel
            self.ui.dialog_open = True
            filepath = filedialog.asksaveasfilename(
                parent=self.ui.root, 
                initialdir=config.MODEL_PATH,
                initialfile=default_filename,
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
                title="Save Best Brain"
            )
            self.ui.dialog_open = False

        self.ui.is_paused = was_paused

        if not filepath:
            return

        # be in charge of saving file temporarily
        def save_worker():
            try:
                with self.ui.game_lock:
                    brain = self.ui.engine.best_all_time_brain
                    if brain is None:
                        return
                    w1 = brain.w1.tolist()
                    w2 = brain.w2.tolist()
                    b1 = brain.b1.tolist()
                    b2 = brain.b2.tolist()
                    record_time = self.ui.engine.record_achieved_time
                    record_utc  = self.ui.engine.record_achieved_utc
                    total_sec   = self.ui.engine.total_time_to_max_score
                    max_score   = self.ui.engine.max_score
                    best_gen    = self.ui.engine.best_generation

                save_time  = datetime.now()
                save_utc   = datetime.now(timezone.utc)
                rec_time   = record_time or save_time
                rec_utc    = record_utc  or save_utc
                train_mins = int(total_sec // 60)
                train_secs = round(total_sec % 60, 2)

                metadata = {
                    "Project": "Flappy Bird AI - Neural Network & Genetic Algorithm",
                    "AI_Performance": {
                        "Max_Score": max_score,
                        "Achieved_At_Generation": best_gen,
                        "Total_Training_Time": f"{train_mins} minutes {train_secs} seconds",
                        "Total_Training_Seconds": total_sec
                    },
                    "Record_Achieved_At": {
                        "Local_Time": rec_time.strftime("%d/%m/%Y %H:%M:%S"),
                        "UTC_TimeStamp": rec_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                    },
                    "File_Exported_At": {
                        "Local_Time": save_time.strftime("%d/%m/%Y %H:%M:%S"),
                        "UTC_TimeStamp": save_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                    }
                }
                data = {
                    "metadata": metadata,
                    "weights": {"w1": w1, "w2": w2, "b1": b1, "b2": b2}
                }
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=4)

                self.ui.root.after(0, lambda: self.ui.show_toast(f"Saved: {os.path.basename(filepath)}", color="#238636"))

            except Exception as e:
                err = str(e)
                self.ui.root.after(0, lambda: self.ui.show_toast(f"Error: {err}", color="#da3633"))

        # spawn new thread only to run the save_worker()
        threading.Thread(target=save_worker, daemon=True).start()

    def action_load(self) -> None:
        was_paused = self.ui.is_paused
        self.ui.is_paused = True

        with self.ui.game_lock:
            self.ui.root.update()
            self.ui.dialog_open = True
            filepath = filedialog.askopenfilename(
                parent=self.ui.root,
                initialdir=config.MODEL_PATH,
                filetypes=[("JSON files", "*.json")],
                title="Load Demo Brain"
            )
            self.ui.dialog_open = False

        self.ui.is_paused = was_paused

        if not filepath:
            return

        try:
            loaded_brain = self.ui.data_mgr.load_brain(filepath)
            if loaded_brain is not None:
                def apply_brain():
                    self.ui.engine.demo_brain = loaded_brain
                    self.ui.engine.reset_game()

                self._with_game_paused(apply_brain)
                self.ui.root.after(0, lambda: self.ui.show_toast(f"Loaded: {os.path.basename(filepath)}", color="#238636"))
            else:
                messagebox.showerror(parent=self.ui.root, title="ERROR", message="File bị hỏng hoặc không hợp lệ!")
        except Exception as e:
            self.ui.root.after(0, lambda: self.ui.show_toast(f"Error: {e}", color="#da3633"))

    # same logic as action_save()
    # check dialog and worker thread
    def action_export(self) -> None:
        if len(self.ui.engine.history_data) == 0:
            self.ui.root.after(0, lambda: self.ui.show_toast("No data to export yet", color="#e94560"))
            return

        was_paused = self.ui.is_paused
        self.ui.is_paused = True

        with self.ui.game_lock:
            self.ui.root.update()
            now_str = datetime.now().strftime("%Y%m%d_%H%M")
            default_filename = f"training_log_{now_str}.csv"

            self.ui.dialog_open = True
            filepath = filedialog.asksaveasfilename(
                parent=self.ui.root,
                initialdir=config.EXPORT_PATH,
                initialfile=default_filename,
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Export Training History"
            )
            self.ui.dialog_open = False

        self.ui.is_paused = was_paused

        if not filepath:
            return

        def export_worker():
            try:
                with self.ui.game_lock:
                    history_copy = [dict(row) for row in self.ui.engine.history_data]

                if not history_copy:
                    return

                headers = history_copy[0].keys()
                with open(filepath, "w", newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(history_copy)

                self.ui.root.after(0, lambda: self.ui.show_toast(f"Exported: {os.path.basename(filepath)}", color="#238636"))
            except Exception as e:
                err = str(e)
                self.ui.root.after(0, lambda: self.ui.show_toast(f"Error: {err}", color="#da3633"))

        threading.Thread(target=export_worker, daemon=True).start()

    def action_reset(self) -> None:
        self.with_game_paused(self.ui.engine.reset_training)
        if hasattr(self.ui, 'last_chart_gen_count'):
            self.ui.last_chart_gen_count = 0