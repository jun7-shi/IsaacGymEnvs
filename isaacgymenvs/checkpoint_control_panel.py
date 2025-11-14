#!/usr/bin/env python3
"""
Checkpoint Control Panel for train_play.py

This panel provides a GUI to control which checkpoint is loaded during visualization.
It communicates with the train_play.py play mode process via a shared state file.

Usage:
    python isaacgymenvs/checkpoint_control_panel.py experiment=<experiment_name>
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import json
import glob
import argparse
from datetime import datetime
from pathlib import Path


class CheckpointControlPanel:
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.checkpoint_dir = os.path.join(experiment_dir, 'nn')
        self.state_file = os.path.join(experiment_dir, '.checkpoint_state.json')

        # Create main window
        self.root = tk.Tk()
        self.root.title(f"Checkpoint Control - {os.path.basename(experiment_dir)}")
        self.root.geometry("600x500")

        # Initialize state
        self.current_mode = "auto"
        self.current_checkpoint = None
        self.checkpoints = []

        # Setup UI
        self.setup_ui()

        # Load initial state
        self.load_state()
        self.refresh_checkpoints()

        # Auto-refresh checkpoints every 5 seconds
        self.root.after(5000, self.auto_refresh)

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="Checkpoint Control Panel",
                                font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Experiment info
        exp_label = ttk.Label(main_frame, text=f"Experiment: {os.path.basename(self.experiment_dir)}")
        exp_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))

        # Mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Mode", padding="10")
        mode_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.mode_var = tk.StringVar(value="auto")
        auto_radio = ttk.Radiobutton(mode_frame, text="Auto (always load latest.pth)",
                                     variable=self.mode_var, value="auto",
                                     command=self.on_mode_change)
        auto_radio.grid(row=0, column=0, sticky=tk.W, pady=2)

        manual_radio = ttk.Radiobutton(mode_frame, text="Manual (select checkpoint)",
                                       variable=self.mode_var, value="manual",
                                       command=self.on_mode_change)
        manual_radio.grid(row=1, column=0, sticky=tk.W, pady=2)

        # Checkpoint list
        list_frame = ttk.LabelFrame(main_frame, text="Available Checkpoints", padding="10")
        list_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        main_frame.rowconfigure(3, weight=1)

        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Listbox
        self.checkpoint_listbox = tk.Listbox(list_frame, height=10,
                                             yscrollcommand=scrollbar.set,
                                             selectmode=tk.SINGLE)
        self.checkpoint_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.checkpoint_listbox.bind('<<ListboxSelect>>', self.on_checkpoint_select)
        scrollbar.config(command=self.checkpoint_listbox.yview)

        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=(0, 10))

        refresh_btn = ttk.Button(button_frame, text="Refresh List", command=self.refresh_checkpoints)
        refresh_btn.grid(row=0, column=0, padx=5)

        load_btn = ttk.Button(button_frame, text="Load Selected", command=self.load_selected)
        load_btn.grid(row=0, column=1, padx=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var,
                                relief=tk.SUNKEN, anchor=tk.W)
        status_label.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # Current checkpoint display
        current_frame = ttk.LabelFrame(main_frame, text="Current Checkpoint", padding="10")
        current_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        self.current_label = ttk.Label(current_frame, text="None", wraplength=550)
        self.current_label.grid(row=0, column=0, sticky=tk.W)

    def refresh_checkpoints(self):
        """Scan checkpoint directory and update list"""
        try:
            if not os.path.exists(self.checkpoint_dir):
                self.status_var.set(f"Checkpoint directory not found: {self.checkpoint_dir}")
                return

            # Find all .pth files
            pattern = os.path.join(self.checkpoint_dir, "*.pth")
            checkpoint_files = glob.glob(pattern)

            # Sort by modification time (newest first)
            checkpoint_files.sort(key=os.path.getmtime, reverse=True)

            # Clear listbox
            self.checkpoint_listbox.delete(0, tk.END)
            self.checkpoints = []

            # Add checkpoints to listbox
            for ckpt_path in checkpoint_files:
                filename = os.path.basename(ckpt_path)
                mod_time = os.path.getmtime(ckpt_path)
                mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)

                display_text = f"{filename} ({size_mb:.2f} MB) - {mod_time_str}"
                self.checkpoint_listbox.insert(tk.END, display_text)
                self.checkpoints.append(ckpt_path)

            self.status_var.set(f"Found {len(checkpoint_files)} checkpoint(s)")

        except Exception as e:
            self.status_var.set(f"Error scanning checkpoints: {e}")

    def on_mode_change(self):
        """Handle mode change"""
        self.current_mode = self.mode_var.get()

        if self.current_mode == "auto":
            self.current_checkpoint = os.path.join(self.checkpoint_dir, "latest.pth")
            self.save_state()
            self.update_current_display()
            self.status_var.set("Switched to auto mode - will load latest.pth")
        else:
            self.status_var.set("Switched to manual mode - select a checkpoint")

    def on_checkpoint_select(self, event):
        """Handle checkpoint selection from list"""
        selection = self.checkpoint_listbox.curselection()
        if selection and self.mode_var.get() == "manual":
            idx = selection[0]
            self.current_checkpoint = self.checkpoints[idx]
            self.update_current_display()

    def load_selected(self):
        """Load the selected checkpoint"""
        if self.mode_var.get() == "manual":
            selection = self.checkpoint_listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a checkpoint first")
                return

            idx = selection[0]
            self.current_checkpoint = self.checkpoints[idx]
        else:
            self.current_checkpoint = os.path.join(self.checkpoint_dir, "latest.pth")

        self.save_state()
        self.update_current_display()

        filename = os.path.basename(self.current_checkpoint)
        self.status_var.set(f"Selected: {filename} - Restart play mode to apply")

        # Show info message
        messagebox.showinfo("Checkpoint Selected",
                           f"Selected: {filename}\n\n"
                           "To apply this checkpoint:\n"
                           "1. Press ESC in the visualization window\n"
                           "2. Re-run the play mode command")

    def save_state(self):
        """Save current state to file"""
        try:
            state = {
                "mode": self.current_mode,
                "checkpoint_path": self.current_checkpoint,
                "last_update": datetime.now().isoformat(),
                "experiment_dir": self.experiment_dir
            }

            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            self.status_var.set(f"Error saving state: {e}")

    def load_state(self):
        """Load state from file if exists"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)

                self.current_mode = state.get("mode", "auto")
                self.current_checkpoint = state.get("checkpoint_path", None)
                self.mode_var.set(self.current_mode)
                self.update_current_display()

        except Exception as e:
            self.status_var.set(f"Error loading state: {e}")

    def update_current_display(self):
        """Update the current checkpoint display"""
        if self.current_checkpoint:
            filename = os.path.basename(self.current_checkpoint)
            if os.path.exists(self.current_checkpoint):
                mod_time = os.path.getmtime(self.current_checkpoint)
                mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                size_mb = os.path.getsize(self.current_checkpoint) / (1024 * 1024)
                text = f"{filename} ({size_mb:.2f} MB) - Modified: {mod_time_str}"
            else:
                text = f"{filename} (not found)"
            self.current_label.config(text=text)
        else:
            self.current_label.config(text="None")

    def auto_refresh(self):
        """Auto-refresh checkpoint list every 5 seconds"""
        self.refresh_checkpoints()
        self.update_current_display()
        self.root.after(5000, self.auto_refresh)

    def run(self):
        """Run the control panel"""
        self.root.mainloop()


def find_experiment_dir(experiment_name):
    """Find experiment directory, with auto-matching like train_play.py"""
    experiment_dir = os.path.join('runs', experiment_name)

    if os.path.exists(experiment_dir):
        return experiment_dir

    # Try to find with glob pattern
    pattern = os.path.join('runs', f"{experiment_name}*")
    matching_dirs = sorted(glob.glob(pattern), reverse=True)

    if matching_dirs:
        return matching_dirs[0]

    return None


def main():
    parser = argparse.ArgumentParser(description='Checkpoint Control Panel for train_play.py')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment directory name (e.g., Ant_2025-01-14_10-30-45 or just Ant)')

    args = parser.parse_args()

    # Find experiment directory
    experiment_dir = find_experiment_dir(args.experiment)

    if not experiment_dir:
        print(f"Error: Experiment directory not found for '{args.experiment}'")
        print(f"Please check available experiments in 'runs/' directory")
        return

    if not os.path.exists(os.path.join(experiment_dir, 'nn')):
        print(f"Error: No 'nn' directory found in {experiment_dir}")
        print(f"This doesn't appear to be a valid training experiment directory")
        return

    print(f"Starting control panel for: {experiment_dir}")

    # Create and run panel
    panel = CheckpointControlPanel(experiment_dir)
    panel.run()


if __name__ == "__main__":
    main()
