import numpy as np
import matplotlib.pyplot as plt
from .config import SLAMConfig
from .environment import FireScoutEnvironment
from .robot import FireScoutRobot
from .ekf import EKFEstimator
from tqdm.notebook import tqdm
import math


class FireScoutSimulation:
    def __init__(self, config: SLAMConfig):
        self.config = config
        self.env = FireScoutEnvironment(config)
        self.robot = FireScoutRobot(config, self.env)
        self.ekf = EKFEstimator(config)
        
        # Stats
        self.time_elapsed = 0.0
        self.mapping_complete = False
        self.mapped_cells = set()

    def run(self):
        print(f"Starting Simulation: {self.config.grid_rows}x{self.config.grid_cols} Grid")
        
        for step in tqdm(range(self.config.num_steps), desc="Simulating"):
            if self.mapping_complete:
                break
                
            # 1. Move
            u = self.robot.move_wall_follower()
            
            # 2. Predict
            self.ekf.predict(u)
            
            # 3. Sense
            observations = self.robot.sense()
            
            # 4. Update
            self.ekf.update(observations)
            
            # 5. Check Completion
            self._check_completion()
            self._update_mapped_coverage()
            self.time_elapsed += self.config.time_per_step

        self.visualize_results()

    def _check_completion(self):
        # Check if robot has visited all target locations within tolerance
        visited_count = 0
        for tx, ty in self.env.targets:
            # Check distance from current robot path history
            for (rx, ry) in self.robot.path_history:
                dist = math.sqrt((tx - rx)**2 + (ty - ry)**2)
                if dist <= self.config.target_tolerance:
                    visited_count += 1
                    break
        
        if visited_count == len(self.env.targets):
            self.mapping_complete = True
            print(f"MISSION COMPLETE: All targets visited in {self.time_elapsed:.2f}s")

    def _update_mapped_coverage(self):
        # Mark cells in FOV as mapped
        r, c = int(self.robot.true_pos[0]), int(self.robot.true_pos[1])
        rad = self.config.fov_radius
        for i in range(r - rad, r + rad + 1):
            for j in range(c - rad, c + rad + 1):
                if 0 <= i < self.env.rows and 0 <= j < self.env.cols:
                    self.mapped_cells.add((i, j))

    def visualize_results(self):
        """Generates the final plot."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 1. Plot Grid (Walls)
        # Create an image for the static grid
        display_grid = np.zeros_like(self.env.grid)
        display_grid[self.env.grid == 1] = 1 # Walls
        
        # Plot Base Map
        ax.imshow(display_grid, cmap='Greys', origin='upper')
        
        # 2. Plot True Robot Path
        path_arr = np.array(self.robot.path_history)
        ax.plot(path_arr[:, 1], path_arr[:, 0], c='lime', label='True Path', linewidth=1.5, alpha=0.8)
        
        # 3. Plot Estimated Robot Path? 
        # (We didn't store history of EKF state in this simplified loop, but current pos is known)
        est_r, est_c = self.ekf.mu[0], self.ekf.mu[1]
        ax.scatter(est_c, est_r, c='blue', marker='o', s=100, label='Est. Robot Pos')
        
        # 4. Plot Landmarks (True vs Estimated)
        # True
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                feat = self.env.grid[r, c]
                if feat == 2:
                    ax.scatter(c, r, c='orange', marker='^', label='Debris (True)' if 'Debris (True)' not in [l.get_label() for l in ax.lines] else "")
                elif feat == 3:
                    ax.scatter(c, r, c='red', marker='*', label='Hotspot (True)' if 'Hotspot (True)' not in [l.get_label() for l in ax.lines] else "")

        # Estimated Landmarks
        for lm_id, idx in self.ekf.landmark_registry.items():
            lx = self.ekf.mu[idx]
            ly = self.ekf.mu[idx+1]
            ax.scatter(ly, lx, c='purple', marker='x', s=80, label='Est. Landmark' if 'Est. Landmark' not in [l.get_label() for l in ax.collections] else "")

        # 5. Targets
        for tx, ty in self.env.targets:
            circle = plt.Circle((ty, tx), self.config.target_tolerance, color='cyan', fill=False, linestyle='--')
            ax.add_patch(circle)
            ax.text(ty, tx, "T", color='cyan', ha='center', va='center', fontweight='bold')

        ax.set_title(f"Fire Scout SLAM Mission - Time: {self.time_elapsed}s")
        ax.legend(loc='upper right')
        plt.show()