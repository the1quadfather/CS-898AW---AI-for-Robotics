import numpy as np
from typing import List, Tuple, Dict, Any
from .config import SLAMConfig, Direction, Vector  # Relative import
from .environment import FireScoutEnvironment

class FireScoutRobot:
    """
    Represents the robot agent.
    Handles True State (not estimated), Kinematics, and Sensing.
    """
    def __init__(self, config: SLAMConfig, env: FireScoutEnvironment):
        self.config = config
        self.env = env
        
        # True State [row, col]
        self.true_pos = np.array(config.start_pos, dtype=float).reshape(2, 1)
        self.orientation = Direction.UP
        
        # Navigation Memory
        self.path_history: List[Tuple[int, int]] = [config.start_pos]
        self.recent_path: List[Tuple[int, int]] = [] 

    def move_wall_follower(self) -> Vector:
        """
        Calculates the next move vector based on wall-following logic.
        Returns control input u = [delta_row, delta_col].
        """
        row, col = int(np.round(self.true_pos[0, 0])), int(np.round(self.true_pos[1, 0]))
        
        # Calculate adjacent cells based on current orientation
        # Standard transforms relative to orientation: Forward, Left, Right
        moves = {
            Direction.UP:    {'L': (0, -1), 'F': (-1, 0), 'R': (0, 1)},
            Direction.RIGHT: {'L': (-1, 0), 'F': (0, 1),  'R': (1, 0)},
            Direction.DOWN:  {'L': (0, 1),  'F': (1, 0),  'R': (0, -1)},
            Direction.LEFT:  {'L': (1, 0),  'F': (0, -1), 'R': (-1, 0)},
        }
        
        rel = moves[self.orientation]
        
        # Check collisions (Walls)
        wall_left = not self.env.is_valid_location(row + rel['L'][0], col + rel['L'][1])
        wall_fwd  = not self.env.is_valid_location(row + rel['F'][0], col + rel['F'][1])
        wall_right= not self.env.is_valid_location(row + rel['R'][0], col + rel['R'][1])

        # Wall Following Logic (Right-hand rule heuristic or similar)
        # Priority: 
        # 1. If no wall on left, turn left and move (follow left wall).
        # 2. If no wall forward, move forward.
        # 3. If wall forward, turn right.
        
        next_move = (0, 0)
        
        if not wall_left:
            next_move = rel['L']
            self.orientation = Direction((self.orientation - 1) % 4) # Turn Left
        elif not wall_fwd:
            next_move = rel['F']
            # Orientation stays same
        elif not wall_right:
            next_move = rel['R']
            self.orientation = Direction((self.orientation + 1) % 4) # Turn Right
        else:
            # Dead end, turn right (or 180)
            self.orientation = Direction((self.orientation + 1) % 4)
            
        # --- Memory Check (Avoid Loops) ---
        # A simple check to prevent getting stuck in loops, effectively "bumping" the robot
        candidate_pos = (row + next_move[0], col + next_move[1])
        
        if len(self.recent_path) > 1 and candidate_pos in self.recent_path:
             # If intended move leads to recent spot, try forcing a right turn
             if not wall_right:
                 next_move = rel['R']
                 self.orientation = Direction((self.orientation + 1) % 4)
             else:
                 next_move = (0, 0) # Wait/Turn in place

        # Execute Move (Update True State)
        self.true_pos += np.array(next_move).reshape(2, 1)
        
        # Update History
        pos_tuple = (int(np.round(self.true_pos[0])), int(np.round(self.true_pos[1])))
        self.path_history.append(pos_tuple)
        self.recent_path.append(pos_tuple)
        if len(self.recent_path) > self.config.memory_length:
            self.recent_path.pop(0)
            
        return np.array(next_move).reshape(2, 1)

    def sense(self) -> List[Dict[str, Any]]:
        """
        Simulates LIDAR/Visual sensor. 
        Returns list of observations: {'type': str, 'global_pos': np.array}
        """
        observations = []
        r_center, c_center = int(np.round(self.true_pos[0])), int(np.round(self.true_pos[1]))
        rad = self.config.fov_radius
        
        for r in range(r_center - rad, r_center + rad + 1):
            for c in range(c_center - rad, c_center + rad + 1):
                if 0 <= r < self.env.rows and 0 <= c < self.env.cols:
                    feat = self.env.grid[r, c]
                    # We only 'sense' walls, debris, or hotspots
                    if feat in [1, 2, 3]:
                        # Measurement noise could be added here to 'global_pos' for realism
                        # For this grid sim, we assume precise grid detection but noisy position
                        ftype = 'wall' if feat == 1 else ('debris' if feat == 2 else 'hotspot')
                        observations.append({
                            'type': ftype,
                            'global_pos': np.array([[r], [c]], dtype=float)
                        })
        return observations