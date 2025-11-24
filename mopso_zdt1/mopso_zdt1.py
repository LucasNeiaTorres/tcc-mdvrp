import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv

def zdt1(x):
    """
    Calculates the two objectives for the ZDT1 function.
    Args:
        x (np.array): Decision vector (n dimensions, values between 0 and 1).
    Returns:
        tuple: (f1, f2)
    """
    n = len(x)
    if n < 2:
        raise ValueError("ZDT1 requires at least 2 dimensions")

    f1 = x[0]

    # Calculate f2(x)
    g = 1.0 + (9.0 / (n - 1)) * np.sum(x[1:])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h

    return f1, f2

def dominates(obj1, obj2):
    """
    Checks if the solution with objectives obj1 dominates the solution with obj2.
    Assumes minimization for both objectives.
    Args:
        obj1 (tuple): Objectives of the first solution (f1, f2).
        obj2 (tuple): Objectives of the second solution (f1, f2).
    Returns:
        bool: True if obj1 dominates obj2, False otherwise.
    """
    # At least one objective in obj1 is strictly better
    at_least_one_better = obj1[0] < obj2[0] or obj1[1] < obj2[1]

    # No objective in obj1 is worse
    none_worse = obj1[0] <= obj2[0] and obj1[1] <= obj2[1]
    return at_least_one_better and none_worse

class Particle:
    def __init__(self, n_dim, bounds):
        """
        Initializes a particle.
        Args:
            n_dim (int): Number of dimensions.
            bounds (tuple): (lower_bound, upper_bound) for variables.
        """
        self.position = np.random.uniform(bounds[0], bounds[1], n_dim)
        # Initialize velocity to a small random value
        vel_init_range = abs(bounds[1]-bounds[0]) * 0.1
        self.velocity = np.random.uniform(-vel_init_range, vel_init_range, n_dim)
        self.pbest_position = self.position.copy()
        self.objectives = zdt1(self.position)
        self.pbest_objectives = self.objectives

    def update_pbest(self):
        """Updates the pbest if the current position dominates the previous pbest."""
        new_objectives = zdt1(self.position)
        if dominates(new_objectives, self.pbest_objectives):
            self.pbest_position = self.position.copy()
            self.pbest_objectives = new_objectives
        # Handle non-dominated case (replace randomly)
        elif not dominates(self.pbest_objectives, new_objectives):
            if random.random() < 0.5:
                 self.pbest_position = self.position.copy()
                 self.pbest_objectives = new_objectives
        self.objectives = new_objectives # Store current objectives regardless

class ArchiveManager:
    def __init__(self, max_size, n_grid_divisions):
        """
        Initializes the archive manager.
        Args:
            max_size (int): Maximum number of solutions in the archive.
            n_grid_divisions (int): Number of grid divisions per objective dimension.
        """
        self.max_size = max_size
        self.n_grid_divisions = n_grid_divisions
        self.archive_positions = []
        self.archive_objectives = []
        # Grid boundaries in objective space, initialized to extremes
        self.grid_lower_bounds = np.full(2, np.inf) # Assuming 2 objectives for ZDT1
        self.grid_upper_bounds = np.full(2, -np.inf)
        # Dictionary mapping grid cell index (tuple) to list of archive indices in that cell
        self.grid_cell_indices = {}

    def update_archive(self, particle_position, particle_objectives):
        """
        Attempts to add a new solution to the archive, maintaining non-dominance
        and handling archive size limits.
        Args:
            particle_position (np.array): The position of the potential solution.
            particle_objectives (tuple): The objectives of the potential solution.
        """
        is_dominated_by_archive = False
        dominated_indices_in_archive = []

        # Check dominance against the current archive
        for i in range(len(self.archive_objectives)):
            if dominates(self.archive_objectives[i], particle_objectives):
                is_dominated_by_archive = True
                break # New solution is dominated, do not add
            if dominates(particle_objectives, self.archive_objectives[i]):
                # Mark existing archive member for removal
                dominated_indices_in_archive.append(i)

        if not is_dominated_by_archive:
            # Remove solutions dominated by the new one
            # Iterate backwards when removing to avoid index shifting issues
            for index in sorted(dominated_indices_in_archive, reverse=True):
                del self.archive_positions[index]
                del self.archive_objectives[index]

            # Add the new non-dominated solution
            self.archive_positions.append(particle_position.copy())
            self.archive_objectives.append(particle_objectives)

            # Prune archive if it exceeds max size
            if len(self.archive_objectives) > self.max_size:
                self._prune_archive()

    def _update_grid_bounds(self):
        """Updates the grid boundaries based on the current archive objectives."""
        if not self.archive_objectives:
            return
        objectives_array = np.array(self.archive_objectives)
        self.grid_lower_bounds = np.min(objectives_array, axis=0)
        self.grid_upper_bounds = np.max(objectives_array, axis=0)
        # Add a small margin to prevent boundary issues
        range_bounds = self.grid_upper_bounds - self.grid_lower_bounds
        # Avoid division by zero or negative margin if range is zero
        range_bounds[range_bounds <= 0] = 1.0 # Or some small epsilon
        margin = range_bounds * 0.1
        self.grid_lower_bounds -= margin
        self.grid_upper_bounds += margin


    def _get_grid_index(self, objectives):
        """Calculates the grid cell index (tuple) for a given set of objectives."""
        # Check if objectives are outside the current grid bounds
        if np.any(objectives < self.grid_lower_bounds) or np.any(objectives > self.grid_upper_bounds):
             # This can happen before the first pruning/grid update
             # Handle by assigning to an edge cell or temporarily ignoring
             # Simplification for sketch: return None
             return None

        grid_range = self.grid_upper_bounds - self.grid_lower_bounds
        # Prevent division by zero if all points are identical in a dimension
        grid_range[grid_range == 0] = 1.0

        # Normalize position within the grid range [0, 1]
        normalized_pos = (objectives - self.grid_lower_bounds) / grid_range
        # Calculate grid indices [0, n_grid_divisions-1]
        indices = np.floor(normalized_pos * self.n_grid_divisions).astype(int)
        # Clamp indices to be within the valid range
        indices = np.clip(indices, 0, self.n_grid_divisions - 1)
        # Return a hashable tuple representing the cell index
        return tuple(indices)

    def _build_grid_indices(self):
        """Builds the mapping from grid cells to the indices of archive members within them."""
        self._update_grid_bounds() # Ensure bounds are current
        self.grid_cell_indices.clear()
        for i, obj in enumerate(self.archive_objectives):
            grid_idx = self._get_grid_index(obj)
            if grid_idx is not None:
                if grid_idx not in self.grid_cell_indices:
                    self.grid_cell_indices[grid_idx] = []
                self.grid_cell_indices[grid_idx].append(i)

    def _prune_archive(self):
        """Removes one solution from the archive if it exceeds the maximum size,
           targeting the most crowded grid cell."""
        self._build_grid_indices() # Rebuild grid map before pruning
        while len(self.archive_objectives) > self.max_size:
            if not self.grid_cell_indices: # Should not happen if archive > max_size > 0
                break
            # Find the most crowded cell
            most_crowded_cell = max(self.grid_cell_indices, key=lambda k: len(self.grid_cell_indices[k]))
            # Choose a random member from that cell to remove
            index_to_remove = random.choice(self.grid_cell_indices[most_crowded_cell])

            # Remove from grid mapping *before* removing from archive lists
            self.grid_cell_indices[most_crowded_cell].remove(index_to_remove)
            if not self.grid_cell_indices[most_crowded_cell]:
                del self.grid_cell_indices[most_crowded_cell] # Remove cell if empty

            # Effectively remove from archive lists
            # Note: This is inefficient. A better way would be to track indices carefully
            # or use a different data structure. For simplicity, we pop by index.
            # Popping requires adjusting subsequent indices in grid_cell_indices or rebuilding.
            # Rebuilding is simpler for this sketch.
            del self.archive_positions[index_to_remove]
            del self.archive_objectives[index_to_remove]
            # Rebuild grid indices after removal to maintain consistency
            self._build_grid_indices()


    def select_leader(self):
        """Selects a leader from the archive using the grid-based roulette wheel mechanism."""
        if not self.archive_objectives:
            return None # Return None if archive is empty

        # Ensure the grid index mapping is up-to-date for selection
        if not self.grid_cell_indices or len(self.archive_objectives) <= self.max_size : # Rebuild if empty or if pruning might have occurred
             self._build_grid_indices()
             if not self.grid_cell_indices: # Check again after potential rebuild
                 return None

        # Calculate cell fitness (inversely proportional to density)
        # Add 1 to denominator to avoid division by zero and give empty cells zero effective fitness
        cell_fitness = {idx: 1.0 / (len(indices) + 1) for idx, indices in self.grid_cell_indices.items() if indices} # Only consider cells with members
        total_fitness = sum(cell_fitness.values())

        if total_fitness == 0: # Handle cases where fitness calculation fails
             valid_cells_with_members = [idx for idx, indices in self.grid_cell_indices.items() if indices]
             if not valid_cells_with_members: return None
             selected_cell_idx = random.choice(valid_cells_with_members)

        else:
            # Roulette Wheel Selection
            pick = random.uniform(0, total_fitness)
            current_fitness_sum = 0
            selected_cell_idx = None
            # Sort keys for consistent roulette wheel behavior
            sorted_cell_indices = sorted(cell_fitness.keys())
            for cell_idx in sorted_cell_indices:
                current_fitness_sum += cell_fitness[cell_idx]
                if current_fitness_sum >= pick:
                    selected_cell_idx = cell_idx
                    break
            # Fallback in case of floating point issues or empty cells selected
            if selected_cell_idx is None or not self.grid_cell_indices.get(selected_cell_idx):
                 valid_cells_with_members = [idx for idx, indices in self.grid_cell_indices.items() if indices]
                 if not valid_cells_with_members: return None
                 selected_cell_idx = random.choice(valid_cells_with_members)

        # Select a random member from the chosen cell
        leader_archive_index = random.choice(self.grid_cell_indices[selected_cell_idx])
        return self.archive_positions[leader_archive_index]
    
# def mopsopt(problem_func, n_dim, bounds, swarm_size, max_iter, archive_size, n_grid_divisions,
#                              c1, c2, w_max, w_min,
#                              swarm_hist_file="swarm_history.csv", archive_hist_file="archive_history.csv"):
#     """
#     Executes the Multi-Objective Particle Swarm Optimization algorithm.
#     Args:
#         problem_func: The function to optimize (e.g., zdt1).
#         n_dim (int): Number of decision variables.
#         bounds (tuple): (lower_bound, upper_bound) for decision variables.
#         swarm_size (int): Number of particles in the swarm.
#         max_iter (int): Maximum number of iterations.
#         archive_size (int): Maximum size of the external archive.
#         n_grid_divisions (int): Number of grid divisions per objective dimension.
#         c1 (float): Cognitive coefficient.
#         c2 (float): Social coefficient.
#         w_max (float): Initial inertia weight.
#         w_min (float): Final inertia weight.
#     Returns:
#         list: Positions of the non-dominated solutions found in the archive.
#         list: Corresponding objectives of the non-dominated solutions.
#     """
#     # Initialize swarm
#     swarm = [Particle(n_dim, bounds) for _ in range(swarm_size)]
#     # Initialize archive manager
#     archive_manager = ArchiveManager(archive_size, n_grid_divisions)

#     # Store history
#     swarm_objectives_history = []
#     archive_objectives_history = []

#     # Initialize archive with non-dominated solutions from the initial swarm
#     for p in swarm:
#         archive_manager.update_archive(p.position, p.objectives)

#     # Initial state for history (iteration 0)
#     current_swarm_obj = [p.objectives for p in swarm]
#     current_archive_obj = list(archive_manager.archive_objectives) # Make a copy
#     swarm_objectives_history.append(current_swarm_obj)
#     archive_objectives_history.append(current_archive_obj)

#     # Main optimization loop
#     for t in range(max_iter):
#         # Update inertia weight (linear decrease)
#         w = w_max - (w_max - w_min) * t / max_iter

#         # Build/update grid before leader selection for this iteration
#         # This ensures density information is current
#         archive_manager._build_grid_indices()

#         current_swarm_obj_iter = [] # Store objectives for this iteration

#         # Update each particle
#         for i in range(swarm_size):
#             particle = swarm[i]

#             # Select leader from the archive
#             leader_position = archive_manager.select_leader()

#             # Fallback if archive is empty or leader selection fails
#             if leader_position is None:
#                  # Using particle's own pbest is a simple fallback
#                  # Alternative: Use pbest of a randomly chosen particle
#                  leader_position = particle.pbest_position

#             # --- Update velocity ---
#             r1 = np.random.rand(n_dim)
#             r2 = np.random.rand(n_dim)
#             cognitive_velocity = c1 * r1 * (particle.pbest_position - particle.position)
#             social_velocity = c2 * r2 * (leader_position - particle.position)
#             particle.velocity = w * particle.velocity + cognitive_velocity + social_velocity

#             # --- Optional: Velocity clamping ---
#             # Define max velocity, e.g., a fraction of the search space range
#             max_vel = (bounds[1] - bounds[0]) * 0.5
#             particle.velocity = np.clip(particle.velocity, -max_vel, max_vel)

#             # --- Update position ---
#             particle.position += particle.velocity

#             # --- Enforce bounds ---
#             particle.position = np.clip(particle.position, bounds[0], bounds[1])

#             # --- Evaluate new position and update pbest ---
#             particle.update_pbest() # This also evaluates the new position
#             current_swarm_obj_iter.append(particle.objectives)

#             # --- Try to update the external archive ---
#             archive_manager.update_archive(particle.position, particle.objectives)

#         # Store state for this iteration
#         swarm_objectives_history.append(current_swarm_obj_iter)
#         archive_objectives_history.append(list(archive_manager.archive_objectives)) # Store copy

#         # Optional: Print progress
#         if (t + 1) % 50 == 0:
#             print(f"Iteration {t+1}/{max_iter}, Archive Size: {len(archive_manager.archive_objectives)}")

#     # Return the final archive contents
#     return archive_manager.archive_positions, archive_manager.archive_objectives

def mopsopt_and_save_history(problem_func, n_dim, bounds, swarm_size, max_iter, archive_size, n_grid_divisions,
                             c1, c2, w_max, w_min,
                             swarm_hist_file="swarm_history.csv", archive_hist_file="archive_history.csv"):
    """
    Executes MOPSO and saves objective history to CSV files.
    Each row in the CSV represents: iteration, particle/archive_index, f1, f2
    """
    swarm = [Particle(n_dim, bounds) for _ in range(swarm_size)]
    archive_manager = ArchiveManager(archive_size, n_grid_divisions)

    # Initialize archive
    for p in swarm:
        archive_manager.update_archive(p.position, p.objectives)

    # Open CSV files for writing history
    with open(swarm_hist_file, 'w', newline='') as sf, open(archive_hist_file, 'w', newline='') as af:
        swarm_writer = csv.writer(sf)
        archive_writer = csv.writer(af)
        swarm_writer.writerow(['iteration', 'particle_index', 'f1', 'f2'])
        archive_writer.writerow(['iteration', 'archive_index', 'f1', 'f2'])

        # --- Save initial state (Iteration 0) ---
        for idx, p in enumerate(swarm):
            swarm_writer.writerow([0, idx, p.objectives[0], p.objectives[1]])
        for idx, obj in enumerate(archive_manager.archive_objectives):
            archive_writer.writerow([0, idx, obj[0], obj[1]])

        # --- Main optimization loop ---
        for t in range(max_iter):
            iteration_num = t + 1 # Iteration count starts from 1
            w = w_max - (w_max - w_min) * t / max_iter
            archive_manager._build_grid_indices()

            # --- Update particles ---
            for i in range(swarm_size):
                particle = swarm[i]
                leader_position = archive_manager.select_leader()
                if leader_position is None:
                    leader_position = particle.pbest_position # Fallback

                # Update velocity
                r1, r2 = np.random.rand(n_dim), np.random.rand(n_dim)
                cognitive_velocity = c1 * r1 * (particle.pbest_position - particle.position)
                social_velocity = c2 * r2 * (leader_position - particle.position)
                particle.velocity = w * particle.velocity + cognitive_velocity + social_velocity

                # Velocity clamping
                max_vel = (bounds[1] - bounds[0]) * 0.5
                particle.velocity = np.clip(particle.velocity, -max_vel, max_vel)

                # Update position & enforce bounds
                particle.position += particle.velocity
                particle.position = np.clip(particle.position, bounds[0], bounds[1])

                # Evaluate, update pbest
                particle.update_pbest()

                # Save swarm particle state for this iteration
                swarm_writer.writerow([iteration_num, i, particle.objectives[0], particle.objectives[1]])

                # Update archive
                archive_manager.update_archive(particle.position, particle.objectives)

            # --- Save archive state for this iteration ---
            for idx, obj in enumerate(archive_manager.archive_objectives):
                archive_writer.writerow([iteration_num, idx, obj[0], obj[1]])

            # Optional: Print progress
            if iteration_num % 50 == 0:
                print(f"Iteration {iteration_num}/{max_iter}, Archive Size: {len(archive_manager.archive_objectives)}")

    # Return final archive (still useful)
    return archive_manager.archive_positions, archive_manager.archive_objectives

def create_animation(swarm_history, archive_history, max_iter, filename="mopso_zdt1_animation.gif"):
    """
    Creates and saves an animation of the MOPSO execution.
    Args:
        swarm_history (list): List of swarm objectives per iteration.
        archive_history (list): List of archive objectives per iteration.
        max_iter (int): Total number of iterations (frames).
        filename (str): Output filename for the animation (e.g., .gif, .mp4).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot theoretical front only once
    x_front = np.linspace(0, 1, 100)
    y_front = 1 - np.sqrt(x_front)
    ax.plot(x_front, y_front, 'r--', label='Theoretical Pareto Front (ZDT1)', zorder=1)

    # Initialize scatter plots (will be updated in the loop)
    swarm_scatter = ax.scatter([], [], s=15, alpha=0.5, label='Swarm Particles', c='blue', zorder=2)
    archive_scatter = ax.scatter([], [], s=30, marker='*', label='Archive (Pareto Front)', c='green', zorder=3)

    # Set plot limits (adjust if necessary based on observed objective ranges)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.2) # Might need adjustment
    ax.set_xlabel('f1(x)')
    ax.set_ylabel('f2(x)')
    ax.legend()
    ax.grid(True)
    title = ax.set_title('MOPSO Iteration 0')

    # Update function for animation frames
    def update(frame):
        # Update swarm data
        if frame < len(swarm_history) and swarm_history[frame]:
            swarm_obj = np.array(swarm_history[frame])
            swarm_scatter.set_offsets(swarm_obj)
        else:
             swarm_scatter.set_offsets(np.empty((0, 2))) # Clear if no data

        # Update archive data
        if frame < len(archive_history) and archive_history[frame]:
            archive_obj = np.array(archive_history[frame])
            archive_scatter.set_offsets(archive_obj)
        else:
            archive_scatter.set_offsets(np.empty((0, 2))) # Clear if no data

        title.set_text(f'MOPSO Iteration {frame}')
        return swarm_scatter, archive_scatter, title

    # Create animation
    # Note: max_iter + 1 because history includes the initial state (iteration 0)
    print("Generating animation...")
    ani = animation.FuncAnimation(fig, update, frames=max_iter + 1, interval=100, blit=True) # interval in ms

    # Save animation
    # Might require ffmpeg (for mp4) or imagemagick (for gif) installed
    try:
        ani.save(filename, writer='pillow', fps=10) # Using pillow for GIF
        # Or use: ani.save(filename, writer='ffmpeg', fps=10) # For MP4
        print(f"Animation saved as '{filename}'")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Saving might require installing 'ffmpeg' (for MP4) or 'imagemagick/pillow' (for GIF).")
        print("Try: pip install pillow")

    plt.close(fig) # Close the figure after saving

if __name__ == "__main__":
    # Parameters for ZDT1
    N_DIM = 30
    BOUNDS = (0.0, 1.0)
    SWARM_SIZE = 100
    MAX_ITER = 50 # Iterations from MOPSO experimental comparison paper
    ARCHIVE_SIZE = 100 # Archive size from MOPSO experimental comparison paper
    N_GRID_DIVISIONS = 10 # Common value, can be tuned
    C1 = 2.0 # Common value
    C2 = 2.0 # Common value
    W_MAX = 0.9 # Common initial inertia
    W_MIN = 0.4 # Common final inertia
    SWARM_CSV = "mopso_swarm_history.csv"
    ARCHIVE_CSV = "mopso_archive_history.csv"

    print("Running MOPSO for ZDT1 and saving history...")
    final_positions, final_objectives = mopsopt_and_save_history(
        zdt1, N_DIM, BOUNDS, SWARM_SIZE, MAX_ITER, ARCHIVE_SIZE, N_GRID_DIVISIONS,
        C1, C2, W_MAX, W_MIN,
        swarm_hist_file=SWARM_CSV, archive_hist_file=ARCHIVE_CSV
    )

    print(f"\nExecution finished. Final Archive Size: {len(final_objectives)}")
    print(f"Swarm history saved to '{SWARM_CSV}'")
    print(f"Archive history saved to '{ARCHIVE_CSV}'")

    # Optional: Save final front separately
    try:
        np.savetxt('mopso_zdt1/mopso_zdt1_final_objectives_en.csv', np.array(final_objectives), delimiter=',', header='f1,f2', comments='')
        print("Final Pareto Front objectives saved to 'mopso_zdt1_final_objectives_en.csv'")
    except Exception as e:
        print(f"Error saving final objectives CSV: {e}")