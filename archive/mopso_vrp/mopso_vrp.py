import numpy as np
import random
import math
import csv # For history saving
from collections import defaultdict # For loading history

# --- VRP Instance Data (Example) ---
# Coordinates (depot is index 0)
coords = np.array([
    [0, 0],   # Depot 0
    [10, 5],  # Customer 1
    [-5, 8],  # Customer 2
    [8, -6],  # Customer 3
    [-3, -7], # Customer 4
    [2, 9]    # Customer 5
])
demands = np.array([0, 10, 8, 12, 6, 9]) # Demands (depot demand is 0)
vehicle_capacity = 25
num_customers = len(coords) - 1 # Exclude depot

# --- Distance Calculation ---
def calculate_distance_matrix(coords):
    n_locations = len(coords)
    dist_matrix = np.zeros((n_locations, n_locations))
    for i in range(n_locations):
        for j in range(i + 1, n_locations):
            dist = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix

distance_matrix = calculate_distance_matrix(coords)

# --- VRP Decoding Heuristic (Simple Example: Priority + Sequential) ---
def decode_vrp(priorities, demands, capacity, dist_matrix):
    """
    Decodes a priority vector into VRP routes using a simple sequential heuristic.
    Args:
        priorities (np.array): Particle's position (lower value = higher priority).
                               Length num_customers. Index corresponds to customer ID - 1.
        demands (np.array): Demands including depot (index 0).
        capacity (float): Vehicle capacity.
        dist_matrix (np.array): Precomputed distances.
    Returns:
        list: A list of routes, where each route is a list of customer indices (e.g., [1, 3, 2]).
    """
    routes = []
    # Customer indices sorted by priority (add 1 to map back to original indices)
    # Argsort gives indices of the *sorted* array; we need indices *of the original* based on sort order
    customer_indices_sorted = np.argsort(priorities) + 1
    customers_to_visit = list(customer_indices_sorted)

    while customers_to_visit:
        current_route = []
        current_load = 0
        last_visited = 0 # Start from depot

        # Greedily add customers from the priority list to the current route
        possible_next_customers = list(customers_to_visit) # Copy to allow removal while iterating
        for customer_idx in possible_next_customers:
            demand = demands[customer_idx]
            if current_load + demand <= capacity:
                current_route.append(customer_idx)
                current_load += demand
                customers_to_visit.remove(customer_idx) # Mark as visited
                last_visited = customer_idx # Keep track for distance (simplistic, better heuristics exist)
            # If a customer doesn't fit, try the next in priority for this vehicle
            # This is a very basic heuristic. More complex ones might reorder.

        if current_route: # Only add non-empty routes
            routes.append(current_route)

    return routes

# --- VRP Objective Functions ---
def calculate_total_distance(routes, dist_matrix):
    """Calculates total distance for a set of routes."""
    total_dist = 0
    for route in routes:
        if not route: continue
        # Distance from depot to first customer
        total_dist += dist_matrix[0, route[0]]
        # Distance between customers in the route
        for i in range(len(route) - 1):
            total_dist += dist_matrix[route[i], route[i+1]]
        # Distance from last customer back to depot
        total_dist += dist_matrix[route[-1], 0]
    return total_dist

def calculate_num_vehicles(routes):
    """Calculates the number of vehicles used (number of routes)."""
    return len(routes)

# --- Objective Function Wrapper for MOPSO ---
def vrp_objectives(priority_vector):
    """
    Takes the particle's position (priorities) and returns the VRP objectives.
    This function replaces zdt1 in the MOPSO structure.
    """
    # Decode the priority vector into routes
    routes = decode_vrp(priority_vector, demands, vehicle_capacity, distance_matrix)

    # Calculate objectives based on the routes
    f1 = calculate_total_distance(routes, distance_matrix)
    f2 = calculate_num_vehicles(routes)

    return f1, f2

# --- Helper Functions (Dominance) ---
# (Same as before)
def dominates(obj1, obj2):
    # Assumes minimization for both f1 (distance) and f2 (vehicles)
    at_least_one_better = obj1[0] < obj2[0] or obj1[1] < obj2[1]
    none_worse = obj1[0] <= obj2[0] and obj1[1] <= obj2[1]
    return at_least_one_better and none_worse

# --- Particle Class (Modified for VRP) ---
class Particle:
    def __init__(self, n_dim, bounds):
        """ n_dim is now num_customers """
        self.position = np.random.uniform(bounds[0], bounds[1], n_dim)

        print(self.position)

        vel_init_range = abs(bounds[1]-bounds[0]) * 0.1
        self.velocity = np.random.uniform(-vel_init_range, vel_init_range, n_dim)
        self.pbest_position = self.position.copy()

        # Evaluate initial position
        self.objectives = vrp_objectives(self.position)
        self.pbest_objectives = self.objectives

    def update_pbest(self):
        """ Evaluate current position and update pbest based on dominance """
        # Evaluate objectives for the current position *before* comparison
        new_objectives = vrp_objectives(self.position)
        if dominates(new_objectives, self.pbest_objectives):
            self.pbest_position = self.position.copy()
            self.pbest_objectives = new_objectives
        # Store current objectives
        self.objectives = new_objectives

# --- Archive and Grid Management ---
# (The ArchiveManager class remains conceptually the same as before)
# It stores particle positions (priority vectors) and works with
# the objective values (distance, num_vehicles) for grid and dominance checks.
class ArchiveManager:
    # (Use the same ArchiveManager class code from the previous ZDT1 example)
    # No changes needed here as it operates on positions and objectives generically.
    # Just ensure it's copied/included here.
    def __init__(self, max_size, n_grid_divisions):
        self.max_size = max_size
        self.n_grid_divisions = n_grid_divisions
        self.archive_positions = []
        self.archive_objectives = []
        self.grid_lower_bounds = np.full(2, np.inf)
        self.grid_upper_bounds = np.full(2, -np.inf)
        self.grid_cell_indices = {}

    def update_archive(self, particle_position, particle_objectives):
        is_dominated_by_archive = False
        dominated_indices_in_archive = []
        for i in range(len(self.archive_objectives)):
            if dominates(self.archive_objectives[i], particle_objectives):
                is_dominated_by_archive = True
                break
            if dominates(particle_objectives, self.archive_objectives[i]):
                dominated_indices_in_archive.append(i)

        if not is_dominated_by_archive:
            for index in sorted(dominated_indices_in_archive, reverse=True):
                del self.archive_positions[index]
                del self.archive_objectives[index]
            self.archive_positions.append(particle_position.copy())
            self.archive_objectives.append(particle_objectives)
            if len(self.archive_objectives) > self.max_size:
                self._prune_archive()

    def _update_grid_bounds(self):
        if not self.archive_objectives:
            # Handle empty archive case - maybe set default bounds?
            self.grid_lower_bounds = np.array([0.0, 0.0]) # Example default lower
            self.grid_upper_bounds = np.array([1000.0, 10.0]) # Example default upper
            return
        objectives_array = np.array(self.archive_objectives)
        self.grid_lower_bounds = np.min(objectives_array, axis=0)
        self.grid_upper_bounds = np.max(objectives_array, axis=0)
        range_bounds = self.grid_upper_bounds - self.grid_lower_bounds
        range_bounds[range_bounds <= 0] = 1.0 # Avoid zero or negative range
        margin = range_bounds * 0.1
        # Prevent lower bounds from becoming negative if min was 0
        self.grid_lower_bounds = np.maximum(0.0, self.grid_lower_bounds - margin)
        self.grid_upper_bounds += margin

    def _get_grid_index(self, objectives):
        # Handle cases where objectives might be outside the calculated bounds
        if np.any(objectives < self.grid_lower_bounds) or np.any(objectives > self.grid_upper_bounds):
            # Option 1: Clamp objectives to bounds for index calculation
            clamped_objectives = np.clip(objectives, self.grid_lower_bounds, self.grid_upper_bounds)
            # Option 2: Assign to a special 'out-of-bounds' index (None here)
            # return None # Returning None might break pruning/selection
            objectives_for_index = clamped_objectives # Use clamped for index calc
        else:
            objectives_for_index = objectives

        grid_range = self.grid_upper_bounds - self.grid_lower_bounds
        grid_range[grid_range == 0] = 1.0
        normalized_pos = (objectives_for_index - self.grid_lower_bounds) / grid_range
        indices = np.floor(normalized_pos * self.n_grid_divisions).astype(int)
        indices = np.clip(indices, 0, self.n_grid_divisions - 1)
        return tuple(indices)


    def _build_grid_indices(self):
        self._update_grid_bounds()
        self.grid_cell_indices.clear()
        for i, obj in enumerate(self.archive_objectives):
            grid_idx = self._get_grid_index(obj)
            if grid_idx is not None:
                if grid_idx not in self.grid_cell_indices:
                    self.grid_cell_indices[grid_idx] = []
                self.grid_cell_indices[grid_idx].append(i)

    def _prune_archive(self):
        self._build_grid_indices() # Ensure grid is up-to-date
        while len(self.archive_objectives) > self.max_size:
            if not self.grid_cell_indices: break # Safety check

            # Find cell with the most members
            # Use .get(k, []) to handle potential empty lists safely during sorting
            most_crowded_cell = max(self.grid_cell_indices, key=lambda k: len(self.grid_cell_indices.get(k, [])))

            # Check if the supposedly most crowded cell actually has members
            if not self.grid_cell_indices.get(most_crowded_cell):
                 # This can happen if pruning removed the last member of this cell
                 # in a previous step within this while loop. Remove the empty cell key.
                 if most_crowded_cell in self.grid_cell_indices:
                     del self.grid_cell_indices[most_crowded_cell]
                 continue # Re-evaluate the most crowded cell

            # Select random member from the crowded cell
            index_to_remove = random.choice(self.grid_cell_indices[most_crowded_cell])

            # Remove from grid mapping first
            self.grid_cell_indices[most_crowded_cell].remove(index_to_remove)
            if not self.grid_cell_indices[most_crowded_cell]:
                del self.grid_cell_indices[most_crowded_cell]

            # Remove from actual archive lists by index
            # This requires careful index handling or rebuilding the grid map frequently
            # Simple approach: delete by index and rebuild grid map *after* the loop finishes
            del self.archive_positions[index_to_remove]
            del self.archive_objectives[index_to_remove]

            # Rebuild grid map after removal for consistency if loop continues
            # This is less efficient but simpler to implement correctly in a sketch
            self._build_grid_indices()


    def select_leader(self):
        if not self.archive_objectives: return None
        # Ensure grid map is built if needed (e.g., first call or after pruning)
        if not self.grid_cell_indices or len(self.archive_objectives) <= self.max_size:
             self._build_grid_indices()
             if not self.grid_cell_indices: return None # No valid cells

        # Calculate fitness only for cells that currently have members
        cell_fitness = {idx: 1.0 / (len(indices) + 1)
                        for idx, indices in self.grid_cell_indices.items() if indices}
        if not cell_fitness: return None # No cells with members found

        total_fitness = sum(cell_fitness.values())

        if total_fitness <= 0: # Handle cases with zero fitness (e.g., all cells single member)
             selected_cell_idx = random.choice(list(cell_fitness.keys()))
        else:
            # Roulette Wheel
            pick = random.uniform(0, total_fitness)
            current_fitness_sum = 0
            selected_cell_idx = None
            sorted_cell_indices = sorted(cell_fitness.keys()) # For consistency
            for cell_idx in sorted_cell_indices:
                current_fitness_sum += cell_fitness[cell_idx]
                if current_fitness_sum >= pick:
                    selected_cell_idx = cell_idx
                    break
            # Fallback if roulette fails (shouldn't happen with correct logic)
            if selected_cell_idx is None:
                 selected_cell_idx = random.choice(list(cell_fitness.keys()))

        # Select random member from the chosen cell
        leader_archive_index = random.choice(self.grid_cell_indices[selected_cell_idx])
        return self.archive_positions[leader_archive_index]


# --- MOPSO Algorithm (adapted for VRP objective function) ---
def mopsopt_vrp(n_customers, bounds, swarm_size, max_iter, archive_size, n_grid_divisions, c1, c2, w_max, w_min,
                swarm_hist_file="swarm_history_vrp.csv", archive_hist_file="archive_history_vrp.csv"):
    """
    Executes MOPSO for VRP using priority encoding.
    Args:
        n_customers: Number of customers (dimensionality).
        bounds: Bounds for priority values (e.g., (0.0, 1.0)).
        ... (other MOPSO parameters)
    """
    n_dim = n_customers # Dimensionality is number of customers
    swarm = [Particle(n_dim, bounds) for _ in range(swarm_size)]
    archive_manager = ArchiveManager(archive_size, n_grid_divisions)

    # Initialize archive
    for p in swarm:
        archive_manager.update_archive(p.position, p.objectives)

    # --- History Saving Setup ---
    with open(swarm_hist_file, 'w', newline='') as sf, open(archive_hist_file, 'w', newline='') as af:
        swarm_writer = csv.writer(sf)
        archive_writer = csv.writer(af)
        swarm_writer.writerow(['iteration', 'particle_index', 'f1_distance', 'f2_vehicles'])
        archive_writer.writerow(['iteration', 'archive_index', 'f1_distance', 'f2_vehicles'])

        # Save initial state (Iteration 0)
        for idx, p in enumerate(swarm):
            swarm_writer.writerow([0, idx, p.objectives[0], p.objectives[1]])
        for idx, obj in enumerate(archive_manager.archive_objectives):
            archive_writer.writerow([0, idx, obj[0], obj[1]])

        # --- Main loop ---
        for t in range(max_iter):
            iteration_num = t + 1
            w = w_max - (w_max - w_min) * t / max_iter
            archive_manager._build_grid_indices()

            for i in range(swarm_size):
                particle = swarm[i]
                leader_position = archive_manager.select_leader()
                if leader_position is None:
                    leader_position = particle.pbest_position

                # --- Update velocity and position (acting on priority vector) ---
                r1, r2 = np.random.rand(n_dim), np.random.rand(n_dim)
                cognitive_velocity = c1 * r1 * (particle.pbest_position - particle.position)
                social_velocity = c2 * r2 * (leader_position - particle.position)
                particle.velocity = w * particle.velocity + cognitive_velocity + social_velocity

                max_vel = (bounds[1] - bounds[0]) * 0.5
                particle.velocity = np.clip(particle.velocity, -max_vel, max_vel)

                particle.position += particle.velocity
                particle.position = np.clip(particle.position, bounds[0], bounds[1])
                # --- Crucial Step: Evaluation involves decoding ---
                # update_pbest now calls vrp_objectives, which includes decoding
                particle.update_pbest()

                # --- Save swarm state ---
                swarm_writer.writerow([iteration_num, i, particle.objectives[0], particle.objectives[1]])

                # --- Update archive ---
                # Pass the *position* (priority vector) and the *calculated objectives*
                archive_manager.update_archive(particle.position, particle.objectives)

            # --- Save archive state ---
            for idx, obj in enumerate(archive_manager.archive_objectives):
                archive_writer.writerow([iteration_num, idx, obj[0], obj[1]])

            if iteration_num % 20 == 0: # Adjusted print frequency
                print(f"Iter {iteration_num}/{max_iter}, Archive: {len(archive_manager.archive_objectives)}")

    return archive_manager.archive_positions, archive_manager.archive_objectives


# --- Main execution block for VRP ---
if __name__ == "__main__":
    # VRP Parameters
    N_CUSTOMERS = num_customers # Use from VRP instance data
    PRIORITY_BOUNDS = (0.0, 1.0) # Bounds for the priority vector elements

    # MOPSO Parameters (adjust as needed for VRP)
    SWARM_SIZE_VRP = 50
    MAX_ITER_VRP = 100 # VRP decoding can be slower
    ARCHIVE_SIZE_VRP = 50
    N_GRID_DIVISIONS_VRP = 7 # Might need tuning based on objective ranges
    C1_VRP = 1.8 # Common to adjust coefficients slightly
    C2_VRP = 1.8
    W_MAX_VRP = 0.9
    W_MIN_VRP = 0.4
    SWARM_CSV_VRP = "mopso_vrp/mopso_vrp_swarm_history.csv"
    ARCHIVE_CSV_VRP = "mopso_vrp/mopso_vrp_archive_history.csv"

    print("Running MOPSO for VRP...")
    final_vrp_positions, final_vrp_objectives = mopsopt_vrp(
        N_CUSTOMERS, PRIORITY_BOUNDS, SWARM_SIZE_VRP, MAX_ITER_VRP, ARCHIVE_SIZE_VRP,
        N_GRID_DIVISIONS_VRP, C1_VRP, C2_VRP, W_MAX_VRP, W_MIN_VRP,
        swarm_hist_file=SWARM_CSV_VRP, archive_hist_file=ARCHIVE_CSV_VRP
    )

    print(f"\nExecution finished. Final VRP Archive Size: {len(final_vrp_objectives)}")
    print(f"Swarm history saved to '{SWARM_CSV_VRP}'")
    print(f"Archive history saved to '{ARCHIVE_CSV_VRP}'")

    # Optional: Plot the final Pareto front for VRP objectives
    try:
        import matplotlib.pyplot as plt
        objectives_array = np.array(final_vrp_objectives)
        # Ensure correct shape for plotting
        if objectives_array.ndim == 1:
            objectives_array = objectives_array.reshape(1, -1)

        if objectives_array.shape[1] == 2: # Check if we have 2 objectives
            plt.figure(figsize=(8, 6))
            # Use integers for number of vehicles on the axis if possible
            unique_vehicles = sorted(list(set(objectives_array[:, 1].astype(int))))
            if len(unique_vehicles) > 0 and len(unique_vehicles) < 15 : # Avoid too many ticks
                 plt.yticks(unique_vehicles)

            plt.scatter(objectives_array[:, 0], objectives_array[:, 1], s=20, label='Pareto Front (VRP)')
            plt.title('MOPSO - VRP Results (Distance vs Vehicles)')
            plt.xlabel('Total Distance')
            plt.ylabel('Number of Vehicles')
            plt.legend()
            plt.grid(True)
            plt.savefig('mopso_vrp_pareto_front.png')
            print("VRP Pareto Front plot saved to 'mopso_vrp_pareto_front.png'")
            plt.show()
        else:
             print("Plotting requires exactly 2 objectives.")

    except ImportError:
        print("\nMatplotlib not found. Cannot plot results.")
    except Exception as e:
        print(f"Error during plotting: {e}")