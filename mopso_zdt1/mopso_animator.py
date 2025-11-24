import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
from collections import defaultdict

def load_history_from_csv(swarm_hist_file, archive_hist_file):
    """
    Loads swarm and archive objective history from CSV files.
    Returns:
        tuple: (swarm_history, archive_history, max_iteration)
               swarm_history[iter] = list of (f1, f2) tuples for swarm at iter
               archive_history[iter] = list of (f1, f2) tuples for archive at iter
    """
    swarm_history = defaultdict(list)
    archive_history = defaultdict(list)
    max_iteration = 0

    # Load swarm history
    try:
        with open(swarm_hist_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip header
            for row in reader:
                iteration = int(row[0])
                f1 = float(row[2])
                f2 = float(row[3])
                swarm_history[iteration].append((f1, f2))
                max_iteration = max(max_iteration, iteration)
        print(f"Loaded swarm history from '{swarm_hist_file}'")
    except FileNotFoundError:
        print(f"Error: Swarm history file '{swarm_hist_file}' not found.")
        return None, None, 0
    except Exception as e:
        print(f"Error reading swarm history file: {e}")
        return None, None, 0


    # Load archive history
    try:
        with open(archive_hist_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip header
            for row in reader:
                iteration = int(row[0])
                f1 = float(row[2])
                f2 = float(row[3])
                archive_history[iteration].append((f1, f2))
                # Max iteration should be consistent, but check anyway
                max_iteration = max(max_iteration, iteration)
        print(f"Loaded archive history from '{archive_hist_file}'")
    except FileNotFoundError:
        print(f"Error: Archive history file '{archive_hist_file}' not found.")
        return None, None, 0
    except Exception as e:
        print(f"Error reading archive history file: {e}")
        return None, None, 0

    # Convert defaultdicts to lists indexed by iteration (including iteration 0 if present)
    swarm_hist_list = [swarm_history.get(i, []) for i in range(max_iteration + 1)]
    archive_hist_list = [archive_history.get(i, []) for i in range(max_iteration + 1)]

    return swarm_hist_list, archive_hist_list, max_iteration

# --- Animation Function (Same as before) ---
def create_animation(swarm_history, archive_history, max_iter, filename="mopso_zdt1_animation.gif"):
    """
    Creates and saves an animation of the MOPSO execution from loaded history.
    Args:
        swarm_history (list): List of swarm objectives per iteration.
        archive_history (list): List of archive objectives per iteration.
        max_iter (int): Total number of iterations (frames).
        filename (str): Output filename for the animation (e.g., .gif, .mp4).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    x_front = np.linspace(0, 1, 100)
    y_front = 1 - np.sqrt(x_front)
    ax.plot(x_front, y_front, 'r--', label='Theoretical Pareto Front (ZDT1)', zorder=1)

    swarm_scatter = ax.scatter([], [], s=15, alpha=0.5, label='Swarm Particles', c='blue', zorder=2)
    archive_scatter = ax.scatter([], [], s=30, marker='*', label='Archive (Pareto Front)', c='green', zorder=3)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.2) # Adjust if needed
    ax.set_xlabel('f1(x)')
    ax.set_ylabel('f2(x)')
    ax.legend()
    ax.grid(True)
    title = ax.set_title('MOPSO Iteration 0')

    def update(frame):
        # Update swarm data
        if frame < len(swarm_history) and swarm_history[frame]:
            swarm_obj = np.array(swarm_history[frame])
            # Ensure shape is (N, 2) even if only one particle
            if swarm_obj.ndim == 1:
                 swarm_obj = swarm_obj.reshape(1, -1)
            if swarm_obj.shape[1] == 2: # Check if data is valid
                 swarm_scatter.set_offsets(swarm_obj)
            else:
                 swarm_scatter.set_offsets(np.empty((0, 2)))
        else:
             swarm_scatter.set_offsets(np.empty((0, 2)))

        # Update archive data
        if frame < len(archive_history) and archive_history[frame]:
            archive_obj = np.array(archive_history[frame])
            # Ensure shape is (N, 2)
            if archive_obj.ndim == 1:
                archive_obj = archive_obj.reshape(1, -1)
            if archive_obj.shape[1] == 2: # Check if data is valid
                archive_scatter.set_offsets(archive_obj)
            else:
                 archive_scatter.set_offsets(np.empty((0, 2)))
        else:
            archive_scatter.set_offsets(np.empty((0, 2)))

        title.set_text(f'MOPSO Iteration {frame}')
        # Use return value for blitting
        return swarm_scatter, archive_scatter, title

    print("Generating animation...")
    # Frames go from 0 to max_iter (inclusive), so max_iter + 1 frames
    ani = animation.FuncAnimation(fig, update, frames=max_iter + 1, interval=100, blit=True)

    try:
        ani.save(filename.replace(".gif", ".mp4"), writer='ffmpeg', fps=2) # fps = 1000 / interval
        print(f"Animation saved as '{filename}'")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Saving might require installing 'ffmpeg' or 'imagemagick/pillow'. Try: pip install pillow")

    plt.close(fig)

# --- Main execution block for animator ---
if __name__ == "__main__":
    SWARM_CSV = "mopso_swarm_history.csv"
    ARCHIVE_CSV = "mopso_archive_history.csv"
    ANIMATION_FILE = "mopso_zdt1_animation_from_csv.gif"

    print("Loading history from CSV files...")
    swarm_hist, archive_hist, max_iter_from_data = load_history_from_csv(SWARM_CSV, ARCHIVE_CSV)

    if swarm_hist is not None and archive_hist is not None:
        print(f"History loaded for {max_iter_from_data} iterations.")
        create_animation(swarm_hist, archive_hist, max_iter_from_data, filename=ANIMATION_FILE)
    else:
        print("Could not load history. Animation not generated.")