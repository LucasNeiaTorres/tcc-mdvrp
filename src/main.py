from pathlib import Path

from utils.data_loader import read_cordeau_data_file, read_cordeau_solution_file
from utils.visualizer import visualize_instance, visualize_solution


def main() -> None:
    """Load and visualize the p01 instance with its solution."""
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / "data" / "raw" / "cordeau" / "p01"
    solution_file = base_dir / "data" / "raw" / "cordeau_sol" / "p01.res"
    
    instance = read_cordeau_data_file(str(data_file))
    solution = read_cordeau_solution_file(str(solution_file))
    
    visualize_instance(instance)
    visualize_solution(instance, solution)


if __name__ == "__main__":
    main()
