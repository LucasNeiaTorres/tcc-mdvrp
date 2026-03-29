from pathlib import Path

from algorithms.greedy import GreedyAlgorithm
from utils.converter import build_customers, build_depots
from utils.data_loader import read_cordeau_data_file, read_cordeau_solution_file
from utils.visualizer import visualize_instance, visualize_comparison


def main() -> None:
    """Load p01, run the greedy algorithm and visualize the result."""
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / "data" / "raw" / "cordeau" / "p01"
    solution_file = base_dir / "data" / "raw" / "cordeau_sol" / "p01.res"

    # Load raw instance and reference solution
    instance = read_cordeau_data_file(str(data_file))
    reference_solution = read_cordeau_solution_file(str(solution_file))

    # Build domain entities
    customers = build_customers(instance)
    depots = build_depots(instance)

    # Run greedy algorithm
    algorithm = GreedyAlgorithm()
    solution = algorithm.solve(customers, depots)

    print(f"Algorithm   : {algorithm}")
    print(f"Total cost  : {solution.total_cost():.2f}")
    print(f"Feasible    : {solution.is_feasible()}")
    print(f"Reference   : {reference_solution.objective:.2f}")

    # Visualize
    visualize_instance(instance)
    visualize_comparison(
        instance,
        [reference_solution, solution],
        titles=[
            f"Reference (obj: {reference_solution.objective:.2f})",
            f"Greedy (cost: {solution.total_cost():.2f})",
        ],
    )


if __name__ == "__main__":
    main()
