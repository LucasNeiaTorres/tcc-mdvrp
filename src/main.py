import argparse

from algorithms.greedy import GreedyAlgorithm
from algorithms.ccbc_pso import CCBCPSOAlgorithm
from utils.config import load_config
from utils.converter import load_instance
from utils.visualizer import visualize_instance, visualize_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and visualize the MDVRP solver on one instance.")
    parser.add_argument("--instance", default="p01", metavar="NAME", help="Instance name (default: p01).")
    args = parser.parse_args()

    loaded = load_instance(args.instance)
    customers = loaded.customers
    depots = loaded.depots
    reference_solution = loaded.reference

    cfg = load_config()

    # Run greedy algorithm
    # greedy = GreedyAlgorithm()
    # greedy_solution = greedy.solve(customers, depots)

    # Run CCBC+PSO algorithm
    algorithm = CCBCPSOAlgorithm(cfg)
    solution = algorithm.solve(customers, depots)

    if reference_solution is not None:
        print(f"Reference   : {reference_solution.objective:.2f}")
    print(f"{algorithm}")
    print(f"  cost: {solution.total_cost():.2f}  feasible: {solution.is_feasible()}")

    # Visualize
    visualize_instance(loaded.raw)
    if reference_solution is not None:
        visualize_comparison(
            loaded.raw,
            [reference_solution, solution],
            titles=[
                f"Reference (obj: {reference_solution.objective:.2f})",
                f"CCBC+PSO (cost: {solution.total_cost():.2f})",
            ],
        )
    else:
        visualize_comparison(
            loaded.raw,
            [solution],
            titles=[f"CCBC+PSO (cost: {solution.total_cost():.2f})"],
        )


if __name__ == "__main__":
    main()
