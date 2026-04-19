from pathlib import Path

from algorithms.greedy import GreedyAlgorithm
from algorithms.ga_pso import GAPSOAlgorithm
from utils.config import load_config
from utils.converter import build_customers, build_depots
from utils.data_loader import read_cordeau_data_file, read_cordeau_solution_file, read_failures_file
from utils.results_io import save_clustering_result, save_routing_result
from utils.visualizer import visualize_instance, visualize_comparison
from scenario.simulator import run_simulation


def main() -> None:
    """Load p01, run the greedy algorithm and visualize the result."""
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / "data" / "raw" / "cordeau" / "p01"
    solution_file = base_dir / "data" / "raw" / "cordeau_sol" / "p01.res"
    failures_file = base_dir / "data" / "processed" / "failures" / "p01_seed41.json"

    # Load raw instance and reference solution
    instance = read_cordeau_data_file(str(data_file))
    reference_solution = read_cordeau_solution_file(str(solution_file), instance)
    failures = read_failures_file(str(failures_file))

    # Build domain entities
    customers = build_customers(instance)
    depots = build_depots(instance)

    cfg = load_config()

    # Run greedy algorithm
    # greedy = GreedyAlgorithm()
    # greedy_solution = greedy.solve(customers, depots)

    # Run GA+PSO algorithm
    ga_pso = GAPSOAlgorithm(cfg)
    ga_pso_solution = ga_pso.solve(customers, depots)

    results_dir = base_dir / "data" / "processed" / "results"
    clustering_file = results_dir / f"{data_file.name}_clusters.json"
    routing_file = results_dir / f"{data_file.name}_routes.json"

    save_clustering_result(
        output_path=str(clustering_file),
        instance_name=data_file.name,
        algorithm_name=str(ga_pso),
        clusters=ga_pso.last_clusters,
    )
    save_routing_result(
        output_path=str(routing_file),
        instance_name=data_file.name,
        algorithm_name=str(ga_pso),
        solution=ga_pso_solution,
    )
    
    run_simulation(
        # instance=instance,
        initial_solution=ga_pso_solution,
        failures=failures,
        instance_name=data_file.name,
        # output_dir=base_dir / "data" / "processed" / "simulations" / data_file.name,
    )

    print(f"Reference   : {reference_solution.objective:.2f}")
    # print(f"{greedy}  cost: {greedy_solution.total_cost():.2f}  feasible: {greedy_solution.is_feasible()}")
    print(f"{ga_pso}")
    print(f"  cost: {ga_pso_solution.total_cost():.2f}  feasible: {ga_pso_solution.is_feasible()}")
    print(f"Saved clusters : {clustering_file}")
    print(f"Saved routes   : {routing_file}")

    # Visualize
    visualize_instance(instance)
    visualize_comparison(
        instance,
        # [reference_solution, greedy_solution, ga_pso_solution],
        [reference_solution, ga_pso_solution],
        titles=[
            f"Reference (obj: {reference_solution.objective:.2f})",
            # f"Greedy (cost: {greedy_solution.total_cost():.2f})",
            f"GA+PSO (cost: {ga_pso_solution.total_cost():.2f})",
        ],
    )


if __name__ == "__main__":
    main()
