import argparse
from pathlib import Path

from algorithms.greedy import GreedyAlgorithm
from algorithms.ccbc_pso import CCBCPSOAlgorithm
from utils.config import load_config
from utils.converter import build_customers, build_depots, load_instance
from utils.data_loader import read_cordeau_data_file, read_cordeau_solution_file, read_failures_file
from utils.results_io import save_clustering_result, save_routing_result
from utils.visualizer import visualize_instance, visualize_comparison
from scenario.simulator import SIMULATION_LOG_DIR, run_simulation
from tools.validate_simulation_log import validate_simulation_log


def main() -> int:
    base_dir = Path(__file__).parent.parent
    default_failures_file = base_dir / "data" / "processed" / "failures" / "p01_seed44_events40.json"

    parser = argparse.ArgumentParser(description="Run and visualize the MDVRP solver on one instance.")
    parser.add_argument("--instance", default="p01", metavar="NAME", help="Instance name (default: p01).")
    parser.add_argument(
        "--failures-file",
        default=str(default_failures_file),
        metavar="PATH",
        help="Path to the failures JSON file (default: data/processed/failures/p01_seed44_events40.json).",
    )
    args = parser.parse_args()

    loaded = load_instance(args.instance)
    customers = loaded.customers
    depots = loaded.depots
    reference_solution = loaded.reference
    
    """Load p01, run the greedy algorithm and visualize the result."""
    data_file = base_dir / "data" / "raw" / "cordeau" / "p01"
    solution_file = base_dir / "data" / "raw" / "cordeau_sol" / "p01.res"
    failures_file = Path(args.failures_file)

    # Load raw instance and reference solution
    instance = read_cordeau_data_file(str(data_file))
    reference_solution = read_cordeau_solution_file(str(solution_file), instance)
    failures = read_failures_file(str(failures_file))

    cfg = load_config()

    # Run greedy algorithm
    # greedy = GreedyAlgorithm()
    # greedy_solution = greedy.solve(customers, depots)

    # Run CCBC+PSO algorithm
    algorithm = CCBCPSOAlgorithm(cfg)
    solution = algorithm.solve(customers, depots)

    results_dir = base_dir / "data" / "processed" / "results"
    clustering_file = results_dir / f"{data_file.name}_clusters.json"
    routing_file = results_dir / f"{data_file.name}_routes.json"

    save_clustering_result(
        output_path=str(clustering_file), 
        instance_name=data_file.name,
        algorithm_name=str(algorithm),
        clusters=algorithm.last_clusters,
    )
    save_routing_result(
        output_path=str(routing_file),
        instance_name=data_file.name,
        algorithm_name=str(algorithm),
        solution=solution,
    )

    print(f"Reference   : {reference_solution.objective:.2f}")
    # print(f"{greedy}  cost: {greedy_solution.total_cost():.2f}  feasible: {greedy_solution.is_feasible()}")
    print(f"{algorithm}")
    print(f"  cost: {solution.total_cost():.2f}  feasible: {solution.is_feasible()}")
    print(f"Saved clusters : {clustering_file}")
    print(f"Saved routes   : {routing_file}")

    # Visualize
    visualize_instance(instance)
    visualize_comparison(
        instance,
        # [reference_solution, greedy_solution, ga_pso_solution],
        [reference_solution, solution],
        titles=[
            f"Reference (obj: {reference_solution.objective:.2f})",
            # f"Greedy (cost: {greedy_solution.total_cost():.2f})",
            f"GA+PSO (cost: {solution.total_cost():.2f})",
        ],
    )
    
    run_simulation(
        # instance=instance,
        initial_solution=solution,
        failures=failures,
        instance_name=data_file.name,
        algorithm=algorithm
        # output_dir=base_dir / "data" / "processed" / "simulations" / data_file.name,
    )

    log_path = SIMULATION_LOG_DIR / f"{data_file.name}_log.json"
    violations = validate_simulation_log(log_path)
    if violations:
        route_id, node_a, node_b, depart_time, arrival_time, block_time = violations[0]
        print(
            f"Validation failed: {len(violations)} blocked-edge violation(s) found in {log_path}\n"
            f"  first violation: route {route_id} used edge {node_a} <-> {node_b} "
            f"between t={depart_time:.3f}min and t={arrival_time:.3f}min, "
            f"but it was blocked at t={block_time:.3f}min"
        )
        return 1

    print(f"Validation passed: no blocked-edge violations found in {log_path}")
    return 0
    
    visualize_comparison(
        instance,
        # [reference_solution, greedy_solution, ga_pso_solution],
        [reference_solution, solution],
        titles=[
            f"Reference (obj: {reference_solution.objective:.2f})",
            # f"Greedy (cost: {greedy_solution.total_cost():.2f})",
            f"GA+PSO (cost: {solution.total_cost():.2f})",
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
