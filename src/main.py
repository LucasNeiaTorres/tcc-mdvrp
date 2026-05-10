import argparse
from pathlib import Path

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
    default_failures_file = None

    parser = argparse.ArgumentParser(description="Run and visualize the MDVRP solver on one instance.")
    parser.add_argument("--instance", default="p01", metavar="NAME", help="Instance name (default: p01).")
    parser.add_argument(
        "--failures-file",
        default=default_failures_file,
        metavar="PATH",
        help="Path to the failures JSON file (default: auto-detected for the selected instance).",
    )
    parser.add_argument(
        "--no-simulate",
        action="store_true",
        default=False,
        help="Skip the simulation phase (default: run simulation if a failures file is found).",
    )
    args = parser.parse_args()

    loaded = load_instance(args.instance)
    customers = loaded.customers
    depots = loaded.depots
    reference_solution = loaded.reference

    """Load the selected instance, run the algorithm and visualize the result."""
    data_file = base_dir / "data" / "raw" / "cordeau" / args.instance
    solution_file = base_dir / "data" / "raw" / "cordeau_sol" / f"{args.instance}.res"
    failures_dir = base_dir / "data" / "processed" / "failures"
    if args.failures_file is not None:
        provided_failures = Path(args.failures_file)
        if provided_failures.is_absolute():
            failures_file = provided_failures
        elif provided_failures.exists():
            failures_file = provided_failures
        elif (base_dir / provided_failures).exists():
            failures_file = base_dir / provided_failures
        else:
            failures_file = failures_dir / provided_failures.name
    else:
        default_failure_candidates = sorted(failures_dir.glob(f"{args.instance}_*.json"))
        if not default_failure_candidates:
            failures_file = None
        else:
            failures_file = default_failure_candidates[-1]

    # Load raw instance and reference solution
    instance = read_cordeau_data_file(str(data_file))
    reference_solution = read_cordeau_solution_file(str(solution_file), instance)
    failures = read_failures_file(str(failures_file)) if failures_file is not None else None

    cfg = load_config()

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
    print(f"{algorithm}")
    print(
        f"  cost: {solution.total_cost():.2f} "
        f"feasible: {solution.fully_feasible()} "
        f"(routes: {solution.is_feasible()}, fleet: {solution.fleet_is_feasible()})"
    )
    print(f"Saved clusters : {clustering_file}")
    print(f"Saved routes   : {routing_file}")

    # Visualize
    visualize_instance(instance)
    visualize_comparison(
        instance,
        [reference_solution, solution],
        titles=[
            f"Reference (obj: {reference_solution.objective:.2f})",
            f"GA+PSO (cost: {solution.total_cost():.2f})",
        ],
    )
    
    if failures is not None and not args.no_simulate:
        simulated_solution, history_log = run_simulation(
            # instance=instance,
            initial_solution=solution,
            failures=failures,
            instance_name=data_file.name,
            algorithm=algorithm
            # output_dir=base_dir / "data" / "processed" / "simulations" / data_file.name,
        )

        visualize_comparison(
            instance,
            [reference_solution, simulated_solution],
            titles=[
                f"Reference (obj: {reference_solution.objective:.2f})",
                f"GA+PSO after simulation (cost: {simulated_solution.total_cost():.2f})",
            ],
        )

        log_path = SIMULATION_LOG_DIR / f"{data_file.name}_log.json"
        validation_result = validate_simulation_log(log_path)
        blocked_edge_violations = validation_result["blocked_edge_violations"]
        unserved_customers = validation_result["unserved_customers"]

        if blocked_edge_violations:
            route_id, node_a, node_b, depart_time, arrival_time, block_time = blocked_edge_violations[0]
            print(
                f"Validation failed: {len(blocked_edge_violations)} blocked-edge violation(s) found in {log_path}\n"
                f"  first violation: route {route_id} used edge {node_a} <-> {node_b} "
                f"between t={depart_time:.3f}min and t={arrival_time:.3f}min, "
                f"but it was blocked at t={block_time:.3f}min"
            )
        if unserved_customers:
            print(
                f"Validation failed: {len(unserved_customers)} unserved customer(s) found in {log_path}\n"
                f"  unserved customers: {unserved_customers}"
            )

        if blocked_edge_violations or unserved_customers:
            return 1

        print(f"Validation passed: no blocked-edge violations found in {log_path}")
        print(f"Validation passed: all customers were served in {log_path}")
    else:
        if args.no_simulate:
            print("Simulation skipped (--no-simulate).")
        else:
            print("No failures file found; skipping simulation.")
    


if __name__ == "__main__":
    main()
