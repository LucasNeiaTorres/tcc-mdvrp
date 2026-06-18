"""Generate reproducible random failure scenarios for Cordeau instances."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import List

try:
	from src.scenario.models import FailureEvent
	from utils.data_loader import read_cordeau_data_file, read_cordeau_solution_file, read_json_solution_file
except ModuleNotFoundError:
	# Allow direct execution: python src/scenario/generate_failures.py
	repo_root = Path(__file__).resolve().parents[2]
	src_dir = Path(__file__).resolve().parents[1]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))
	if str(src_dir) not in sys.path:
		sys.path.insert(0, str(src_dir))
	try:
		from src.scenario.models import FailureEvent
	except ModuleNotFoundError:
		from scenario.models import FailureEvent
	from utils.data_loader import read_cordeau_data_file, read_cordeau_solution_file, read_json_solution_file

# Example usage (random uniform):
#   python3 src/scenario/generate_failures.py --instance p01 --seed 42 --dod 0.10 --max-time 120.0
#
# Example usage (Collapse Zones — requires a static solution file):
#   python3 src/scenario/generate_failures.py \
#     --instance p01 --seed 42 --severity medium \
#     --dod 0.20 --edod 0.3 --max-time 120.0 \
#     --routes-file data/processed/solutions/p01.txt

def _default_data_file(instance: str) -> Path:
	base_dir = Path(__file__).resolve().parents[2]
	return base_dir / "data" / "raw" / "cordeau" / instance


def _default_output_file(instance: str, seed: int, dod: float) -> Path:
	base_dir = Path(__file__).resolve().parents[2]
	dod_percent = int(dod * 100)
	return base_dir / "data" / "processed" / "failures" / f"{instance}_seed{seed}_dod{dod_percent}.json"


def generate_events(
	node_ids: List[int],
	rng: random.Random,
	n_events: int,
	max_time: float,
	dod: float,
) -> List[FailureEvent]:
	"""Generate sorted edge_block events with random trigger times and node pairs."""
	if n_events < 1:
		raise ValueError("n_events must be >= 1")
	if max_time <= 0:
		raise ValueError("max_time must be > 0")
	if len(node_ids) < 2:
		raise ValueError("Need at least 2 nodes to generate edge events")
	if not (0.0 <= dod <= 1.0):
		raise ValueError("dod must be between 0.0 and 1.0")

	events: List[FailureEvent] = []
	used_edges = set()
	while len(events) < n_events:
		node_a, node_b = rng.sample(node_ids, 2)
		edge_key = (node_a, node_b)
		if edge_key in used_edges:
			continue
		used_edges.add(edge_key)
		trigger_time = round(rng.uniform(0.0, max_time), 1)
		events.append(
			FailureEvent(
				trigger_time=trigger_time,
				type="edge_block",
				node_a=node_a,
				node_b=node_b,
			)
		)

	events.sort(key=lambda e: e.trigger_time)
	return events

def _sample_t_block(rng: random.Random, edod: float, max_time: float) -> float:
    """Sample a trigger time whose expected value is EDOD * max_time.
    Uses a flattened Beta(1.2, β) distribution to allow variance (early events in high EDOD).
    """
    if edod <= 0.0:
        return 0.0
    if edod >= 1.0:
        return round(max_time, 1)
    
    alpha = 1.2  # Curva mais achatada para espalhar melhor os eventos
    beta_param = alpha * (1.0 - edod) / edod
    return round(rng.betavariate(alpha, beta_param) * max_time, 1)


def generate_collapse_zones(
	routes: object,
	G: object,
	DOD: float,
	EDOD: float,
	max_time: float,
	rng: random.Random,
) -> List[FailureEvent]:
	"""Generate failure events via spatial Collapse Zones, avoiding Evasion Bias.

	Only edges that the fleet actually travels (extracted from *routes*) are
	candidates for failure.  Disaster epicenters are drawn randomly from the
	graph nodes; every active edge whose endpoint lies within the epicenter
	radius R (Euclidean) is blocked.  All edges inside the same zone share a
	single trigger time derived from EDOD so the temporal distribution of events
	matches the expected degree of dynamism.

	Args:
		routes: Static solution (CordeauSolution) with the fleet's planned routes.
		G:      Problem instance (CordeauInstance) providing node coordinates.
		DOD:    Degree of Dynamism — fraction of active edges that must fail.
		EDOD:   Expected Degree of Dynamism — controls when failures happen:
		        0 → all events near t=0; 1 → all events near t=max_time.
		max_time: Upper bound for trigger times (same unit as the simulation).
		rng:    Seeded RNG for reproducibility.

	Returns:
		List of FailureEvent tuples sorted by trigger_time.
	"""
	all_nodes = {node.index: node for node in G.customers + G.depots}

	# Directed active edges: every consecutive pair in each route's full sequence
	active_edges: set = set()
	for route in routes.routes:
		full_seq = [route.depot] + list(route.nodes) + [route.depot]
		for i in range(len(full_seq) - 1):
			u, v = full_seq[i], full_seq[i + 1]
			if u != v:
				active_edges.add((u, v))

	if not active_edges:
		raise ValueError("No active edges found in routes")

	n_target = min(max(1, round(len(active_edges) * DOD)), len(active_edges))

	xs = [node.x for node in all_nodes.values()]
	ys = [node.y for node in all_nodes.values()]
	diagonal = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) or 1.0

	node_list = list(all_nodes.keys())
	failed_edges: set = set()
	events: List[FailureEvent] = []
	stale = 0

	while len(failed_edges) < n_target and stale < 10:
		prev_count = len(failed_edges)
		epicenter = all_nodes[rng.choice(node_list)]
		R = rng.uniform(0.02, 0.10) * diagonal
		t_block = _sample_t_block(rng, EDOD, max_time)

		for u, v in active_edges:
			if (u, v) in failed_edges:
				continue
			node_u = all_nodes.get(u)
			node_v = all_nodes.get(v)
			if node_u is None or node_v is None:
				continue
			dist_u = math.sqrt((node_u.x - epicenter.x) ** 2 + (node_u.y - epicenter.y) ** 2)
			dist_v = math.sqrt((node_v.x - epicenter.x) ** 2 + (node_v.y - epicenter.y) ** 2)
			if dist_u <= R or dist_v <= R:
				failed_edges.add((u, v))
				events.append(FailureEvent(
					trigger_time=t_block,
					type="edge_block",
					node_a=u,
					node_b=v,
				))

		stale = 0 if len(failed_edges) > prev_count else stale + 1

	events.sort(key=lambda e: e.trigger_time)
	return events


def build_payload(instance: str, seed: int, severity: str, dod: float, edod: float, events: List[FailureEvent]) -> dict:
	"""Build the JSON payload in the requested schema."""
	return {
		"metadata": {
			"instance": instance,
			"seed": seed,
			"severity": severity,
			"dod": dod,
			"edod": round(edod, 4),
			"generated_at": str(date.today()),
		},
		"events": [asdict(event) for event in events],
	}


def _dod_type(value: str) -> float:
	try:
		dod = float(value)
	except ValueError as exc:
		raise argparse.ArgumentTypeError("dod must be a float between 0.0 and 1.0") from exc
	if not (0.0 <= dod <= 1.0):
		raise argparse.ArgumentTypeError("dod must be between 0.0 and 1.0")
	return dod


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate random edge-failure scenario JSON.")
	parser.add_argument("--instance", default="p01", help="Cordeau instance name (e.g., p01, p23)")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
	parser.add_argument(
		"--severity",
		default="medium",
		choices=["low", "medium", "high"],
		help="Scenario severity label stored in metadata",
	)
	parser.add_argument(
		"--dod",
		type=_dod_type,
		default=0.10,
		help="Degree of dynamism (0.0 to 1.0) as a disruption ratio",
	)
	parser.add_argument(
		"--max-time",
		type=float,
		default=60.0,
		help="Upper bound for trigger_time sampling",
	)
	parser.add_argument(
		"--data-file",
		type=Path,
		default=None,
		help="Optional path to the Cordeau data file",
	)
	parser.add_argument(
		"--routes-file",
		type=Path,
		default=None,
		help="Path to a static solution file; activates spatial Collapse Zones generation",
	)
	parser.add_argument(
		"--edod",
		type=float,
		default=0.5,
		help="Expected Degree of Dynamism (0.0–1.0); shapes trigger-time distribution in Collapse Zones mode",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Output JSON path",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	data_file = args.data_file or _default_data_file(args.instance)
	output_file = args.output or _default_output_file(args.instance, args.seed, args.dod)

	instance = read_cordeau_data_file(str(data_file))
	rng = random.Random(args.seed)

	if args.routes_file is not None:
		if str(args.routes_file).endswith(".json"):
			solution = read_json_solution_file(str(args.routes_file))
		else:
			solution = read_cordeau_solution_file(str(args.routes_file), instance=instance)
		events = generate_collapse_zones(
			routes=solution,
			G=instance,
			DOD=args.dod,
			EDOD=args.edod,
			max_time=args.max_time,
			rng=rng,
		)
	else:
		node_ids = [customer.index for customer in instance.customers]
		total_edges = len(node_ids) * (len(node_ids) - 1)
		n_events = max(1, round(total_edges * args.dod))
		events = generate_events(
			node_ids=node_ids,
			rng=rng,
			n_events=n_events,
			max_time=args.max_time,
			dod=args.dod,
		)

	soma_tempos = sum(evento.trigger_time for evento in events)
	edod = (soma_tempos / len(events)) / args.max_time
 
	payload = build_payload(
		instance=args.instance,
		seed=args.seed,
		severity=args.severity,
		dod=args.dod,
		edod=edod,
		events=events,
	)

	output_file.parent.mkdir(parents=True, exist_ok=True)
	with output_file.open("w", encoding="utf-8") as f:
		json.dump(payload, f, ensure_ascii=True, indent=2)
		f.write("\n")

	print(f"Scenario written to: {output_file}")


if __name__ == "__main__":
	main()
