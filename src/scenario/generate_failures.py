"""Generate reproducible random failure scenarios for Cordeau instances."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import List

try:
	from src.scenario.models import FailureEvent
	from utils.data_loader import read_cordeau_data_file
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
	from utils.data_loader import read_cordeau_data_file

# Example usage:
"""
python3 src/scenario/generate_failures.py \
  --instance p01 \
  --seed 42 \
  --severity medium \
	--dod 0.10 \
  --max-time 120.0
This will generate a scenario with a DoD of 10% for instance p01, with trigger times between 0 and 120 minutes, and save it to data/processed/failures/p01_seed42_dod10.json.
"""

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
	node_ids = [customer.index for customer in instance.customers]
	total_edges = len(node_ids) * (len(node_ids) - 1)
	n_events = max(1, round(total_edges * args.dod))

	rng = random.Random(args.seed)
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
