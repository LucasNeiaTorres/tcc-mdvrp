"""Stage-3 global cross-depot repair utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from algorithms.ga_local_search import local_search_stage1_intra
from core.entities import Customer, Route
from scenario.state import VehicleState


@dataclass(frozen=True)
class _BestInsertion:
	"""Internal holder for the globally cheapest feasible insertion."""

	vehicle_id: int
	suffix_start: int
	edge_position: int
	delta_cost: float


def _matrix_distance(
	distance_matrix: Sequence[Sequence[float]],
	node_i: int,
	node_j: int,
) -> float:
	"""Return distance_matrix[node_i][node_j] with defensive error reporting."""
	try:
		return float(distance_matrix[node_i][node_j])
	except Exception as exc:  # pragma: no cover - defensive branch
		raise ValueError(
			"distance_matrix must support distance_matrix[node_i][node_j] indexing."
		) from exc


def _collect_customers(vehicle_states: dict[int, VehicleState]) -> dict[int, Customer]:
	"""Build a node-indexed customer catalog from all vehicle routes."""
	catalog: dict[int, Customer] = {}
	for state in vehicle_states.values():
		for customer in state.route.customers:
			catalog.setdefault(customer.index, customer)
		for node_index, customer in state.customers_by_index.items():
			catalog.setdefault(node_index, customer)
	return catalog


def _is_completed_state(state: VehicleState) -> bool:
	"""Normalize completion check across legacy and newer state encodings."""
	return state.status == "completed" or state.is_complete()


def _insert_target_on_suffix(
	*,
	customers: list[Customer],
	suffix_start: int,
	edge_position: int,
	target_customer: Customer,
) -> tuple[list[Customer], list[Customer]]:
	"""Insert target after one edge in the unfrozen suffix."""
	suffix = list(customers[suffix_start:])
	insertion_index = edge_position + 1
	suffix.insert(insertion_index, target_customer)
	return customers[:suffix_start] + suffix, suffix


def _route_with_same_history(state: VehicleState, customers: list[Customer]) -> Route:
	"""Create a route preserving historical wasted time/distance fields."""
	return Route(
		depot=state.route.depot,
		customers=customers,
		wasted_duration=state.route.wasted_duration,
		wasted_distance=state.route.wasted_distance,
	)


def _clone_state_with_route(state: VehicleState, route: Route) -> VehicleState:
	"""Clone a vehicle state while swapping in a new route."""
	updated = VehicleState(
		route_id=state.route_id,
		route=route,
		current_node_index=state.current_node_index,
		next_stop_index=state.next_stop_index,
		last_event_time_min=state.last_event_time_min,
		visited_customer_ids=set(state.visited_customer_ids),
		pending_customer_ids=set(),
		capacity_total=state.capacity_total,
		load_current=state.load_current,
		customers_by_index={},
		status=state.status,
	)
	updated.customers_by_index = {customer.index: customer for customer in route.customers}
	updated.pending_customer_ids = (
		{customer.index for customer in route.customers} - updated.visited_customer_ids
	)
	return updated


def stage3_global_cross_depot_repair(
	target_node: int,
	vehicle_states: dict[int, VehicleState],
	distance_matrix: Sequence[Sequence[float]],
	blocked_vehicle_id: int,
) -> VehicleState | None:
	"""
	Stage 3 fallback: global cross-depot rescue using cheapest insertion + VND.

	Rules implemented from the provided specification:
	- Candidate vehicles are scanned globally (all depots), excluding the blocked one.
	- A candidate is valid only when it is not completed.
	- Prefix is frozen up to current_step-1 (mapped to next_stop_index-1).
	- Insertion is tested only in the unfrozen suffix (after current_step).
	- Feasibility enforces full-turn capacity and max shift duration.
	- Winning route receives VND intra-route refinement (M1/M2/M3) on suffix.

	Returns
	-------
	VehicleState | None
		Updated winner vehicle state, or None if no feasible insertion exists.
	"""
	print(
		"[Stage 3][INFO] Global cross-depot repair triggered: "
		f"target_node={target_node}, blocked_vehicle={blocked_vehicle_id}."
	)

	if blocked_vehicle_id not in vehicle_states:
		print(f"[Stage 3][DEBUG] Aborted: blocked vehicle {blocked_vehicle_id} was not found.")
		return None

	customer_catalog = _collect_customers(vehicle_states)
	target_customer = customer_catalog.get(target_node)
	if target_customer is None:
		print(
			"[Stage 3][DEBUG] Aborted: "
			f"target_node={target_node} not found in active vehicle routes."
		)
		return None

	best: _BestInsertion | None = None
	scanned_vehicle_count = 0

	# Global vehicle scan: evaluate cheapest feasible insertion for each valid candidate.
	for vehicle_id, state in vehicle_states.items():
		if vehicle_id == blocked_vehicle_id:
			print(f"[Stage 3][DEBUG] Vehicle {vehicle_id} rejected: blocked vehicle.")
			continue
		if _is_completed_state(state):
			print(f"[Stage 3][DEBUG] Vehicle {vehicle_id} rejected: completed state.")
			continue

		scanned_vehicle_count += 1

		if target_node in state.visited_customer_ids:
			print(
				f"[Stage 3][DEBUG] Vehicle {vehicle_id} rejected: "
				"target_node already visited."
			)
			continue
		if any(customer.index == target_node for customer in state.route.customers):
			print(
				f"[Stage 3][DEBUG] Vehicle {vehicle_id} rejected: "
				"target_node already present in route."
			)
			continue

		current_step = max(1, state.next_stop_index)
		suffix_start = current_step - 1
		route_customers = list(state.route.customers)
		if suffix_start >= len(route_customers):
			print(
				f"[Stage 3][DEBUG] Vehicle {vehicle_id} rejected: "
				"no unfrozen suffix "
				f"(next_stop_index={state.next_stop_index}, "
				f"route_size={len(route_customers)})."
			)
			continue

		suffix = route_customers[suffix_start:]
		scan_nodes = [customer.index for customer in suffix] + [state.route.depot.index]

		# Cheapest insertion over edges (i, j) in the unfrozen suffix.
		for edge_position in range(len(scan_nodes) - 1):
			node_i = scan_nodes[edge_position]
			node_j = scan_nodes[edge_position + 1]

			delta_cost = (
				_matrix_distance(distance_matrix, node_i, target_node)
				+ _matrix_distance(distance_matrix, target_node, node_j)
				- _matrix_distance(distance_matrix, node_i, node_j)
			)
   
			if delta_cost == float("inf"):
				print(
					f"[Stage 3][DEBUG] Vehicle {vehicle_id} edge ({node_i},{node_j}) "
					"rejected: insertion attempts to cross a blocked edge."
				)
				continue

			tentative_customers, _ = _insert_target_on_suffix(
				customers=route_customers,
				suffix_start=suffix_start,
				edge_position=edge_position,
				target_customer=target_customer,
			)
			tentative_route = _route_with_same_history(state, tentative_customers)

			total_demand = sum(customer.demand for customer in tentative_customers)
			if total_demand > state.route.depot.max_capacity:
				print(
					f"[Stage 3][DEBUG] Vehicle {vehicle_id} edge ({node_i},{node_j}) "
					"rejected: "
					f"capacity {total_demand:.2f} > {state.route.depot.max_capacity:.2f}."
				)
				continue

			max_shift_duration = state.route.depot.max_duration
			if max_shift_duration > 0 and tentative_route.total_duration() > max_shift_duration:
				print(
					f"[Stage 3][DEBUG] Vehicle {vehicle_id} edge ({node_i},{node_j}) "
					"rejected: "
					f"duration {tentative_route.total_duration():.2f} > "
					f"{max_shift_duration:.2f}."
				)
				continue

			if best is None or delta_cost < best.delta_cost:
				best = _BestInsertion(
					vehicle_id=vehicle_id,
					suffix_start=suffix_start,
					edge_position=edge_position,
					delta_cost=delta_cost,
				)

	print(f"[Stage 3][INFO] Scanned {scanned_vehicle_count} candidate vehicles globally.")

	if best is None:
		print(f"[Stage 3][INFO] No feasible insertion found for target_node={target_node}.")
		return None

	winner_state = vehicle_states[best.vehicle_id]
	winner_customers = list(winner_state.route.customers)
	full_customers_after_insert, inserted_suffix = _insert_target_on_suffix(
		customers=winner_customers,
		suffix_start=best.suffix_start,
		edge_position=best.edge_position,
		target_customer=target_customer,
	)

	# Keep current_step customer fixed and optimize only the remaining suffix tail.
	fixed_first = inserted_suffix[0]

	def _dist(node_i: int, node_j: int) -> float:
		return _matrix_distance(distance_matrix, node_i, node_j)

	optimized_tail = local_search_stage1_intra(
		customers=inserted_suffix[1:],
		start_node=fixed_first,
		end_node=winner_state.route.depot,
		dist_fn=_dist,
	)
	optimized_suffix = [fixed_first, *optimized_tail]

	updated_winner_customers = [
		*full_customers_after_insert[: best.suffix_start],
		*optimized_suffix,
	]
	updated_route = _route_with_same_history(winner_state, updated_winner_customers)
	updated_winner_state = _clone_state_with_route(winner_state, updated_route)

	print(
		"[Stage 3][INFO] Winner selected: "
		f"vehicle={updated_winner_state.route_id}, "
		f"depot={updated_winner_state.route.depot.index}, "
		f"target_node={target_node}, "
		f"delta_c={best.delta_cost:.4f}."
	)
	return updated_winner_state
