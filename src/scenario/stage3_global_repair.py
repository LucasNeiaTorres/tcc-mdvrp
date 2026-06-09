"""Stage-3 global cross-depot repair utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from algorithms.ga_local_search import local_search_stage1_intra
from core.entities import Customer, Route
from scenario.state import VehicleState


EPS = 1e-9


@dataclass(frozen=True)
class _InsertionEvaluation:
	"""Single insertion evaluation record used by Stage 3."""

	vehicle_id: int
	suffix_start: int
	edge_position: int
	delta_cost: float
	distance_cost: float
	fitness_cost: float
	overcapacity: float
	overtime: float


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


def _evaluate_insertion(
	*,
	state: VehicleState,
	route_customers: list[Customer],
	suffix_start: int,
	edge_position: int,
	target_node: int,
	target_customer: Customer,
	distance_matrix: Sequence[Sequence[float]],
	penalty_overcapacity_per_unit: float,
	penalty_overtime_per_minute: float,
) -> _InsertionEvaluation | None:
	"""Evaluate one candidate insertion (hard + soft metrics) in a single pass."""
	suffix = route_customers[suffix_start:]
	scan_nodes = [customer.index for customer in suffix] + [state.route.depot.index]
	node_i = scan_nodes[edge_position]
	node_j = scan_nodes[edge_position + 1]

	delta_cost = (
		_matrix_distance(distance_matrix, node_i, target_node)
		+ _matrix_distance(distance_matrix, target_node, node_j)
		- _matrix_distance(distance_matrix, node_i, node_j)
	)

	# Hard rule: any blocked-edge traversal is rejected immediately.
	if delta_cost == float("inf"):
		return None

	tentative_customers, _ = _insert_target_on_suffix(
		customers=route_customers,
		suffix_start=suffix_start,
		edge_position=edge_position,
		target_customer=target_customer,
	)
	tentative_route = _route_with_same_history(state, tentative_customers)

	overcapacity = tentative_route.capacity_excess()
	overtime = tentative_route.overtime_excess()
	distance_cost = tentative_route.total_distance()
	fitness_cost = tentative_route.fitness_cost(
		penalty_overcapacity_per_unit=penalty_overcapacity_per_unit,
		penalty_overtime_per_minute=penalty_overtime_per_minute,
	)
	return _InsertionEvaluation(
		vehicle_id=state.route_id,
		suffix_start=suffix_start,
		edge_position=edge_position,
		delta_cost=delta_cost,
		distance_cost=distance_cost,
		fitness_cost=fitness_cost,
		overcapacity=overcapacity,
		overtime=overtime,
	)


def _is_hard_feasible(evaluation: _InsertionEvaluation) -> bool:
	"""Layer-1 hard validation (strict legal insertion)."""
	return evaluation.overcapacity <= 0.0 and evaluation.overtime <= 0.0


def stage3_global_cross_depot_repair(
	target_node: int,
	vehicle_states: dict[int, VehicleState],
	distance_matrix: Sequence[Sequence[float]],
	blocked_vehicle_id: int,
	penalty_overcapacity_per_unit: float,
	penalty_overtime_per_minute: float,
	diagnostics_out: dict[str, int] | None = None,
) -> VehicleState | None:
	"""
	Stage 3 fallback with single-pass two-layer validation.

	Layer 1 prioritizes fully legal (hard-feasible) insertions.
	Layer 2 keeps the best soft-feasible insertion by penalized fitness,
	used only when no legal insertion exists.
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

	best_feasible: _InsertionEvaluation | None = None
	best_infeasible: _InsertionEvaluation | None = None
	scanned_vehicle_count = 0
	eligible_vehicle_count = 0
	vehicles_with_hard_feasible_insertion_count = 0
	routes_with_open_insertion_count = 0
	routes_all_insertions_blocked_count = 0

	# Single-pass scan over all candidate vehicles and all insertion edges.
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

		eligible_vehicle_count += 1
		suffix = route_customers[suffix_start:]
		edge_count = len(suffix)  # suffix nodes + depot yields exactly len(suffix) edges.
		route_has_open_insertion = False
		vehicle_has_hard_feasible_insertion = False
		for edge_position in range(edge_count):
			evaluation = _evaluate_insertion(
				state=state,
				route_customers=route_customers,
				suffix_start=suffix_start,
				edge_position=edge_position,
				target_node=target_node,
				target_customer=target_customer,
				distance_matrix=distance_matrix,
				penalty_overcapacity_per_unit=penalty_overcapacity_per_unit,
				penalty_overtime_per_minute=penalty_overtime_per_minute,
			)
			if evaluation is None:
				continue

			route_has_open_insertion = True
			# In this scenario, hard constraints are only blocked-edge traversals.
			# Any non-None evaluation has already passed that hard filter.
			vehicle_has_hard_feasible_insertion = True

			if _is_hard_feasible(evaluation):
				if (
					best_feasible is None
					or evaluation.delta_cost < best_feasible.delta_cost - EPS
					or (
						abs(evaluation.delta_cost - best_feasible.delta_cost) <= EPS
						and evaluation.distance_cost < best_feasible.distance_cost - EPS
					)
				):
					best_feasible = evaluation
				continue

			if (
				best_infeasible is None
				or evaluation.fitness_cost < best_infeasible.fitness_cost - EPS
				or (
					abs(evaluation.fitness_cost - best_infeasible.fitness_cost) <= EPS
					and evaluation.delta_cost < best_infeasible.delta_cost - EPS
				)
			):
				best_infeasible = evaluation

		if vehicle_has_hard_feasible_insertion:
			vehicles_with_hard_feasible_insertion_count += 1

		if route_has_open_insertion:
			routes_with_open_insertion_count += 1
		else:
			routes_all_insertions_blocked_count += 1

	print(f"[Stage 3][INFO] Scanned {scanned_vehicle_count} candidate vehicles globally.")
	print(
		"[Stage 3][INFO] Vehicles with at least 1 blocked-edge-safe insertion "
		"(hard constraints): "
		f"{vehicles_with_hard_feasible_insertion_count}."
	)
	if diagnostics_out is not None:
		diagnostics_out["active_other_routes_count"] = scanned_vehicle_count
		diagnostics_out["eligible_other_routes_count"] = eligible_vehicle_count
		diagnostics_out[
			"vehicles_with_hard_feasible_insertion_count"
		] = vehicles_with_hard_feasible_insertion_count
		diagnostics_out[
			"vehicles_with_blocked_edge_safe_insertion_count"
		] = vehicles_with_hard_feasible_insertion_count
		# Backward-compatible alias for previous key naming.
		diagnostics_out[
			"routes_with_hard_feasible_insertion_count"
		] = vehicles_with_hard_feasible_insertion_count
		diagnostics_out["routes_with_open_insertion_count"] = routes_with_open_insertion_count
		diagnostics_out["routes_all_insertions_blocked_count"] = routes_all_insertions_blocked_count

	# Two-layer decision gate: feasible first, then best penalized infeasible.
	selected = best_feasible
	if selected is not None:
		print(
			"[Stage 3][INFO] Winner selected from hard-feasible set: "
			f"vehicle={selected.vehicle_id}, target_node={target_node}, "
			f"distance_cost={selected.distance_cost:.4f}, delta_c={selected.delta_cost:.4f}."
		)
	else:
		selected = best_infeasible
		if selected is None:
			print(f"[Stage 3][INFO] No insertion found for target_node={target_node}.")
			return None
		print(
			"[Stage 3][WARN] Soft-constraint contingency activated: "
			f"vehicle={selected.vehicle_id}, target_node={target_node}, "
			f"overcapacity={selected.overcapacity:.2f}, overtime={selected.overtime:.2f}, "
			f"fitness={selected.fitness_cost:.4f}."
		)

	winner_state = vehicle_states[selected.vehicle_id]
	winner_customers = list(winner_state.route.customers)
	full_customers_after_insert, inserted_suffix = _insert_target_on_suffix(
		customers=winner_customers,
		suffix_start=selected.suffix_start,
		edge_position=selected.edge_position,
		target_customer=target_customer,
	)

	# Keep current-step customer fixed and optimize only the remaining suffix tail.
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
		*full_customers_after_insert[: selected.suffix_start],
		*optimized_suffix,
	]
	updated_route = _route_with_same_history(winner_state, updated_winner_customers)
	updated_winner_state = _clone_state_with_route(winner_state, updated_route)

	print(
		"[Stage 3][INFO] Winner route committed: "
		f"vehicle={updated_winner_state.route_id}, "
		f"depot={updated_winner_state.route.depot.index}, "
		f"target_node={target_node}."
	)
	return updated_winner_state
