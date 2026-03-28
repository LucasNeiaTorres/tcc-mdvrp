import os
from src.utils.data_loader import read_cordeau_data_file, read_cordeau_solution_file


def test_read_cordeau_data_file():
    base = os.path.join(os.path.dirname(__file__), "..")
    data_file = os.path.abspath(os.path.join(base, "data", "raw", "cordeau", "p01"))
    if not os.path.exists(data_file):
        data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cordeau", "p01"))

    inst = read_cordeau_data_file(data_file)
    assert inst.problem_type == 2
    assert inst.vehicle_count == 4
    assert inst.customer_count == 50
    assert len(inst.duration_limits) == 4
    assert len(inst.capacity_limits) == 4
    assert len(inst.customers) == 50
    assert len(inst.depots) == 4
    # Check one known customer data
    first_customer = inst.customers[0]
    assert first_customer.index == 1
    assert first_customer.x == 37
    assert first_customer.y == 52
    assert first_customer.demand == 7
    assert first_customer.frequency == 1


def test_read_cordeau_solution_file():
    base = os.path.join(os.path.dirname(__file__), "..")
    sol_file = os.path.abspath(os.path.join(base, "data", "raw", "cordeau_sol", "p01.res"))
    if not os.path.exists(sol_file):
        sol_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cordeau_sol", "p01.res"))

    sol = read_cordeau_solution_file(sol_file)
    assert sol.objective > 0
    assert len(sol.routes) > 0
    route0 = sol.routes[0]
    assert route0.depot == 1
    assert route0.vehicle == 1
    assert route0.duration > 0
    assert route0.load > 0
    assert all(isinstance(n, int) for n in route0.nodes)
