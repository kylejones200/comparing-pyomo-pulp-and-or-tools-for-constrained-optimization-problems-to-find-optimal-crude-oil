"""Generated from Jupyter notebook: OR-Tools: Inventory Optimization Example

Magics and shell lines are commented out. Run with a normal Python interpreter."""

from ortools.linear_solver import pywraplp


def main():
    demand = [20, 30, 40, 35, 25]
    holding_cost = 2
    order_cost = 50
    capacity = 100
    order_quantity = 100
    solver = pywraplp.Solver.CreateSolver("SCIP")
    order = [solver.BoolVar(f"order_{t}") for t in range(len(demand))]
    inventory = [
        solver.IntVar(0, capacity, f"inventory_{t}") for t in range(len(demand))
    ]
    solver.Add(inventory[0] == 0)
    for t in range(1, len(demand)):
        solver.Add(
            inventory[t] == inventory[t - 1] + order_quantity * order[t] - demand[t]
        )
    total_cost = sum(
        (
            order_cost * order[t] + holding_cost * inventory[t]
            for t in range(len(demand))
        )
    )
    solver.Minimize(total_cost)
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print("✓ Optimal solution found!\n")
        print("Day | Order | Inventory | Demand")
        print("-" * 40)
        for t in range(len(demand)):
            order_val = int(order[t].solution_value())
            inv_val = int(inventory[t].solution_value())
            print(
                f" {t + 1:2d} |   {order_val}   |    {inv_val:3d}    |   {demand[t]:2d}"
            )
        print(f"\nTotal Cost: ${solver.Objective().Value():.2f}")
    else:
        print("❌ No optimal solution found.")


def main() -> None:
    main()


if __name__ == "__main__":
    main()
