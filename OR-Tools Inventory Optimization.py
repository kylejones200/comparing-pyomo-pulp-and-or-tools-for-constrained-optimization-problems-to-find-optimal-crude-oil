"""Generated from Jupyter notebook: OR-Tools: Inventory Optimization Example

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

# Install OR-Tools if needed
# !pip install ortools  # Jupyter-only


# --- code cell ---

from ortools.linear_solver import pywraplp


def main():
    # Parameters
    demand = [20, 30, 40, 35, 25]  # Forecasted daily demand
    holding_cost = 2  # Cost per unit of inventory held
    order_cost = 50  # Fixed order cost
    capacity = 100  # Maximum inventory capacity
    order_quantity = 100  # Units per order


    # --- code cell ---

    # Create solver
    solver = pywraplp.Solver.CreateSolver("SCIP")

    # Decision Variables
    # order[t] = 1 if we place an order on day t, 0 otherwise
    order = [solver.BoolVar(f"order_{t}") for t in range(len(demand))]

    # inventory[t] = inventory level at end of day t
    inventory = [solver.IntVar(0, capacity, f"inventory_{t}") for t in range(len(demand))]


    # --- code cell ---

    # Constraints
    # Starting inventory is zero
    solver.Add(inventory[0] == 0)

    # Inventory balance equation for each day
    for t in range(1, len(demand)):
        solver.Add(inventory[t] == inventory[t - 1] + order_quantity * order[t] - demand[t])


    # --- code cell ---

    # Objective: Minimize total cost (ordering costs + holding costs)
    total_cost = sum(
        order_cost * order[t] + holding_cost * inventory[t] for t in range(len(demand))
    )
    solver.Minimize(total_cost)


    # --- code cell ---

    # Solve
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print("✓ Optimal solution found!\n")
        print("Day | Order | Inventory | Demand")
        print("-" * 40)

        for t in range(len(demand)):
            order_val = int(order[t].solution_value())
            inv_val = int(inventory[t].solution_value())
            print(f" {t + 1:2d} |   {order_val}   |    {inv_val:3d}    |   {demand[t]:2d}")

        print(f"\nTotal Cost: ${solver.Objective().Value():.2f}")
    else:
        print("❌ No optimal solution found.")


if __name__ == "__main__":
    main()
