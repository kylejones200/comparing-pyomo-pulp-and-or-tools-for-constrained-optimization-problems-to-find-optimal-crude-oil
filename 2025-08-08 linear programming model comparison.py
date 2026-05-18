"""Generated from Jupyter notebook: 2025-08-08 linear programming model comparison

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import gc
import math
import os
import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil


def _rss_bytes():
    return psutil.Process(os.getpid()).memory_info().rss


def bench(label, fn):
    gc.collect()
    tracemalloc.start()
    rss_before = _rss_bytes()
    t0 = time.perf_counter()
    try:
        obj, orders = fn()
        status, note = ("ok", "")
    except Exception as e:
        obj, orders = (math.nan, pd.DataFrame())
        status, note = ("fail", f"{type(e).__name__}: {e}")
    runtime = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_delta_mb = (_rss_bytes() - rss_before) / 1024**2
    row = {
        "Library": label,
        "Status": status,
        "Runtime (s)": round(runtime, 6),
        "Peak Memory (KB)": int(peak / 1024),
        "RSS Δ (MB)": round(rss_delta_mb, 3),
        "Objective Value": obj,
        "Note": note,
    }
    return (row, orders)


def make_problem(scale=1):
    months = ["M+5", "M+6"]
    tanks = ["Light", "Medium", "Heavy"]
    capacity = (
        pd.Series({"Light": 80000, "Medium": 120000, "Heavy": 100000}, dtype=float)
        * scale
    )
    init_inv = (
        pd.Series({"Light": 10000, "Medium": 20000, "Heavy": 15000}, dtype=float)
        * scale
    )
    demand = pd.DataFrame(
        {
            "tank": np.repeat(tanks, len(months)),
            "month": months * len(tanks),
            "barrels": [50000, 55000, 70000, 65000, 60000, 60000],
        }
    )
    demand["barrels"] = demand["barrels"] * scale
    sources = pd.DataFrame(
        {
            "source": ["Foreign", "Canada", "Domestic"],
            "lead": [3, 2, 1],
            "cost": [62.0, 66.0, 70.0],
            "max_M+5": [120000, 90000, 80000],
            "max_M+6": [120000, 90000, 80000],
        }
    )
    sources[["max_M+5", "max_M+6"]] = sources[["max_M+5", "max_M+6"]] * scale
    mix = pd.DataFrame(
        {
            "source": [
                "Foreign",
                "Foreign",
                "Foreign",
                "Canada",
                "Canada",
                "Canada",
                "Domestic",
                "Domestic",
                "Domestic",
            ],
            "tank": ["Light", "Medium", "Heavy"] * 3,
            "frac": [0.1, 0.3, 0.6, 0.2, 0.5, 0.3, 0.5, 0.35, 0.15],
        }
    )
    dem = (
        demand.pivot(index="tank", columns="month", values="barrels")
        .astype(float)
        .fillna(0.0)
    )
    R = (
        mix.pivot(index="source", columns="tank", values="frac")
        .reindex(sources["source"])
        .astype(float)
        .fillna(0.0)
    )
    costs = sources.set_index("source")["cost"].astype(float)
    smax = (
        sources.set_index("source")[["max_M+5", "max_M+6"]]
        .rename(columns={"max_M+5": "M+5", "max_M+6": "M+6"})
        .astype(float)
    )
    return (months, tanks, capacity, init_inv, dem, R, costs, smax, sources)


def objective_from_orders(orders: pd.DataFrame, costs: pd.Series) -> float:
    return float(
        orders.merge(costs.rename("cost"), on="source").eval("cost*barrels").sum()
    )


def solve_cvxpy(data):
    months, tanks, capacity, init_inv, dem, R, costs, smax, sources = data
    import cvxpy as cp

    S = list(sources["source"])
    M = months
    T = tanks
    S_idx = {s: i for i, s in enumerate(S)}
    M_idx = {m: i for i, m in enumerate(M)}
    Order = cp.Variable((len(S), len(M)), nonneg=True)
    costs_vec = costs.reindex(S).to_numpy()[:, None]
    obj = cp.Minimize(cp.sum(cp.multiply(costs_vec, Order)))
    cons = []
    for t in T:
        r = R.loc[S, t].to_numpy()
        for mo in M:
            mcol = M_idx[mo]
            arrivals = r @ Order[:, mcol]
            endinv = float(init_inv[t]) + arrivals - float(dem.loc[t, mo])
            cons += [endinv >= 0.0, endinv <= float(capacity[t])]
    for s in S:
        i = S_idx[s]
        for mo in M:
            j = M_idx[mo]
            cons += [Order[i, j] <= float(smax.loc[s, mo])]
    try:
        Order_prob = cp.Problem(obj, cons)
        Order_prob.solve(
            solver=cp.ECOS,
            verbose=False,
            feastol=1e-08,
            reltol=1e-08,
            abstol=1e-08,
            max_iters=2000,
        )
    except Exception:
        Order_prob = cp.Problem(obj, cons)
        Order_prob.solve(
            solver=cp.OSQP, verbose=False, eps_abs=1e-08, eps_rel=1e-08, max_iter=100000
        )
    if Order.value is None:
        raise RuntimeError("CVXPY failed to produce a solution")
    rows = []
    for s in S:
        for mo in M:
            rows.append(
                {
                    "source": s,
                    "month": mo,
                    "barrels": float(Order.value[S_idx[s], M_idx[mo]]),
                }
            )
    orders = pd.DataFrame(rows)
    return (objective_from_orders(orders, costs), orders)


def solve_mip(data):
    months, tanks, capacity, init_inv, dem, R, costs, smax, sources = data
    from mip import CBC, Model, minimize, xsum

    mdl = Model(sense=minimize, solver_name=CBC)
    S = list(sources["source"])
    M = months
    Order = {(s, mo): mdl.add_var(lb=0) for s in S for mo in M}
    mdl.objective = xsum((float(costs.loc[s]) * Order[s, mo] for s in S for mo in M))
    for t in tanks:
        for mo in M:
            arrivals = xsum((float(R.loc[s, t]) * Order[s, mo] for s in S))
            mdl += float(init_inv[t]) + arrivals - float(dem.loc[t, mo]) >= 0
            mdl += float(init_inv[t]) + arrivals - float(dem.loc[t, mo]) <= float(
                capacity[t]
            )
    for mo in M:
        for s in S:
            mdl += Order[s, mo] <= float(smax.loc[s, mo])
    mdl.optimize()
    orders = pd.DataFrame(
        [{"source": s, "month": mo, "barrels": Order[s, mo].x} for s in S for mo in M]
    )
    return (objective_from_orders(orders, costs), orders)


def solve_poi_highs(data):
    months, tanks, capacity, init_inv, dem, R, costs, smax, sources = data
    import math

    import pyoptinterface as poi
    from pyoptinterface import highs

    model = highs.Model()
    S = list(sources["source"])
    M = months
    Order = {}
    for s in S:
        for mo in M:
            Order[s, mo] = model.add_variable(
                domain=poi.VariableDomain.Continuous,
                lb=0.0,
                ub=math.inf,
                name=f"Order[{s},{mo}]",
            )
    obj = 0.0
    for s in S:
        for mo in M:
            obj += float(costs.loc[s]) * Order[s, mo]
    model.set_objective(obj, poi.ObjectiveSense.Minimize)
    for t in tanks:
        for mo in M:
            expr = 0.0
            for s in S:
                expr += float(R.loc[s, t]) * Order[s, mo]
            model.add_linear_constraint(
                expr + float(init_inv[t]) - float(dem.loc[t, mo]) >= 0.0
            )
            model.add_linear_constraint(
                expr + float(init_inv[t]) - float(dem.loc[t, mo]) <= float(capacity[t])
            )
    for mo in M:
        for s in S:
            model.add_linear_constraint(Order[s, mo] <= float(smax.loc[s, mo]))
    model.set_model_attribute(poi.ModelAttribute.Silent, True)
    model.optimize()
    rows = [
        {"source": s, "month": mo, "barrels": model.get_value(Order[s, mo])}
        for s in S
        for mo in M
    ]
    orders = pd.DataFrame(rows)
    total = float(
        orders.merge(costs.rename("cost"), on="source").eval("cost*barrels").sum()
    )
    return (total, orders)


def solve_pulp(data):
    months, tanks, capacity, init_inv, dem, R, costs, smax, sources = data
    import pulp

    prob = pulp.LpProblem("supply_lp", pulp.LpMinimize)
    S = list(sources["source"])
    M = months
    Order = pulp.LpVariable.dicts("Order", ((s, mo) for s in S for mo in M), lowBound=0)
    prob += pulp.lpSum((float(costs.loc[s]) * Order[s, mo] for s in S for mo in M))
    for t in tanks:
        for mo in M:
            arrivals = pulp.lpSum((float(R.loc[s, t]) * Order[s, mo] for s in S))
            prob += float(init_inv[t]) + arrivals - float(dem.loc[t, mo]) >= 0
            prob += float(init_inv[t]) + arrivals - float(dem.loc[t, mo]) <= float(
                capacity[t]
            )
    for mo in M:
        for s in S:
            prob += Order[s, mo] <= float(smax.loc[s, mo])
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    orders = pd.DataFrame(
        [
            {"source": s, "month": mo, "barrels": Order[s, mo].value()}
            for s in S
            for mo in M
        ]
    )
    return (objective_from_orders(orders, costs), orders)


def solve_pyoframe(data):
    months, tanks, capacity, init_inv, dem, R, costs, smax, sources = data
    import pyoframe as pf

    m = pf.Model(solver="highs")
    order = {}
    for mo in months:
        v = pf.Variable(sources[["source"]], lb=0.0)
        setattr(m, f"Order_{mo}", v)
        order[mo] = getattr(m, f"Order_{mo}")
    m.minimize = sum((pf.sum(over=["source"], expr=costs * order[mo]) for mo in months))
    for t in tanks:
        coef_s = R[t]
        for mo in months:
            arrivals = pf.sum(over=["source"], expr=coef_s * order[mo])
            endinv = float(init_inv[t]) + arrivals - float(dem.loc[t, mo])
            setattr(m, f"nn_{t}_{mo}", endinv >= 0.0)
            setattr(m, f"cap_{t}_{mo}", endinv <= float(capacity[t]))
    for mo in months:
        setattr(m, f"max_{mo}", order[mo] <= smax[mo])
    m.optimize()
    frames = []
    for mo in months:
        tmp = order[mo].solution.to_pandas().rename(columns={"solution": "barrels"})
        tmp["month"] = mo
        frames.append(tmp)
    orders = pd.concat(frames, ignore_index=True)
    total = float(
        orders.merge(costs.rename("cost"), on="source").eval("cost*barrels").sum()
    )
    return (total, orders)


def solve_pyomo(data):
    months, tanks, capacity, init_inv, dem, R, costs, smax, sources = data
    from pyomo.environ import (
        ConcreteModel,
        Constraint,
        NonNegativeReals,
        Objective,
        Set,
        SolverFactory,
        Var,
        minimize,
        value,
    )

    m = ConcreteModel()
    S = list(sources["source"])
    M = months
    m.S = Set(initialize=S)
    m.M = Set(initialize=M)
    m.T = Set(initialize=tanks)
    cost = {s: float(costs.loc[s]) for s in S}
    cap = {t: float(capacity[t]) for t in tanks}
    inv0 = {t: float(init_inv[t]) for t in tanks}
    demd = {(t, mo): float(dem.loc[t, mo]) for t in tanks for mo in M}
    rout = {(s, t): float(R.loc[s, t]) for s in S for t in tanks}
    smax_dict = {(s, mo): float(smax.loc[s, mo]) for s in S for mo in M}
    m.Order = Var(m.S, m.M, domain=NonNegativeReals)

    def obj_rule(m):
        return sum((cost[s] * m.Order[s, mo] for s in m.S for mo in m.M))

    m.OBJ = Objective(rule=obj_rule, sense=minimize)

    def nn_rule(m, t, mo):
        arr = sum((rout[s, t] * m.Order[s, mo] for s in m.S))
        return inv0[t] + arr - demd[t, mo] >= 0

    def cap_rule(m, t, mo):
        arr = sum((rout[s, t] * m.Order[s, mo] for s in m.S))
        return inv0[t] + arr - demd[t, mo] <= cap[t]

    m.nn = Constraint(m.T, m.M, rule=nn_rule)
    m.cap = Constraint(m.T, m.M, rule=cap_rule)

    def max_rule(m, s, mo):
        return m.Order[s, mo] <= smax_dict[s, mo]

    m.maxc = Constraint(m.S, m.M, rule=max_rule)
    for cand in ["cbc", "highs", "glpk"]:
        opt = SolverFactory(cand)
        if opt.available(False):
            solver = opt
            break
    else:
        raise RuntimeError("No Pyomo solver found (need 'cbc', 'highs', or 'glpk')")
    solver.solve(m, tee=False)
    orders = pd.DataFrame(
        [
            {"source": s, "month": mo, "barrels": float(value(m.Order[s, mo]))}
            for s in S
            for mo in M
        ]
    )
    return (objective_from_orders(orders, costs), orders)


def main() -> None:
    scale = 1
    data = make_problem(scale=scale)
    tests = [
        ("PyOFrame + HiGHS", lambda: solve_pyoframe(data)),
        ("Python-MIP + CBC", lambda: solve_mip(data)),
        ("PyOptInterface + HiGHS", lambda: solve_poi_highs(data)),
        ("PuLP + CBC", lambda: solve_pulp(data)),
        ("Pyomo CBC/HiGHS/GLPK", lambda: solve_pyomo(data)),
        ("CVXPY + ECOS/OSQP", lambda: solve_cvxpy(data)),
    ]
    rows, solutions = ([], {})
    for name, fn in tests:
        try:
            row, orders = bench(name, fn)
        except ImportError as e:
            row, orders = (
                {
                    "Library": name,
                    "Status": "skip",
                    "Runtime (s)": math.nan,
                    "Peak Memory (KB)": math.nan,
                    "RSS Δ (MB)": math.nan,
                    "Objective Value": math.nan,
                    "Note": f"ImportError: {e}",
                },
                pd.DataFrame(),
            )
        rows.append(row)
        solutions[name] = orders

    df = pd.DataFrame(rows)
    df = df[
        [
            "Library",
            "Status",
            "Runtime (s)",
            "Peak Memory (KB)",
            "Objective Value",
            "Note",
        ]
    ]
    df.sort_values(["Status", "Runtime (s)"], inplace=True, na_position="last")
    print("\n=== Benchmark Summary ===")
    print(df.to_string(index=False))
    out_csv = "benchmark_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")
    ok = df[df["Status"] == "ok"]
    if not ok.empty:
        plt.figure(figsize=(8, 4))
        plt.bar(ok["Library"], ok["Runtime (s)"])
        plt.title("Solver Runtime in Seconds (Lower is better)")
        plt.xticks(rotation=45, ha="right")
        ax = plt.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.tight_layout()
        plt.savefig("benchmark_runtime.png")
        plt.show()
        print("Saved benchmark_runtime.png")
        plt.figure(figsize=(8, 4))
        plt.bar(ok["Library"], ok["Peak Memory (KB)"])
        plt.title("Peak Python-Tracked Memory in KB (Lower is better)")
        plt.xticks(rotation=45, ha="right")
        ax = plt.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.tight_layout()
        plt.savefig("benchmark_memory.png")
        plt.show()
        print("Saved benchmark_memory.png")

    first_ok = next((name for name, _ in tests if name in ok["Library"].tolist()), None)
    if first_ok:
        print(f"\nSample solution from {first_ok}:")
        print(solutions[first_ok].to_string(index=False))


if __name__ == "__main__":
    main()
