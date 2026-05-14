---
author: "Kyle Jones"
date_published: "July 14, 2025"
date_exported_from_medium: "November 10, 2025"
canonical_link: "https://medium.com/@kyle-t-jones/comparing-pyomo-pulp-and-or-tools-for-constrained-optimization-problems-to-find-optimal-crude-oil-385ad1931694"
---

# Comparing Pyomo, PuLP, and OR-Tools for Constrained Optimization problems to find optimal crude oil... Assume we have three crude types:

### Comparing Pyomo, PuLP, and OR-Tools for Constrained Optimization problems to find optimal crude oil blends
Assume we have three crude types:


Our task is to blend 6000 barrels such that:

- API ≥ 35
- Sulfur ≤ 1.0%
- No crude exceeds available volume

IN this scenario, our Objective is to minimize total cost. This could be changed to maximise expected profit (delta between cost and market price). Ot some other objective function like mazimilzing stability over time

### ✅ Pyomo
```python
import pyomo.environ as pyo

model = pyo.ConcreteModel()
crudes = ['A', 'B', 'C']
cost = {'A': 70, 'B': 80, 'C': 65}
api = {'A': 34, 'B': 40, 'C': 30}
sulfur = {'A': 1.2, 'B': 0.5, 'C': 2.0}
avail = {'A': 5000, 'B': 3000, 'C': 4000}
model.crudes = pyo.Set(initialize=crudes)
model.vol = pyo.Var(model.crudes, domain=pyo.NonNegativeReals)
model.cost = pyo.Objective(expr=sum(model.vol[c] * cost[c] for c in crudes), sense=pyo.minimize)
model.total_volume = pyo.Constraint(expr=sum(model.vol[c] for c in crudes) == 6000)
model.sulfur = pyo.Constraint(expr=sum(model.vol[c]*sulfur[c] for c in crudes) <= 6000*1.0)
model.api = pyo.Constraint(expr=sum(model.vol[c]*api[c] for c in crudes) >= 6000*35)
model.avail = pyo.ConstraintList()
for c in crudes:
    model.avail.add(model.vol[c] <= avail[c])
solver = pyo.SolverFactory('glpk')
solver.solve(model)
```

Pros: Rich modeling language, good for extensions (nonlinear, MIP)\ Cons: Slightly verbose, requires external solver (GLPK, CBC)

### ✅ PuLP
```python
from pulp import *

model = LpProblem("CrudeBlend", LpMinimize)
vol = LpVariable.dicts("vol", ['A', 'B', 'C'], lowBound=0)
model += lpSum([vol[i] * cost[i] for i in crudes])
model += lpSum([vol[i] for i in crudes]) == 6000
model += lpSum([vol[i] * sulfur[i] for i in crudes]) <= 6000 * 1.0
model += lpSum([vol[i] * api[i] for i in crudes]) >= 6000 * 35
for i in crudes:
    model += vol[i] <= avail[i]
model.solve()
```

Pros: Clean syntax, easy to learn, built-in CBC solver\ Cons: Less powerful for nonlinear or large models

### ✅ OR-Tools
```python
from ortools.linear_solver import pywraplp

solver = pywraplp.Solver.CreateSolver('GLOP')
vol = {i: solver.NumVar(0, avail[i], i) for i in crudes}
solver.Add(solver.Sum([vol[i] for i in crudes]) == 6000)
solver.Add(solver.Sum([vol[i]*sulfur[i] for i in crudes]) <= 6000 * 1.0)
solver.Add(solver.Sum([vol[i]*api[i] for i in crudes]) >= 6000 * 35)
solver.Minimize(solver.Sum([vol[i] * cost[i] for i in crudes]))
status = solver.Solve()
```

Pros: Very fast, production-ready for Google-scale systems\ Cons: Less flexible syntax, weaker support for nonlinear or symbolic models

#### 🧠 Recommendation


- Use PuLP for small, readable LP problems.
- Use Pyomo when modeling more complex, multistage, or nonlinear systems.
- Use OR-Tools when you need speed, performance, or route planning.

This is a side-by-side test. The problem is simple, but it is interesting how much variance there was in runtime for this simple problem.

```python
import pyomo.environ as pyo
from pulp import *
from ortools.linear_solver import pywraplp
import time
import pandas as pd

# Shared data
crudes = ['A', 'B', 'C']
cost = {'A': 70, 'B': 80, 'C': 65}
api = {'A': 34, 'B': 40, 'C': 30}
sulfur = {'A': 1.2, 'B': 0.5, 'C': 2.0}
avail = {'A': 5000, 'B': 3000, 'C': 4000}
target_vol = 6000
sulfur_max = 1.0
api_min = 35

results = {}

# Pyomo
start = time.time()
model_pyo = pyo.ConcreteModel()
model_pyo.crudes = pyo.Set(initialize=crudes)
model_pyo.vol = pyo.Var(model_pyo.crudes, domain=pyo.NonNegativeReals)
model_pyo.cost = pyo.Objective(expr=sum(model_pyo.vol[c] * cost[c] for c in crudes), sense=pyo.minimize)
model_pyo.total_volume = pyo.Constraint(expr=sum(model_pyo.vol[c] for c in crudes) == target_vol)
model_pyo.sulfur = pyo.Constraint(expr=sum(model_pyo.vol[c]*sulfur[c] for c in crudes) <= target_vol*sulfur_max)
model_pyo.api = pyo.Constraint(expr=sum(model_pyo.vol[c]*api[c] for c in crudes) >= target_vol*api_min)
model_pyo.avail = pyo.ConstraintList()
for c in crudes:
    model_pyo.avail.add(model_pyo.vol[c] <= avail[c])
solver = pyo.SolverFactory('glpk')
solver.solve(model_pyo)
end = time.time()
results['Pyomo'] = {
    'runtime_sec': round(end - start, 6),
    'volumes': {c: round(pyo.value(model_pyo.vol[c]), 2) for c in crudes},
    'cost': round(sum(pyo.value(model_pyo.vol[c]) * cost[c] for c in crudes), 2)
}

# PuLP
start = time.time()
model_pulp = LpProblem("CrudeBlend", LpMinimize)
vol_pulp = LpVariable.dicts("vol", crudes, lowBound=0)
model_pulp += lpSum([vol_pulp[i] * cost[i] for i in crudes])
model_pulp += lpSum([vol_pulp[i] for i in crudes]) == target_vol
model_pulp += lpSum([vol_pulp[i] * sulfur[i] for i in crudes]) <= target_vol * sulfur_max
model_pulp += lpSum([vol_pulp[i] * api[i] for i in crudes]) >= target_vol * api_min
for i in crudes:
    model_pulp += vol_pulp[i] <= avail[i]
model_pulp.solve()
end = time.time()
results['PuLP'] = {
    'runtime_sec': round(end - start, 6),
    'volumes': {c: round(vol_pulp[c].varValue, 2) for c in crudes},
    'cost': round(value(model_pulp.objective), 2)
}

# OR-Tools
start = time.time()
solver = pywraplp.Solver.CreateSolver('GLOP')
vol_ort = {i: solver.NumVar(0, avail[i], i) for i in crudes}
solver.Add(solver.Sum([vol_ort[i] for i in crudes]) == target_vol)
solver.Add(solver.Sum([vol_ort[i]*sulfur[i] for i in crudes]) <= target_vol * sulfur_max)
solver.Add(solver.Sum([vol_ort[i]*api[i] for i in crudes]) >= target_vol * api_min)
solver.Minimize(solver.Sum([vol_ort[i] * cost[i] for i in crudes]))
status = solver.Solve()
end = time.time()
results['OR-Tools'] = {
    'runtime_sec': round(end - start, 6),
    'volumes': {c: round(vol_ort[c].solution_value(), 2) for c in crudes},
    'cost': round(sum(vol_ort[c].solution_value() * cost[c] for c in crudes), 2)
}

df = pd.DataFrame(results).T
```
