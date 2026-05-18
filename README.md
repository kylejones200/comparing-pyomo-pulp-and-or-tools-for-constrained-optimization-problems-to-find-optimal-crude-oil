# Comparing Pyomo PuLP and OR Tools for Constrained Optimization problems to find optimal crude oil

Published: 2025-07-14
Medium: [https://medium.com/@kyle-t-jones/comparing-pyomo-pulp-and-or-tools-for-constrained-optimization-problems-to-find-optimal-crude-oil-385ad1931694](https://medium.com/@kyle-t-jones/comparing-pyomo-pulp-and-or-tools-for-constrained-optimization-problems-to-find-optimal-crude-oil-385ad1931694)

## Business context

IN this scenario, our Objective is to minimize total cost. This could be changed to maximise expected profit (delta between cost and market price). Ot some other objective function like mazimilzing stability over time

Pros: Rich modeling language, good for extensions (nonlinear, MIP) Cons: Slightly verbose, requires external solver (GLPK, CBC)

Pros: Clean syntax, easy to learn, built-in CBC solver Cons: Less powerful for nonlinear or large models

## About

Place the code for this article in this repository.
The original article export is saved as `article.md`.

## Files

Add your `.ipynb`, `.py`, `.yaml`, `.js`, `.ts`, or other project files here.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).