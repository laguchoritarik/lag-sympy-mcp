# Modifications par rapport au projet d'origine

Ce dépôt est un fork de [sdiehl/sympy-mcp](https://github.com/sdiehl/sympy-mcp) (licence Apache 2.0,
voir `LICENSE`). Le projet d'origine n'est plus maintenu. Modifications apportées par laguchoritarik (2026) :

- `server.py` : 7 nouveaux outils MCP — `factor_expression` (sur Q, Q(i) ou GF(p)), `expand_expression`,
  `cancel_expression`, `apart_expression`, `limit_expression` (bilatérale ou unilatérale, point fini ou ±∞),
  `series_expansion`, `solve_inequality` ; analyseur d'expressions sécurisé sans `eval`
  (`parse_safe_expression`, liste blanche de constructions) ; application des transformations aux deux membres
  d'une équation et à chaque entrée d'une matrice.
- `vars.py` : définitions associées aux nouveaux outils.
- `tests/test_algebra.py` : tests des nouveaux outils.
- `README.md` : documentation des nouveaux outils.
