"""Out-of-tree case generator (dev tooling, not shipped upstream).

Drives the held-out oracle under ``tasks/.../tests/oracle/`` to produce:
- agent-visible bundles at ``tasks/.../environment/data/cases/case_X/``
- held-out ground truth at ``tasks/.../tests/oracle_truth/case_X/``

See ``docs/plan/architecture.md`` and ``docs/plan/oracle-spec.md``.
"""
