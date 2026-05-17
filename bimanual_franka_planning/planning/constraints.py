"""User-defined manifold constraints, CasADi-backed.

Users write the constraint equation as a CasADi symbolic expression
in their own script.  The wrapper handles symbolic Jacobian via
autodiff, C codegen, compilation, caching, and hand-off to the
native C++ ``CompiledConstraint`` adapter.

Two input shapes are supported:

* ``param_sym is None`` — the residual is a function of ``q`` only,
  with all other quantities baked into the expression as ``ca.DM``
  literals.  This is the simplest form and was the original API.

* ``param_sym`` is a ``ca.SX`` vector — the residual additionally
  takes a runtime parameter vector ``p`` whose dimension matches
  ``param_sym.numel()``.  The compiled ``.so`` is then reused across
  every call that supplies a different ``p`` via :meth:`set_params`,
  which avoids the codegen + g++ cycle whenever you'd otherwise need
  to rebuild the constraint just to change a few numeric constants
  (start pose, line direction, …).

No prebuilt constraint primitives are shipped.  Every constraint is
defined inline by the caller as a function of the planner's active
joint vector — typically built from :class:`SymbolicContext` (defined
in :mod:`bimanual_franka_planning.planning.symbolic`).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca
import numpy as np

from .symbolic import SymbolicContext, _cwd


def _cache_root() -> Path:
    """Return the constraint cache directory.

    Honours ``AUTOLIFE_CONSTRAINT_CACHE_DIR`` if set (useful for CI).
    Otherwise falls back to ``~/.cache/bimanual_franka_planning/constraints``.
    """
    override = os.environ.get("AUTOLIFE_CONSTRAINT_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return (base / "bimanual_franka_planning" / "constraints").resolve()


@dataclass
class Constraint:
    """A user-defined holonomic constraint, JIT-compiled via CasADi.

    Constructing this class triggers (on cold cache):
        1. symbolic Jacobian via ``ca.jacobian(residual, q_sym)``
        2. C code generation via CasADi
        3. compilation to a ``.so`` with ``c++ -O3 -shared -fPIC``
        4. caching under ``~/.cache/bimanual_franka_planning/constraints/<sha>/``

    On a cache hit the whole thing is a single ``stat`` + string compare.

    Pass ``param_sym`` (a ``ca.SX`` vector) plus optional initial
    ``params`` numeric values to obtain a parameterized constraint
    whose ``.so`` does not need to be recompiled when only the
    parameter values change.
    """

    residual: ca.SX
    q_sym: ca.SX
    name: str = "constraint"
    param_sym: ca.SX | None = None
    params: np.ndarray | None = None

    _so_path: Path = field(init=False)
    _ambient_dim: int = field(init=False)
    _co_dim: int = field(init=False)
    _param_dim: int = field(init=False)
    _symbol_name: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.q_sym, ca.SX):
            raise TypeError("Constraint.q_sym must be a CasADi SX symbol")
        if self.param_sym is not None and not isinstance(self.param_sym, ca.SX):
            raise TypeError("Constraint.param_sym must be a CasADi SX symbol or None")

        res = ca.reshape(self.residual, -1, 1)

        self._ambient_dim = int(self.q_sym.numel())
        self._co_dim = int(res.numel())
        self._param_dim = (
            int(self.param_sym.numel()) if self.param_sym is not None else 0
        )

        # Jacobian is always wrt q only — the planner projects on q,
        # parameters are held fixed during the Newton iteration.
        jac = ca.densify(ca.jacobian(res, self.q_sym))

        if self.param_sym is None:
            f = ca.Function(self.name, [self.q_sym], [res, jac]).expand()
        else:
            f = ca.Function(
                self.name, [self.q_sym, self.param_sym], [res, jac]
            ).expand()

        raw = f.serialize()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        sha = hashlib.sha256(raw).hexdigest()

        cache_dir = _cache_root() / sha[:2] / sha[2:]
        cache_dir.mkdir(parents=True, exist_ok=True)

        c_path = cache_dir / "constraint.c"
        so_path = cache_dir / "constraint.so"

        if not so_path.exists():
            sys.stderr.write(f"[bimanual_franka] compiling constraint {sha[:8]}... ")
            sys.stderr.flush()
            t0 = time.perf_counter()
            with _cwd(cache_dir):
                f.generate("constraint.c")
            compiler = os.environ.get("AUTOLIFE_CONSTRAINT_CC", "c++")
            subprocess.run(
                [
                    compiler,
                    "-O3",
                    "-shared",
                    "-fPIC",
                    str(c_path),
                    "-o",
                    str(so_path),
                ],
                check=True,
            )
            dt = time.perf_counter() - t0
            sys.stderr.write(f"done ({dt * 1000:.0f} ms)\n")
            sys.stderr.flush()

        self._so_path = so_path
        self._symbol_name = self.name

        if self.params is not None:
            self.set_params(self.params)

    def set_params(self, values) -> None:
        """Update the runtime parameter vector.

        Validates shape against ``param_sym`` and stores a contiguous
        ``float64`` copy under :attr:`params` so the motion planner
        can forward it to the C++ adapter on the next ``set_constraints``.
        """
        if self.param_sym is None:
            raise RuntimeError(
                "Constraint.set_params called but this constraint has no param_sym"
            )
        arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64)).reshape(-1)
        if arr.shape[0] != self._param_dim:
            raise ValueError(
                f"Constraint.set_params: expected {self._param_dim} values, "
                f"got shape {np.asarray(values).shape}"
            )
        self.params = arr

    @property
    def so_path(self) -> Path:
        return self._so_path

    @property
    def ambient_dim(self) -> int:
        return self._ambient_dim

    @property
    def co_dim(self) -> int:
        return self._co_dim

    @property
    def param_dim(self) -> int:
        return self._param_dim

    @property
    def symbol_name(self) -> str:
        return self._symbol_name


__all__ = ["Constraint", "SymbolicContext"]
