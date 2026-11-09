"""The benchmark runner: the matrix file (`load_matrix`, `select_entries`), a `Target` an application implements, `run_once` / `run_matrix` over a
configuration matrix × a scenario set × k, the `Summary` with pass rate, pass^k and pooled
distributions, kept results, and the shared `cli`. See specs/benchmark.md.

Imported explicitly (``from wica.benchmark import cli``), never re-exported from ``wica`` — like
``wica.testing``, whose scenario files and checks it drives.
"""

from wica.benchmark.cli import cli
from wica.benchmark.matrix import (
    Entry,
    Matrix,
    MatrixError,
    load_matrix,
    select_entries,
)
from wica.benchmark.runner import (
    RunResult,
    Skip,
    Skipped,
    Target,
    TargetFactory,
    run_matrix,
    run_once,
)
from wica.benchmark.summary import (
    Cell,
    Summary,
    aggregate,
    git_commit,
    write_result,
    write_summary,
)

__all__ = [
    "Cell",
    "RunResult",
    "Entry",
    "Matrix",
    "MatrixError",
    "Skip",
    "Skipped",
    "Summary",
    "Target",
    "TargetFactory",
    "aggregate",
    "cli",
    "git_commit",
    "load_matrix",
    "run_matrix",
    "run_once",
    "select_entries",
    "write_result",
    "write_summary",
]
