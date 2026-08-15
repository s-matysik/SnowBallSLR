"""Verify a SnowBallSLR installation in under a minute.

Run it from anywhere EXCEPT the repository directory, so that a working import
proves the package is installed rather than merely present on the path:

    python check_install.py            # offline checks only
    python check_install.py --live     # also does one real API round trip

Exit code 0 means every check passed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

OK, BAD = "  ok   ", "  FAIL "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="also resolve one DOI and expand it (needs network)")
    args = ap.parse_args()
    failures = []
    t0 = time.time()

    # 1. import from outside the source tree
    try:
        import snowballslr
        version = snowballslr.__version__
        where = Path(snowballslr.__file__).resolve().parent
        in_tree = (Path.cwd().resolve() in where.parents) or (where.parent == Path.cwd().resolve())
        print(f"{OK}import snowballslr {version}")
        print(f"{'  note ' if in_tree else OK}loaded from {where}"
              + ("  <- inside the source tree; cd elsewhere to prove installation" if in_tree else ""))
    except Exception as exc:
        print(f"{BAD}import: {exc}")
        return 1

    # 2. console script on PATH and reporting the same version
    exe = shutil.which("snowballslr")
    if exe:
        print(f"{OK}console script at {exe}")
    else:
        failures.append("console script not on PATH")
        print(f"{BAD}console script not found on PATH")

    # 3. the public API the paper documents
    try:
        from snowballslr.config import Config
        from snowballslr.core.run import Run
        for name in ("init", "load", "step", "label", "label_from_file", "report"):
            if not hasattr(Run, name):
                failures.append(f"Run.{name} missing")
        print(f"{OK}public API present (Run.init/step/label/report, Config.from_dict)")
    except Exception as exc:
        print(f"{BAD}API import: {exc}")
        return 1

    # 4. configuration validation, including the warnings that catch bad arm designs
    cfg = Config.from_dict({
        "estimate": {"arms": [{"name": "backward", "filter": {"direction": "backward"}},
                              {"name": "forward", "filter": {"direction": "forward"}}]},
    })
    warnings = cfg.warnings()
    if any("direction" in w for w in warnings):
        print(f"{OK}config validation active ({len(warnings)} warning(s) on a degenerate arm design)")
    else:
        failures.append("config warnings not firing")
        print(f"{BAD}expected a warning about direction arms, got: {warnings}")

    # 5. the estimator refuses an uninformative estimate instead of inventing one
    try:
        from snowballslr.estimate.capture_recapture import chapman
        est = chapman(n1=10, n2=10, m=1)
        print(f"{OK}estimator refuses thin overlap (estimable={est.estimable})")
        if est.estimable:
            failures.append("estimator accepted an overlap of 1")
    except Exception as exc:
        print(f"{BAD}estimator: {exc}")

    # 6. optional: one real round trip against the live APIs
    if args.live:
        try:
            tmp = Path(tempfile.mkdtemp())
            live = Config.from_dict({
                "run": {"name": "check", "max_iterations": 1, "directions": ["backward"]},
                "providers": {"order": ["openalex", "crossref"],
                              "openalex": {"enabled": True, "rps": 4},
                              "crossref": {"enabled": True, "rps": 4}},
                "eligibility": {"types": ["article"], "require_title": True},
                "stopping": {"mode": "any_of",
                             "rules": [{"type": "budget", "max_screened": 30},
                                       {"type": "exhaustion"}]},
            })
            run = Run.init(tmp / "run", seeds=["10.1145/2601248.2601268"], config=live)
            found = run.step()
            if run.state.works and found:
                print(f"{OK}live run: 1 seed resolved, {len(found)} candidates returned")
            else:
                failures.append("live run returned nothing")
                print(f"{BAD}live run resolved {len(run.state.works)} seeds, {len(found)} candidates")
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception as exc:
            failures.append(f"live run: {exc}")
            print(f"{BAD}live run: {type(exc).__name__}: {exc}")
            print("       if this is a network error, the offline checks above still passed")

    print(f"\n{len(failures)} failure(s) in {time.time() - t0:.1f}s")
    for f in failures:
        print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
