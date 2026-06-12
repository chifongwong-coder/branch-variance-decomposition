"""Re-render e2_crossing_oracle, e3_main, e3_curves figures from cached
results. e3 uses cached JSON (avoid 7.5h re-run); e2 is fast and re-runs end-to-end."""
import json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import matplotlib
matplotlib.use("Agg")

from core.figure_style import apply_paper_style


# JSON load converts integer dict keys to strings. The plotting code in
# make_figures looks up bandwidth via `k=80` (int). Convert the bandwidth keys
# under `metrics_per_t[*][{"uncond","cond_avg","cond_per_c"}]` back to int.
def _restore_int_keys(results):
    # Only the bandwidth-keyed dicts under metrics_per_t[*]["uncond"] and ["cond_avg"]
    # need int keys; metric_over_t in make_figures uses those.
    for run in results["runs"].values():
        for tm in run.get("metrics_per_t", []):
            for kind_key in ("uncond", "cond_avg"):
                if kind_key in tm:
                    tm[kind_key] = {int(k): v for k, v in tm[kind_key].items()}


def main():
    # Pick up the unified paper style (Type 42 fonts, serif, etc.) before any plot
    # script runs. e3_coupling_comparison and e2_crossing_velocity below otherwise
    # use default matplotlib settings, which leaves figures embedding Type 3 fonts.
    apply_paper_style()

    # ---- E3 re-plot via existing make_figures() with cached 10-seed JSON ----
    os.environ.setdefault("E3C_TAG", "phase2_ci")
    import E3.e3_coupling_comparison as e3_mod
    fig_dir = HERE.parents[1] / "figures"
    fig_dir.mkdir(exist_ok=True)

    e3_json = HERE.parents[1] / "results" / "e3_metrics_phase2_10seeds.json"
    if not e3_json.exists():
        e3_json = HERE.parents[1] / "results" / "e3_metrics_phase2_ci.json"
    print(f"E3: loading {e3_json.name}")
    with open(e3_json) as f:
        results = json.load(f)
    _restore_int_keys(results)

    e3_mod.make_figures(results, fig_dir)
    print("E3 figures regenerated.")

    # ---- E2 re-plot: run the full e2 script (fast Monte Carlo) ----
    print("E2: running e2_crossing_velocity.run()")
    import E2.e2_crossing_velocity as e2_mod
    e2_mod.run()
    print("E2 figure regenerated.")


if __name__ == "__main__":
    main()
