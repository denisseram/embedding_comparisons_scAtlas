"""Run the full offline pipeline in order and report per-step runtimes and peak memory.

Usage: python -m pipeline.run_all [--config config.yaml] [--from STEP] [--skip-export]
"""
from __future__ import annotations

import argparse
import json
import time

from pipeline import export, integrate, knn, measures, metrics, preprocess, regions, variants
from pipeline.common import get_logger, load_config, out_dir, peak_rss_mb

log = get_logger("run_all")
STEPS = ["preprocess", "integrate", "knn", "metrics", "measures", "regions", "variants", "export"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--from", dest="start", default="preprocess", choices=STEPS)
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    modes = cfg["measures"].get("modes", ["jaccard"])
    t_all = time.perf_counter()
    timings = {}
    for step in STEPS[STEPS.index(args.start):]:
        if step == "export" and args.skip_export:
            continue
        t0 = time.perf_counter()
        if step in ("measures", "regions", "variants"):
            mod = {"measures": measures, "regions": regions, "variants": variants}[step]
            for m in modes:
                mod.main(cfg, mode=m)
        else:
            {"preprocess": preprocess, "integrate": integrate, "knn": knn, "metrics": metrics,
             "export": export}[step].main(cfg)
        timings[step] = round(time.perf_counter() - t0, 1)
        log.info(f"== {step} finished in {timings[step]}s (peak RSS of main process {peak_rss_mb():.0f} MB)")
    timings["total"] = round(time.perf_counter() - t_all, 1)
    timings["peak_rss_main_mb"] = round(peak_rss_mb())
    (out_dir(cfg) / "run_all_timings.json").write_text(json.dumps(timings, indent=1))
    log.info(f"pipeline total {timings['total']}s: {timings}")


if __name__ == "__main__":
    main()
