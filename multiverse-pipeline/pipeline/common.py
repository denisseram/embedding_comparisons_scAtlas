"""Shared helpers: config loading, paths, logging, timing, memory reporting."""
from __future__ import annotations

import json
import logging
import os
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or os.environ.get("MV_CONFIG", ROOT / "config.yaml"))
    if not path.is_absolute():
        path = ROOT / path
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path)
    return cfg


def p(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'] relative to the pipeline root."""
    path = Path(cfg["paths"][key])
    return path if path.is_absolute() else ROOT / path


def out_dir(cfg: dict, *parts: str) -> Path:
    d = p(cfg, "outputs").joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3  # bytes on macOS, KiB on Linux


@contextmanager
def timed(logger: logging.Logger, label: str, store: dict | None = None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    logger.info(f"{label}: {dt:.1f}s (peak RSS {peak_rss_mb():.0f} MB)")
    if store is not None:
        store[label] = round(dt, 2)


def append_runtime(cfg: dict, step: str, seconds: float) -> None:
    f = out_dir(cfg) / "runtimes.json"
    data = json.loads(f.read_text()) if f.exists() else {}
    data[step] = {"seconds": round(seconds, 2), "peak_rss_mb": round(peak_rss_mb(), 1)}
    f.write_text(json.dumps(data, indent=2))


def log_decision(cfg: dict, text: str) -> None:
    """Record a runtime modelling choice / fallback so the README can cite it."""
    f = out_dir(cfg) / "decisions_log.txt"
    with open(f, "a") as fh:
        fh.write(text.rstrip() + "\n")
    get_logger("decision").info(text)
