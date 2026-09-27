"""Run a command and report the peak summed RSS of its whole process tree (incl. workers).

Usage: python -m pipeline.memwatch -- <command ...>
"""
import subprocess
import sys
import time


def tree_rss_mb(root: int) -> float:
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    rows = [tuple(int(x) for x in line.split()) for line in out.strip().splitlines() if line.strip()]
    kids: dict[int, list[int]] = {}
    rss = {}
    for pid, ppid, r in rows:
        kids.setdefault(ppid, []).append(pid)
        rss[pid] = r
    total, stack = 0, [root]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack += kids.get(p, [])
    return total / 1024


def main():
    cmd = sys.argv[sys.argv.index("--") + 1:]
    t0 = time.time()
    proc = subprocess.Popen(cmd)
    peak = 0.0
    while proc.poll() is None:
        peak = max(peak, tree_rss_mb(proc.pid))
        time.sleep(0.5)
    print(f"[memwatch] exit {proc.returncode}, {time.time() - t0:.1f}s, peak process-tree RSS {peak:.0f} MB", flush=True)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
