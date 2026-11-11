"""Render terminal logs (results/*.log, a run's stdout) as terminal-style PNGs for the README (docs/screenshots)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.makedirs("docs/screenshots", exist_ok=True)


def read(path, head=None, tail=None):
    lines = open(path, encoding="utf-8", errors="replace").read().rstrip("\n").splitlines()
    if head is not None:
        lines = lines[:head]
    if tail is not None:
        lines = lines[-tail:]
    return lines


def render(title, cmd, lines, out, width=150):
    lines = [l if len(l) <= width else l[: width - 1] + "…" for l in lines]
    body = [f"$ {cmd}"] + lines
    h = 0.42 + 0.155 * len(body)
    fig = plt.figure(figsize=(11.5, h))
    fig.patch.set_facecolor("#1e1f22")
    fig.text(0.012, 1 - 0.13 / h, f"●  ●  ●    {title}", color="#9a9a9a", family="monospace", fontsize=8, va="top")
    for i, l in enumerate(body):
        col = "#7fd88f" if i == 0 else ("#ff7b72" if ("FAILED" in l or "Error" in l or "rc=1" in l)
                                        else ("#7fd88f" if ("PASSED" in l or " passed" in l) else "#e6e6e6"))
        fig.text(0.012, 1 - (0.36 + 0.155 * i) / h, l, color=col, family="monospace", fontsize=7.2, va="top")
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", out)


render("pytest", "python -m pytest tests -v", [l for l in read("results/pytest.log") if l.strip()][-22:],
       "docs/screenshots/tests.png")
render("benchmarks", "python scripts/run_benchmarks.py", read("results/benchmarks.log"),
       "docs/screenshots/benchmarks.png")
run = "results/runs/ippo_O3_N100_tau0_psi1_s0"
render("one E1 run", f"python scripts/train.py --algo ippo --obs 3 --N 100 --seed 0  &&  evaluate_run.py --ppo_br",
       read(run + ".stdout", tail=24) + ["", "$ tail results/run_experiments.log"] + read("results/run_experiments.log", tail=4),
       "docs/screenshots/experiments.png")
