"""Run the full experiment grid E1-E4 (train + evaluate) with a pool of parallel worker processes.
Resumable: a run whose eval.json exists is skipped.

    python scripts/run_experiments.py --parallel 6            # everything
    python scripts/run_experiments.py --only E1 --parallel 5  # one experiment
"""
import argparse, os, subprocess, sys, time

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def name(algo, obs, N, tau, psi, seed):
    return f"{algo}_O{obs}_N{N}_tau{tau:g}_psi{psi:g}_s{seed}"


def grid():
    jobs = []  # (exp, algo, obs, N, tau, psi, seed, ppo_br)
    for s in range(5):
        for algo in ("ippo", "mappo"):
            jobs.append(("E1", algo, 3, 100, 0.0, 1.0, s, True))
    for N in (10, 30, 300, 1000):
        for s in range(3):
            jobs.append(("E2", "ippo", 3, N, 0.0, 1.0, s, False))
    for obs in (0, 1, 2):
        for s in range(3):
            jobs.append(("E3", "ippo", obs, 100, 0.0, 1.0, s, False))
    for tau in (0.1, 0.2, 0.3):
        for psi in (0.0, 1.0):
            for s in range(3):
                jobs.append(("E4", "ippo", 3, 100, tau, psi, s, False))
    # heavy jobs first so the pool finishes evenly
    jobs.sort(key=lambda j: -j[3])
    return jobs


def cmd(j, out):
    exp, algo, obs, N, tau, psi, s, br = j
    path = os.path.join(out, name(algo, obs, N, tau, psi, s))
    train = [PY, "scripts/train.py", "--algo", algo, "--obs", str(obs), "--N", str(N), "--tau", str(tau),
             "--psi", str(psi), "--seed", str(s), "--out", out, "--threads", "2"]
    ev = [PY, "scripts/evaluate_run.py", path] + (["--ppo_br"] if br else [])
    return path, train, ev


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default="results/runs")
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(a.out, exist_ok=True)
    todo = [j for j in grid() if (a.only is None or j[0] in a.only)]
    todo = [j for j in todo if not os.path.exists(os.path.join(cmd(j, a.out)[0], "eval.json"))]
    print(f"{len(todo)} runs to do", flush=True)
    running = []
    t0 = time.time()
    while todo or running:
        while todo and len(running) < a.parallel:
            j = todo.pop(0)
            path, tr, ev = cmd(j, a.out)
            os.makedirs(path, exist_ok=True)
            logf = open(path + ".stdout", "w")
            need_train = not os.path.exists(os.path.join(path, "model.pt"))
            shell = (" ".join(tr) + " && " if need_train else "") + " ".join(ev)
            p = subprocess.Popen(shell, shell=True, stdout=logf, stderr=subprocess.STDOUT)
            running.append((p, j, logf))
            print(f"[{time.time()-t0:6.0f}s] start {j[0]} {os.path.basename(path)}", flush=True)
        time.sleep(5)
        for item in list(running):
            p, j, logf = item
            if p.poll() is not None:
                logf.close()
                running.remove(item)
                print(f"[{time.time()-t0:6.0f}s] finished {j[0]} {name(*j[1:7])} rc={p.returncode} "
                      f"({len(todo)} queued, {len(running)} running)", flush=True)
    print("all done", flush=True)
