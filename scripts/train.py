"""Train one shared-policy IPPO/MAPPO run and save it to results/runs/<name>/."""
import argparse, os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from econmarl.config import EconConfig
from econmarl.ppo import PPOConfig, Trainer


def run_name(a):
    return f"{a.algo}_O{a.obs}_N{a.N}_tau{a.tau:g}_psi{a.psi:g}_s{a.seed}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="ippo", choices=["ippo", "mappo"])
    ap.add_argument("--obs", type=int, default=3)
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--psi", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = EconConfig(tau=a.tau, psi=a.psi)
    pc = PPOConfig(algo=a.algo, obs_level=a.obs, N=a.N, seed=a.seed, iters=a.iters, lr=a.lr, threads=a.threads)
    path = os.path.join(a.out, run_name(a) + a.tag)
    os.makedirs(path, exist_ok=True)
    logf = open(os.path.join(path, "log.txt"), "w")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    tr = Trainer(cfg, pc, log=log)
    tr.train()
    tr.save(path)
    log("done")
