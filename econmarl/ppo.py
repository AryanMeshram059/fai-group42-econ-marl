"""C4: shared-policy IPPO and MAPPO for the household Markov game.

All N households share one actor pi_theta(p_hat, h_hat | o_i) (parameter sharing), each household
optimises its own reward (general-sum).
  IPPO : critic V(o_i)                       -- independent/local critic
  MAPPO: critic V(o_i, g(s))                 -- centralised critic with a permutation-invariant summary g
Actions are two independent Beta distributions on (0, 1), mapped to [p_min, p_max] x [h_min, h_max].
"""
import json, math, os, time
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta

from .config import EconConfig
from .env import Economy
from .obs import build_obs, global_summary, OBS_DIMS, GLOBAL_DIM
from .markov import productivity_process


@dataclass
class PPOConfig:
    algo: str = "ippo"          # ippo | mappo
    obs_level: int = 3
    N: int = 100
    samples_per_iter: int = 80_000   # agent-steps per iteration (M = samples / (N * T_ep))
    T_ep: int = 200
    iters: int = 300
    lr: float = 1e-3
    gamma: float = 0.96         # = beta, so the RL objective matches the household's
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 8192
    ent_coef: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 64
    reward_scale: float = 0.04  # = 1 - beta: values in "per-period utility" units
    eval_every: int = 20
    seed: int = 0
    threads: int = 2


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class Actor(nn.Module):
    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.net = mlp(obs_dim, hidden, 4)

    def dist(self, obs):
        z = self.net(obs)
        conc = 1.0 + nn.functional.softplus(z)
        return Beta(conc[..., :2], conc[..., 2:])

    def mean_action(self, obs):
        d = self.dist(obs)
        return d.mean


class Critic(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = mlp(in_dim, hidden, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_policy_fn(actor, obs_level, cfg: EconConfig, deterministic=True, gen=None):
    sig = cfg.sigma_e / math.sqrt(1 - cfg.rho_e ** 2)

    @torch.no_grad()
    def fn(state):
        obs = build_obs(state, obs_level, sig)
        if deterministic:
            act = actor.mean_action(obs)
        else:
            act = actor.dist(obs).sample()
        act = act.double().clamp(1e-6, 1 - 1e-6)
        return act[..., 0], act[..., 1]
    return fn


class Trainer:
    def __init__(self, cfg: EconConfig, pc: PPOConfig, b1_ref=None, log=print):
        torch.manual_seed(pc.seed)
        np.random.seed(pc.seed)
        torch.set_num_threads(pc.threads)
        self.cfg, self.pc, self.log = cfg, pc, log
        self.M = max(1, round(pc.samples_per_iter / (pc.N * pc.T_ep)))
        self.env = Economy(cfg, N=pc.N, M=self.M, T_ep=pc.T_ep, seed=10_000 + pc.seed)
        self.sig = cfg.sigma_e / math.sqrt(1 - cfg.rho_e ** 2)
        od = OBS_DIMS[pc.obs_level]
        self.actor = Actor(od, pc.hidden)
        self.critic = Critic(od + (GLOBAL_DIM if pc.algo == "mappo" else 0), pc.hidden)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=pc.lr, eps=1e-5)
        self.b1_ref = b1_ref
        self.history = []

    def critic_in(self, obs, state):
        if self.pc.algo == "mappo":
            return torch.cat([obs, global_summary(state)], -1)
        return obs

    @torch.no_grad()
    def collect(self):
        pc, env = self.pc, self.env
        state = env.reset()
        T = pc.T_ep
        O, CI, A, LP, R, V = [], [], [], [], [], []
        stats = {"kappa": [], "hours": [], "p": [], "K_pc": []}
        for t in range(T):
            obs = build_obs(state, pc.obs_level, self.sig)
            cin = self.critic_in(obs, state)
            d = self.actor.dist(obs)
            act = d.sample().clamp(1e-6, 1 - 1e-6)
            lp = d.log_prob(act).sum(-1)
            v = self.critic(cin)
            rew, done, info = env.step(act[..., 0].double(), act[..., 1].double())
            O.append(obs); CI.append(cin); A.append(act); LP.append(lp); V.append(v)
            R.append(rew.float() * pc.reward_scale)
            if t >= T // 2:
                stats["kappa"].append(info["kappa"].mean().item()); stats["hours"].append(info["hours"].mean().item())
                stats["p"].append(info["p"].mean().item()); stats["K_pc"].append(info["K"].mean().item() / pc.N)
            state = env.state()
        # bootstrap the truncated episode with V(s_T)
        obs = build_obs(state, pc.obs_level, self.sig)
        v_last = self.critic(self.critic_in(obs, state))
        V.append(v_last)
        R = torch.stack(R); V = torch.stack(V)
        adv = torch.zeros_like(R)
        gae = torch.zeros_like(R[0])
        for t in reversed(range(T)):
            delta = R[t] + pc.gamma * V[t + 1] - V[t]
            gae = delta + pc.gamma * pc.lam * gae
            adv[t] = gae
        ret = adv + V[:-1]
        flat = lambda x: x.reshape(-1, *x.shape[3:]) if x.dim() > 3 else x.reshape(-1)
        batch = dict(obs=flat(torch.stack(O)), cin=flat(torch.stack(CI)), act=flat(torch.stack(A)),
                     lp=flat(torch.stack(LP)), adv=flat(adv), ret=flat(ret))
        ep_ret = (R.sum(0) / pc.reward_scale / T).mean().item()  # mean per-period utility
        return batch, dict(mean_u=ep_ret, **{k: float(np.mean(v)) for k, v in stats.items()})

    def update(self, batch, frac):
        pc = self.pc
        for g in self.opt.param_groups:
            g["lr"] = pc.lr * (1.0 - frac)
        n = batch["adv"].shape[0]
        adv = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
        info = {"pl": 0.0, "vl": 0.0, "ent": 0.0, "kl": 0.0}
        cnt = 0
        for _ in range(pc.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, pc.minibatch):
                idx = perm[s:s + pc.minibatch]
                d = self.actor.dist(batch["obs"][idx])
                lp = d.log_prob(batch["act"][idx]).sum(-1)
                ratio = torch.exp(lp - batch["lp"][idx])
                a = adv[idx]
                pl = -torch.min(ratio * a, ratio.clamp(1 - pc.clip, 1 + pc.clip) * a).mean()
                v = self.critic(batch["cin"][idx])
                vl = ((v - batch["ret"][idx]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pl + pc.vf_coef * vl - pc.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), pc.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    info["kl"] += (batch["lp"][idx] - lp).mean().item()
                info["pl"] += pl.item(); info["vl"] += vl.item(); info["ent"] += ent.item(); cnt += 1
        return {k: v / cnt for k, v in info.items()}

    def policy_fn(self, deterministic=True):
        return make_policy_fn(self.actor, self.pc.obs_level, self.cfg, deterministic)

    def evaluate(self, T=300, burn_in=200, M=4):
        from .evaluate import rollout
        o = rollout(self.cfg, self.pc.N, self.policy_fn(True), M=M, T=T, burn_in=burn_in, seed=777)
        return o["stats"], o["welfare"]

    def train(self):
        pc = self.pc
        t0 = time.time()
        for it in range(pc.iters):
            batch, roll = self.collect()
            upd = self.update(batch, it / pc.iters)
            rec = dict(iter=it + 1, samples=(it + 1) * self.M * pc.N * pc.T_ep, time=time.time() - t0, **roll, **upd)
            if (it + 1) % pc.eval_every == 0 or it == 0 or it + 1 == pc.iters:
                st, wf = self.evaluate()
                rec.update({"eval_" + k: st[k] for k in ("kappa", "hours", "save_rate", "mean_p", "gini", "K_pc")})
                rec["eval_J"] = wf["J"]
                self.log(f"[{pc.algo} O{pc.obs_level} N={pc.N} s{pc.seed}] it {it+1:4d} t={rec['time']:.0f}s "
                         f"u={roll['mean_u']:.3f} kappa={st['kappa']:.3f} hours={st['hours']:.3f} "
                         f"p={st['mean_p']:.3f} gini={st['gini']:.3f} J={wf['J']:.2f} ent={upd['ent']:.2f}")
            self.history.append(rec)
        return self.history

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, os.path.join(path, "model.pt"))
        json.dump(dict(econ=self.cfg.to_dict(), ppo=asdict(self.pc), M=self.M), open(os.path.join(path, "config.json"), "w"), indent=1)
        json.dump(self.history, open(os.path.join(path, "history.json"), "w"))


def load_actor(path):
    cfgd = json.load(open(os.path.join(path, "config.json")))
    cfg = EconConfig(**cfgd["econ"])
    pc = PPOConfig(**cfgd["ppo"])
    actor = Actor(OBS_DIMS[pc.obs_level], pc.hidden)
    sd = torch.load(os.path.join(path, "model.pt"))
    actor.load_state_dict(sd["actor"])
    critic = Critic(OBS_DIMS[pc.obs_level] + (GLOBAL_DIM if pc.algo == "mappo" else 0), pc.hidden)
    critic.load_state_dict(sd["critic"])
    return cfg, pc, actor, critic
