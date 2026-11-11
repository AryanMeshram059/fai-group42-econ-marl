"""Aggregate every results/runs/*/eval.json + results/benchmarks.json into
  results/summary.json, report/tables/*.tex, report/figures/*.pdf and figures/*.png.
Every number in the report is produced here from saved results."""
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.makedirs("report/tables", exist_ok=True); os.makedirs("report/figures", exist_ok=True); os.makedirs("figures", exist_ok=True)

# palette (validated categorical order from the dataviz reference palette)
C = dict(ippo="#2a78d6", mappo="#eb6834", b1="#0b0b0b", b2="#1baf7a", rand="#e87ba4", psi0="#6250d6", psi1="#eb6834",
         o0="#2a78d6", o1="#eb6834", o2="#1baf7a", o3="#6250d6", grid="#e4e3df", muted="#52514e")
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.4, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
                     "axes.edgecolor": C["muted"], "figure.dpi": 150, "savefig.bbox": "tight", "font.family": "serif"})


def save(fig, name):
    fig.savefig(f"report/figures/{name}.pdf"); fig.savefig(f"figures/{name}.png", dpi=200)
    plt.close(fig)


runs = []
for f in glob.glob("results/runs/*/eval.json"):
    r = json.load(open(f))
    r["history"] = json.load(open(os.path.join(os.path.dirname(f), "history.json")))
    runs.append(r)
bench = json.load(open("results/benchmarks.json"))
print(f"{len(runs)} evaluated runs")


def sel(**kw):
    out = [r for r in runs if all(r[k] == v for k, v in kw.items())]
    return sorted(out, key=lambda r: r["seed"])


def ms(vals):
    v = np.array([x for x in vals if x is not None], dtype=float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v))


def fmt(m, s, d=3):
    return f"{m:.{d}f} $\\pm$ {s:.{d}f}"


summary = {"n_runs": len(runs)}
B = dict(algo="ippo", obs=3, N=100, tau=0.0, psi=1.0)

# =============================================================== E1
e1 = {}
for algo in ("ippo", "mappo"):
    rs = sel(algo=algo, obs=3, N=100, tau=0.0, psi=1.0)
    if not rs:
        continue
    d = {k: ms([r["marl"][k] for r in rs]) for k in ("kappa", "K_pc", "hours", "save_rate", "mean_p", "r", "gini", "top10")}
    d["welfare_loss_pct"] = ms([r["welfare_loss_pct"] for r in rs])
    d["kappa_gap"] = ms([r["gap"]["kappa_rel_theory"] for r in rs])
    d["kappa_gap_signed"] = ms([r["gap"]["kappa_signed_rel_theory"] for r in rs])
    d["p_gap"] = ms([r["gap"]["p_gap"] for r in rs]); d["h_gap"] = ms([r["gap"]["h_gap"] for r in rs])
    d["exploit_dp"] = ms([r["exploit_dp"]["mean_ce_gain_pct"] for r in rs])
    for q in ("q10", "q50", "q90"):
        d["exploit_dp_" + q] = ms([r["exploit_dp"][q]["ce_gain_pct"] for r in rs])
    if all("exploit_ppo" in r for r in rs):
        d["exploit_ppo"] = ms([r["exploit_ppo"]["mean_ce_gain_pct"] for r in rs])
        for q in ("q10", "q50", "q90"):
            d["exploit_ppo_" + q] = ms([r["exploit_ppo"][q]["ce_gain_pct"] for r in rs])
    d["conv_iter"] = ms([r["convergence_iter"] for r in rs])
    d["train_time_min"] = ms([r["train_time_s"] / 60 for r in rs])
    e1[algo] = d
summary["E1"] = e1
t0 = bench["tau0"]
th = bench["configs"][0]["theory"]
summary["B1"] = dict(theory=th, sim100=t0["B1"]["stats"], exploit_dp=t0["B1"]["exploit_dp"]["mean_ce_gain_pct"],
                     B2=t0["B2"]["params"], B2_loss=t0["B2"]["welfare_loss_pct"], B2_exploit=t0["B2"]["exploit_dp"]["mean_ce_gain_pct"],
                     rand_loss=t0["Random"]["welfare_loss_pct"], b2_stats=t0["B2"]["stats"], rand_stats=t0["Random"]["stats"])

rows = []
rows.append(["B1 (theory, $N{=}\\infty$)", f"{th['kappa']:.3f}", f"{th['K']:.3f}", f"{th['hours']:.3f}",
             f"{th['saving_rate']:.3f}", f"{th['gini']:.3f}", f"{th['top10']:.3f}", "--", "--", "--"])
for name, key in (("B1 in simulator", "B1"), ("B2 (best constant)", "B2"), ("Random", "Random")):
    s = t0[key]["stats"]
    ex = t0[key].get("exploit_dp")
    rows.append([name, f"{s['kappa']:.3f}", f"{s['K_pc']:.3f}", f"{s['hours']:.3f}", f"{s['save_rate']:.3f}",
                 f"{s['gini']:.3f}", f"{s['top10']:.3f}", f"{t0[key]['welfare_loss_pct']:.2f}",
                 f"{ex['mean_ce_gain_pct']:.2f}" if ex else "--", "--"])
for algo, lab in (("ippo", "IPPO"), ("mappo", "MAPPO")):
    if algo not in e1:
        continue
    d = e1[algo]
    rows.append([lab + f" ($n{{=}}{d['kappa'][2]}$)"] + [fmt(*d[k][:2], 3) for k in ("kappa", "K_pc", "hours", "save_rate", "gini", "top10")]
                + [fmt(*d["welfare_loss_pct"][:2], 2), fmt(*d["exploit_dp"][:2], 2),
                   fmt(*d["exploit_ppo"][:2], 2) if "exploit_ppo" in d else "--"])
with open("report/tables/e1.tex", "w") as f:
    f.write("\\begin{tabular}{lccccccccc}\n\\toprule\n")
    f.write("Policy & $\\kappa$ & $K/N$ & hours & $a'/m$ & Gini & Top-10\\% & W-loss (\\%) & DP-BR (\\%) & PPO-BR (\\%)\\\\\n\\midrule\n")
    for r in rows:
        f.write(" & ".join(r) + "\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---- Fig 1: learning curves
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.1))
for algo in ("ippo", "mappo"):
    rs = sel(algo=algo, obs=3, N=100, tau=0.0, psi=1.0)
    if not rs:
        continue
    for ax, key in zip(axs, ("eval_kappa", "eval_gini", "eval_J")):
        its = [h["iter"] for h in rs[0]["history"] if key in h]
        Y = np.array([[h[key] for h in r["history"] if key in h] for r in rs])
        x = np.array(its) * rs[0]["samples"] / rs[0]["history"][-1]["iter"] / 1e6
        ax.plot(x, Y.mean(0), color=C[algo], label=algo.upper())
        ax.fill_between(x, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0), color=C[algo], alpha=0.18, lw=0)
# B1 reference under the same evaluation protocol (seed 777, M=4, T=300, burn-in 200)
from econmarl.config import EconConfig
from econmarl.bench_cache import get_equilibrium
from econmarl.baselines import b1_policy_fn
from econmarl.evaluate import rollout
cfg0 = EconConfig(); eq0 = get_equilibrium(cfg0)
ref = rollout(cfg0, 100, b1_policy_fn(eq0.sol), M=4, T=300, burn_in=200, seed=777)
for ax, v, t in zip(axs, (ref["stats"]["kappa"], ref["stats"]["gini"], ref["welfare"]["J"]),
                    ("capital-labour ratio $\\kappa$", "wealth Gini", "welfare $\\bar J$")):
    ax.axhline(v, color=C["b1"], ls="--", lw=1.1, label="B1")
    ax.set_title(t); ax.set_xlabel("agent-steps (M)")
axs[2].set_ylim(ref["welfare"]["J"] - 4, ref["welfare"]["J"] + 0.6)
axs[0].legend(frameon=False, loc="lower right")
fig.tight_layout(); save(fig, "fig_e1_learning")

# ---- Fig 2: policy functions
r0 = sel(algo="ippo", obs=3, N=100, tau=0.0, psi=1.0)
if r0:
    pcur = r0[0]["policy_curves"]
    a = np.array(pcur["a"])
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
    shades = {"1": 0.45, "3": 0.7, "5": 1.0}
    for k, cv in pcur["curves"].items():
        lab = {"1": "low $e$", "3": "median $e$", "5": "high $e$"}[k]
        al = shades[k]
        axs[0].plot(a, cv["p_marl"], color=C["ippo"], alpha=al, label=f"IPPO, {lab}")
        axs[0].plot(a, cv["p_b1"], color=C["b1"], alpha=al, ls="--", label=f"B1, {lab}")
        axs[1].plot(a, cv["h_marl"], color=C["ippo"], alpha=al)
        axs[1].plot(a, cv["h_b1"], color=C["b1"], alpha=al, ls="--")
    axs[0].set_title("saving ratio $p(a,e)$"); axs[1].set_title("hours $h(a,e)$")
    for ax in axs:
        ax.set_xlabel("assets $a$")
    h_, l_ = axs[0].get_legend_handles_labels()
    order = [0, 2, 4, 1, 3, 5]
    fig.legend([h_[i] for i in order], [l_[i] for i in order], frameon=False, ncol=6, fontsize=6.5,
               loc="upper center", bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(); save(fig, "fig_e1_policy")

# ---- Fig 3: wealth distribution (Lorenz + density)


def lorenz(w):
    w = np.sort(np.asarray(w)); c = np.cumsum(w) / w.sum()
    return np.linspace(0, 1, len(w) + 1), np.concatenate([[0], c])


fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
series = [("B1 in simulator", np.concatenate([r["final_wealth_b1"] for r in sel(**B)]) if sel(**B) else t0["B1"]["final_wealth"], C["b1"], "--"),
          ("IPPO", np.concatenate([r["final_wealth"] for r in sel(**B)]) if sel(**B) else None, C["ippo"], "-"),
          ("MAPPO", np.concatenate([r["final_wealth"] for r in sel(algo="mappo", obs=3, N=100, tau=0.0, psi=1.0)]) if sel(algo="mappo") else None, C["mappo"], "-"),
          ("B2", t0["B2"]["final_wealth"], C["b2"], "-.")]
bins = np.linspace(0, np.log1p(80), 45)
for lab, w, col, ls in series:
    if w is None:
        continue
    x, y = lorenz(w)
    axs[0].plot(x, y, color=col, ls=ls, label=lab)
    hist, edges = np.histogram(np.log1p(w), bins=bins, density=True)
    axs[1].plot(0.5 * (edges[1:] + edges[:-1]), hist, color=col, ls=ls, label=lab)
axs[0].plot([0, 1], [0, 1], color=C["muted"], lw=0.6)
axs[0].set_title("Lorenz curve of wealth"); axs[0].set_xlabel("population share"); axs[0].set_ylabel("wealth share")
axs[1].set_title("density of $\\log(1+a)$"); axs[1].set_xlabel("$\\log(1+a)$"); axs[1].set_xlim(0, 3.3)
axs[0].legend(frameon=False, loc="upper left")
fig.tight_layout(); save(fig, "fig_e1_wealth")

# =============================================================== E2
e2 = {}
for N in (10, 30, 100, 300, 1000):
    rs = sel(algo="ippo", obs=3, N=N, tau=0.0, psi=1.0)
    if not rs:
        continue
    e2[N] = dict(kappa_gap=ms([r["gap"]["kappa_rel_theory"] for r in rs]),
                 kappa_gap_signed=ms([r["gap"]["kappa_signed_rel_theory"] for r in rs]),
                 kappa_gap_sim=ms([r["gap"]["kappa_rel_sim"] for r in rs]),
                 p_gap=ms([r["gap"]["p_gap"] for r in rs]), h_gap=ms([r["gap"]["h_gap"] for r in rs]),
                 welfare_loss=ms([r["welfare_loss_pct"] for r in rs]),
                 exploit_dp=ms([r["exploit_dp"]["mean_ce_gain_pct"] for r in rs]),
                 time_min=ms([r["train_time_s"] / 60 for r in rs]), samples=rs[0]["samples"],
                 b1sim_gap=abs(bench["N_scan"][str(N)]["stats"]["kappa"] - th["kappa"]) / th["kappa"])
Ns = np.array(sorted(e2))
if len(Ns) >= 3:
    y = np.array([e2[n]["kappa_gap"][0] for n in Ns])
    X = np.stack([1.0 / Ns, np.ones(len(Ns))], 1)
    coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    X2 = np.stack([1.0 / np.sqrt(Ns), np.ones(len(Ns))], 1)
    coef2, *_ = np.linalg.lstsq(X2, y, rcond=None)
    r2b = 1 - np.sum((y - X2 @ coef2) ** 2) / np.sum((y - y.mean()) ** 2)
    summary["E2_fit"] = dict(c=float(coef[0]), delta0=float(coef[1]), r2=float(r2), c_sqrt=float(coef2[0]),
                             delta0_sqrt=float(coef2[1]), r2_sqrt=float(r2b))
summary["E2"] = {str(k): v for k, v in e2.items()}
with open("report/tables/e2.tex", "w") as f:
    f.write("\\begin{tabular}{rcccccccc}\n\\toprule\n$N$ & $n$ & $|\\Delta\\kappa|/\\kappa^\\star$ & B1-sim gap & $p$-gap & $h$-gap & W-loss (\\%) & DP-BR (\\%) & time (min)\\\\\n\\midrule\n")
    for N in Ns:
        d = e2[N]
        f.write(f"{N} & {d['kappa_gap'][2]} & {fmt(*d['kappa_gap'][:2])} & {d['b1sim_gap']:.3f} & {fmt(*d['p_gap'][:2])} & "
                f"{fmt(*d['h_gap'][:2])} & {fmt(*d['welfare_loss'][:2], 2)} & {fmt(*d['exploit_dp'][:2], 2)} & {d['time_min'][0]:.1f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
if len(Ns):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.1))
    m = np.array([e2[n]["kappa_gap"][0] for n in Ns]); s = np.array([e2[n]["kappa_gap"][1] for n in Ns])
    axs[0].errorbar(Ns, m, yerr=s, color=C["ippo"], marker="o", ms=4, capsize=2, label="IPPO vs B1")
    axs[0].plot(Ns, [e2[n]["b1sim_gap"] for n in Ns], color=C["b1"], ls="--", marker="s", ms=3, label="B1-sim vs B1")
    if "E2_fit" in summary:
        nn = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 50)
        fz = summary["E2_fit"]
        axs[0].plot(nn, fz["c"] / nn + fz["delta0"], color=C["ippo"], lw=0.9, ls=":", label=f"$c/N+\\Delta_0$ ($R^2$={fz['r2']:.2f})")
    axs[0].set_xscale("log"); axs[0].set_title("relative $\\kappa$ gap"); axs[0].set_xlabel("$N$")
    axs[0].legend(frameon=False, fontsize=6)
    for ax, key, t in ((axs[1], "p_gap", "policy gap (saving ratio)"), (axs[2], "exploit_dp", "DP best-response gain (% CE)")):
        mm = np.array([e2[n][key][0] for n in Ns]); ss = np.array([e2[n][key][1] for n in Ns])
        ax.errorbar(Ns, mm, yerr=ss, color=C["ippo"], marker="o", ms=4, capsize=2)
        ax.set_xscale("log"); ax.set_title(t); ax.set_xlabel("$N$")
    axs[2].axhline(0, color=C["muted"], lw=0.8)
    fig.tight_layout(); save(fig, "fig_e2_population")

# =============================================================== E3
e3 = {}
for o in (0, 1, 2, 3):
    rs = sel(algo="ippo", obs=o, N=100, tau=0.0, psi=1.0)
    if not rs:
        continue
    e3[o] = dict(kappa_gap=ms([r["gap"]["kappa_rel_theory"] for r in rs]), p_gap=ms([r["gap"]["p_gap"] for r in rs]),
                 h_gap=ms([r["gap"]["h_gap"] for r in rs]), welfare_loss=ms([r["welfare_loss_pct"] for r in rs]),
                 gini=ms([r["marl"]["gini"] for r in rs]), conv=ms([r["convergence_iter"] for r in rs]),
                 exploit_dp=ms([r["exploit_dp"]["mean_ce_gain_pct"] for r in rs]),
                 curves=[[h["eval_J"] for h in r["history"] if "eval_J" in h] for r in rs],
                 curve_its=[h["iter"] for h in rs[0]["history"] if "eval_J" in h])
summary["E3"] = {str(k): {kk: vv for kk, vv in v.items() if kk not in ("curves", "curve_its")} for k, v in e3.items()}
with open("report/tables/e3.tex", "w") as f:
    f.write("\\begin{tabular}{lccccccc}\n\\toprule\nObs. & $n$ & $|\\Delta\\kappa|/\\kappa^\\star$ & $p$-gap & $h$-gap & W-loss (\\%) & DP-BR (\\%) & conv. iter\\\\\n\\midrule\n")
    for o, d in e3.items():
        f.write(f"O{o} & {d['kappa_gap'][2]} & {fmt(*d['kappa_gap'][:2])} & {fmt(*d['p_gap'][:2])} & {fmt(*d['h_gap'][:2])} & "
                f"{fmt(*d['welfare_loss'][:2], 2)} & {fmt(*d['exploit_dp'][:2], 2)} & {d['conv'][0]:.0f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
if e3:
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.1))
    labs = [f"O{o}" for o in e3]
    for ax, key, t in ((axs[0], "kappa_gap", "relative $\\kappa$ gap"), (axs[1], "welfare_loss", "welfare loss (% CE)")):
        mm = [e3[o][key][0] for o in e3]; ss = [e3[o][key][1] for o in e3]
        ax.bar(labs, mm, yerr=ss, color=[C[f"o{o}"] for o in e3], width=0.6, capsize=2, edgecolor="white", linewidth=1)
        ax.set_title(t); ax.grid(axis="x", visible=False)
    for o, d in e3.items():
        Y = np.array(d["curves"]); x = np.array(d["curve_its"])
        axs[2].plot(x, Y.mean(0), color=C[f"o{o}"], label=f"O{o}")
    axs[2].axhline(ref["welfare"]["J"], color=C["b1"], ls="--", lw=1.0, label="B1")
    axs[2].set_ylim(ref["welfare"]["J"] - 6, ref["welfare"]["J"] + 1)
    axs[2].set_title("welfare $\\bar J$ during training"); axs[2].set_xlabel("PPO iteration"); axs[2].legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout(); save(fig, "fig_e3_information")

# =============================================================== E4
cfgs = {(c["tau"], c["psi"]): c for c in bench["configs"]}
e4 = {}
for psi in (0.0, 1.0):
    for tau in (0.0, 0.1, 0.2, 0.3):
        rs = sel(algo="ippo", obs=3, N=100, tau=tau, psi=1.0 if tau == 0 else psi)
        cb = cfgs[(tau, 1.0 if tau == 0 else psi)]
        if not rs:
            continue
        e4[(tau, psi)] = dict(K=ms([r["marl"]["K_pc"] for r in rs]), L=ms([r["marl"]["hours"] for r in rs]),
                              kappa=ms([r["marl"]["kappa"] for r in rs]), gini=ms([r["marl"]["gini"] for r in rs]),
                              J=ms([r["welfare_marl"]["J"] for r in rs]),
                              K_b1=cb["theory"]["K"], L_b1=cb["theory"]["hours"], kappa_b1=cb["kappa"],
                              gini_b1=cb["sim100"]["gini"], J_b1=cb["welfare_sim100"]["J"],
                              eff_L_b1=cb["theory"]["L"], n=len(rs))
el = {}
for psi in (0.0, 1.0):
    taus = [t for t in (0.0, 0.1, 0.2, 0.3) if (t, psi) in e4]
    if len(taus) < 3:
        continue
    x = np.log(1 - np.array(taus))
    out = {}
    for var in ("K", "L"):
        ym = np.log([e4[(t, psi)][var][0] for t in taus]); yb = np.log([e4[(t, psi)][var + "_b1"] for t in taus])
        out[var + "_marl"] = float(np.polyfit(x, ym, 1)[0]); out[var + "_b1"] = float(np.polyfit(x, yb, 1)[0])
    el[psi] = out
summary["E4"] = {f"tau{k[0]}_psi{k[1]}": v for k, v in e4.items()}
summary["E4_elasticities"] = {str(k): v for k, v in el.items()}
with open("report/tables/e4.tex", "w") as f:
    f.write("\\begin{tabular}{ccccccccc}\n\\toprule\n$\\tau$ & $\\psi$ & $K/N$ MARL & $K/N$ B1 & hours MARL & hours B1 & Gini MARL & Gini B1 & $\\bar J$ MARL / B1\\\\\n\\midrule\n")
    for (tau, psi), d in sorted(e4.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if tau == 0 and psi == 0:
            continue
        f.write(f"{tau:.1f} & {psi:.0f} & {fmt(*d['K'][:2])} & {d['K_b1']:.3f} & {fmt(*d['L'][:2])} & {d['L_b1']:.3f} & "
                f"{d['gini'][0]:.3f} & {d['gini_b1']:.3f} & {d['J'][0]:.1f} / {d['J_b1']:.1f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
if e4:
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for psi in (0.0, 1.0):
        taus = [t for t in (0.0, 0.1, 0.2, 0.3) if (t, psi) in e4]
        col = C["psi0"] if psi == 0 else C["psi1"]
        for ax, var, t in ((axs[0], "K", "capital per household $K/N$"), (axs[1], "L", "mean hours"), (axs[2], "gini", "wealth Gini")):
            mm = [e4[(tt, psi)][var][0] for tt in taus]; ss = [e4[(tt, psi)][var][1] for tt in taus]
            ax.errorbar(taus, mm, yerr=ss, color=col, marker="o", ms=3.5, capsize=2, label=f"IPPO, $\\psi$={psi:.0f}")
            ax.plot(taus, [e4[(tt, psi)][var + "_b1"] for tt in taus], color=col, ls="--", marker="s", ms=3, label=f"B1, $\\psi$={psi:.0f}")
            ax.set_title(t); ax.set_xlabel("income tax $\\tau$")
    axs[0].legend(frameon=False, fontsize=6)
    fig.tight_layout(); save(fig, "fig_e4_tax")

fh = [r["run"] if "run" in r else f"{r['algo']}_O{r['obs']}_N{r['N']}_tau{r['tau']:g}_psi{r['psi']:g}_s{r['seed']}"
      for r in runs if r["exploit_dp"].get("finite_horizon")]
summary["finite_horizon_br_runs"] = sorted(fh)
summary["marl_r_above_1_over_beta"] = sum(1 for r in runs if r["exploit_dp"].get("beta_Ra", 0) >= 1)
json.dump(summary, open("results/summary.json", "w"), indent=1, default=float)
print(json.dumps({k: summary[k] for k in summary if k in ("E2_fit", "E4_elasticities")}, indent=1))
print("assets written")
