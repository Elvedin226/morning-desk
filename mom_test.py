"""Does the momentum component survive the permutation null?

The ablation showed plain "up 15% over 40 days" carries essentially all of the
flag setup's edge, with 15x the sample. That is the only piece worth testing
properly -- everything layered on top failed to add anything measurable.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pickle, flag

panel, reg = pickle.load(open("data_cache/flag_panel.pkl", "rb"))
H = 63

def momentum_edge(pnl):
    hits, base = [], []
    for _, df in pnl.items():
        c = df["close"]
        sig = (c / c.shift(40) - 1 > 0.15).fillna(False)
        r = reg.reindex(df.index).ffill().fillna(False)
        sig = sig & r
        fwd = df["open"].shift(-1 - H) / df["open"].shift(-1) - 1
        base.extend(fwd[r].dropna().to_numpy())
        last = -10**9
        for i in np.flatnonzero(sig.to_numpy()):
            if i - last < 10: continue
            last = i
            if i + 1 + H < len(df):
                hits.append(df["open"].iloc[i + 1 + H] / df["open"].iloc[i + 1] - 1)
    a, b = np.array(hits), np.array(base); b = b[np.isfinite(b)]
    return a.mean() - b.mean(), len(a)

real, n = momentum_edge(panel)
print(f"real momentum edge: {real*100:+.2f}%  (n={n:,})")

rng = np.random.default_rng(0)
nulls = []
for _ in range(30):
    e, _ = momentum_edge(flag.shuffle_panel(panel, rng))
    if np.isfinite(e): nulls.append(e)
nulls = np.array(nulls)
p = (nulls >= real).mean()
print()
print(f"permutation null ({len(nulls)} shuffled panels):")
print(f"  median {np.median(nulls)*100:+.2f}%   90th pct {np.percentile(nulls,90)*100:+.2f}%"
      f"   max {nulls.max()*100:+.2f}%")
print(f"  shuffles beating the real edge: {p*100:.0f}%   (p = {p:.2f})")
print()
print("  verdict:", "SURVIVES the noise baseline" if p <= 0.10
      else "indistinguishable from noise")
