"""Which condition is actually producing the edge?

A stacked setup can look powerful while one ingredient does all the work. This
strips the high-tight-flag down to each layer and measures what each one adds.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle, flag

panel, reg = pickle.load(open("data_cache/flag_panel.pkl", "rb"))
H = 63

def conditions(df):
    c, v = df["close"], df["volume"]
    s10, s20 = c.rolling(10).mean(), c.rolling(20).mean()
    return {
        "prior_move":  (c / c.shift(40) - 1 > 0.15),
        "pullback":    (c <= s10 * 1.01).rolling(15).max().astype(bool),
        "intact":      c > s20,
        "contraction": flag._atr(df, 10) / flag._atr(df, 40) < 0.85,
        "adr":         (df["high"] / df["low"] - 1).rolling(20).mean() > 0.02,
        "taper":       v.rolling(5).mean() < v.rolling(20).mean(),
        "breakout":    c > df["high"].shift(1).rolling(5).max(),
    }

def measure(keys, cooldown=10):
    hits, base = [], []
    for _, df in panel.items():
        cond = conditions(df)
        sig = cond[keys[0]].fillna(False)
        for k in keys[1:]:
            sig = sig & cond[k].fillna(False)
        r = reg.reindex(df.index).ffill().fillna(False)
        sig = sig & r
        fwd = df["open"].shift(-1 - H) / df["open"].shift(-1) - 1
        base.extend(fwd[r].dropna().to_numpy())
        last = -10**9
        for i in np.flatnonzero(sig.to_numpy()):
            if i - last < cooldown: continue
            last = i
            if i + 1 + H < len(df):
                hits.append(df["open"].iloc[i + 1 + H] / df["open"].iloc[i + 1] - 1)
    a, b = np.array(hits), np.array(base); b = b[np.isfinite(b)]
    se = a.std() / np.sqrt(len(a)) if len(a) > 1 else np.nan
    return {"n": len(a), "mean": a.mean(), "median": np.median(a),
            "edge": a.mean() - b.mean(), "t": (a.mean() - b.mean()) / se if se else np.nan}

LAYERS = [
    ("breakout only",                    ["breakout"]),
    ("prior_move only",                  ["prior_move"]),
    ("prior_move + breakout",            ["prior_move", "breakout"]),
    ("+ pullback + intact",              ["prior_move", "breakout", "pullback", "intact"]),
    ("+ contraction (the 'coil')",       ["prior_move", "breakout", "pullback", "intact", "contraction"]),
    ("+ ADR",                            ["prior_move", "breakout", "pullback", "intact", "contraction", "adr"]),
    ("+ volume taper = FULL SETUP",      ["prior_move", "breakout", "pullback", "intact", "contraction", "adr", "taper"]),
]

print(f"63-day forward return, baseline = same bullish regime\n")
print(f"  {'layer':<32}{'n':>6}{'mean':>9}{'median':>9}{'edge':>9}{'t':>7}")
for name, keys in LAYERS:
    r = measure(keys)
    print(f"  {name:<32}{r['n']:>6}{r['mean']*100:>8.1f}%{r['median']*100:>8.1f}%"
          f"{r['edge']*100:>+8.2f}%{r['t']:>7.2f}")
