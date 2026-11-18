import os, sys, json, time, argparse, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor

SEEDS_ALL = [42, 7, 13, 99, 2024, 1, 2, 3, 4, 5]
T_MIN = 10
N_OP = 11
RISK_COUNTRIES = 5          # country ids 5..9 are the 'risk' set (RU, KP, IR, CN, BY); 0..4 normal
CATS = ["A1 Insider threat", "A2 Compromised admin", "A3 Unusual time", "A4 Abnormal amount",
        "A5 Rapid approvals", "A6 New device", "A7 Geo anomaly", "A8 Collusion"]
FEATURES = ['log_amount', 'amount_zscore', 'hour', 'off_hours', 'weekend', 'signer_count',
            'low_signer_flag', 'approval_time_sec', 'rapid_approval', 'device_known',
            'ip_reputation', 'low_ip_rep', 'country_risk', 'prev_tx_gap_hours', 'high_freq',
            'hour_dev', 'latency_ratio']
NAMES = ['Log amount', 'Amount z-score', 'Hour of day', 'Off-hours flag', 'Weekend flag',
         'Signer count', 'Low-signer flag', 'Approval time (s)', 'Rapid-approval flag',
         'Device known', 'IP reputation', 'Low IP rep flag', 'Country risk flag',
         'Prev tx gap (h)', 'High-frequency flag', 'Hour deviation', 'Latency ratio']
# paper numbering x_j of each column in FEATURES
XID = [1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 6, 10]
BLUE, RED, ORANGE, GREEN, GREY = '#378ADD', '#E24B4A', '#BA7517', '#639922', '#aaaaaa'
GRID = '#e8e8e0'

# population reference used only by the *adversary* in the evasion scenario
_r0 = np.random.default_rng(0)
_a0 = np.clip(_r0.lognormal(8.5, 1.2, 400_000), 500, 500_000)
AMT_EDGE = float(_a0.mean() + 2.9 * _a0.std())     # just below the 3-sigma amount signature


# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA GENERATION
# ──────────────────────────────────────────────────────────────────────────────
def draw_normal(rng, n, drifted=None, drift=(0.0, 0.0, 0.0)):
    """Normal behaviour. drift = (d_hour, d_log_amount, d_latency_factor) applied where drifted."""
    hour = np.clip(rng.normal(13, 3, n), 7, 20)
    amount = rng.lognormal(8.5, 1.2, n)
    appr = np.abs(rng.normal(120, 40, n))
    if drifted is not None:
        hour = np.where(drifted, np.clip(hour + drift[0], 7, 20), hour)
        amount = np.where(drifted, amount * np.exp(drift[1]), amount)
        appr = np.where(drifted, appr * (1 + drift[2]), appr)
    return dict(
        operator=rng.integers(0, N_OP, n),
        hour=hour.astype(int),
        amount=np.round(np.clip(amount, 500, 500_000), 2),
        signers=rng.choice([3, 4, 5], n, p=[0.5, 0.35, 0.15]),
        approval=np.round(appr, 1),
        device_known=np.ones(n, int),
        country=rng.integers(0, 5, n),
        ip=np.round(rng.uniform(0.7, 1.0, n), 3),
        gap=np.round(np.abs(rng.normal(6, 2, n)), 2),
        weekend=(rng.random(n) < 0.28).astype(int),
    )


def _mix_lin(rng, base, tgt, s, e, edge):
    t = (1 - e) * tgt + e * edge
    return (1 - s) * base + s * t


def _mix_log(rng, base, tgt, s, e, edge):
    t = np.exp((1 - e) * np.log(tgt) + e * np.log(edge))
    return np.exp((1 - s) * np.log(base) + s * np.log(t))


def _disc(rng, base, tgt, s, hide_val=None, e=0.0):
    """discrete deviation applied with prob s; with prob e the attacker hides it (hide_val)."""
    n = len(base)
    out = np.where(rng.random(n) < s, tgt, base)
    if hide_val is not None and e > 0:
        out = np.where(rng.random(n) < e, hide_val, out)
    return out


def make_anomalies(rng, per_type, severity, evasion, a2_k, drifted_all=None, drift=(0, 0, 0)):
    rows = []
    s, e = severity, evasion
    for c in range(8):
        n = per_type
        dr = None if drifted_all is None else drifted_all[c * n:(c + 1) * n]
        b = draw_normal(rng, n, dr, drift)
        U = lambda lo, hi: rng.uniform(lo, hi, n)
        edge_h = rng.choice([7, 20], n)
        if c == 0:    # A1 insider
            b['hour'] = _disc(rng, b['hour'], rng.choice([1, 2, 3, 4, 22, 23], n), s, edge_h, e)
            b['device_known'] = _disc(rng, b['device_known'], 0, s, 1, e)
            b['amount'] = _mix_log(rng, b['amount'], U(8e5, 5e6), s, e, AMT_EDGE)
        elif c == 1:  # A2 compromised admin (k = t by signature; a2_k=1 reproduces the old generator)
            b['signers'] = _disc(rng, b['signers'], a2_k, s)
            b['approval'] = _mix_lin(rng, b['approval'], U(1, 8), s, e, 41.0)
            b['amount'] = _mix_log(rng, b['amount'], U(5e5, 3e6), s, e, AMT_EDGE)
        elif c == 2:  # A3 unusual time
            b['hour'] = _disc(rng, b['hour'], rng.choice([0, 1, 2, 3, 4, 23], n), s, edge_h, e)
            b['weekend'] = _disc(rng, b['weekend'], 1, s)
        elif c == 3:  # A4 abnormal amount
            b['amount'] = _mix_log(rng, b['amount'], U(2e6, 1e7), s, e, AMT_EDGE)
        elif c == 4:  # A5 rapid approvals
            b['approval'] = _mix_lin(rng, b['approval'], U(1, 5), s, e, 10.5)
            b['signers'] = _disc(rng, b['signers'], 5, s)
        elif c == 5:  # A6 new device
            b['device_known'] = _disc(rng, b['device_known'], 0, s, 1, e)
            b['amount'] = _mix_log(rng, b['amount'], U(3e5, 2e6), s, e, AMT_EDGE)
        elif c == 6:  # A7 geo anomaly
            b['country'] = _disc(rng, b['country'], rng.integers(5, 10, n), s, rng.integers(0, 5, n), e)
            b['ip'] = _mix_lin(rng, b['ip'], U(0.0, 0.3), s, e, 0.41)
        elif c == 7:  # A8 collusion
            b['approval'] = _mix_lin(rng, b['approval'], U(2, 10), s, e, 10.5)
            b['signers'] = _disc(rng, b['signers'], rng.choice([4, 5], n), s)
            b['gap'] = _mix_lin(rng, b['gap'], U(0.01, 0.5), s, e, 0.51)
        b['amount'] = np.round(np.clip(b['amount'], 500, None), 2)
        b['approval'] = np.round(np.maximum(b['approval'], 0.5), 1)
        b['ip'] = np.round(np.clip(b['ip'], 0, 1), 3)
        b['gap'] = np.round(np.maximum(b['gap'], 0.01), 3)
        b['label'] = np.ones(n, int)
        b['atype'] = np.full(n, c + 1)
        rows.append(pd.DataFrame(b))
    return pd.concat(rows, ignore_index=True)


def make_data(seed, n0=1800, n1=200, severity=1.0, evasion=0.0, benign=0.0, a2_k=3,
              drift=None, drift_from=0.6):
    rng = np.random.default_rng(seed)
    n = n0 + n1
    ts = rng.permutation(n) / n
    drift_v = (0, 0, 0) if drift is None else drift
    dr0 = (ts[:n0] >= drift_from) if drift is not None else None
    normal = pd.DataFrame(draw_normal(rng, n0, dr0, drift_v))
    normal['label'] = 0
    normal['atype'] = 0
    # benign look-alikes: legitimate but unusual normal transactions (label 0)
    if benign > 0:
        k = int(round(benign * n0))
        idx = rng.choice(n0, k, replace=False)
        kind = rng.integers(0, 5, k)
        for j, i in enumerate(idx):
            if kind[j] == 0:   normal.loc[i, 'amount'] = round(float(rng.uniform(2e5, 2e6)), 2)
            elif kind[j] == 1: normal.loc[i, 'hour'] = int(rng.choice([5, 6, 21, 22]))
            elif kind[j] == 2: normal.loc[i, 'device_known'] = 0
            elif kind[j] == 3: normal.loc[i, 'approval'] = round(float(rng.uniform(12, 30)), 1)
            else:
                normal.loc[i, 'country'] = int(rng.integers(5, 10))
                normal.loc[i, 'ip'] = round(float(rng.uniform(0.45, 0.8)), 3)
    dr1 = None if drift is None else (ts[n0:] >= drift_from)
    anom = make_anomalies(rng, n1 // 8, severity, evasion, a2_k, dr1, drift_v)
    df = pd.concat([normal, anom], ignore_index=True)
    df['ts'] = np.r_[ts[:n0], ts[n0:]]
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df['tx_id'] = [f'TX{i:04d}' for i in range(len(df))]
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. PROFILES AND FEATURES
# ──────────────────────────────────────────────────────────────────────────────
def build_profile(hist, mode="meanstd", t_min=T_MIN, use_labels=False, per_op_cap=None):
    """Per-operator profile from a history DataFrame. use_labels=True reproduces the old
    label-0 behaviour (LEAK) and exists only for the protocol-comparison table."""
    src = hist[hist.label == 0] if use_labels else hist
    if per_op_cap is not None:
        src = src.sort_values('ts').groupby('operator').head(per_op_cap) if per_op_cap > 0 else src.iloc[0:0]

    def stats(g):
        if mode == "robust":
            ma = np.median(g.amount); sa = 1.4826 * np.median(np.abs(g.amount - ma))
            mh = np.median(g.hour); ml = np.median(g.approval)
        else:
            ma, sa = g.amount.mean(), g.amount.std()
            mh, ml = g.hour.mean(), g.approval.mean()
        return ma, (sa if np.isfinite(sa) and sa > 0 else 1.0), mh, ml

    ref = hist if len(src) < 2 else src      # population fallback source
    pop = stats(ref)
    tab = np.tile(np.array(pop), (N_OP, 1))
    for op in range(N_OP):
        g = src[src.operator == op]
        if len(g) >= t_min:
            tab[op] = stats(g)
    return tab


def features(df, tab):
    op = df.operator.values
    mu_a, sd_a, mu_h, mu_l = tab[op, 0], tab[op, 1], tab[op, 2], tab[op, 3]
    a = df.amount.values; h = df.hour.values; k = df.signers.values
    l = df.approval.values; ip = df.ip.values; gap = df.gap.values
    cols = [np.log1p(a), (a - mu_a) / (sd_a + 1e-9), h, ((h < 7) | (h > 20)).astype(int),
            df.weekend.values, k, (k < 3).astype(int), l, (l < 10).astype(int),
            df.device_known.values, ip, (ip < 0.4).astype(int),
            (df.country.values >= RISK_COUNTRIES).astype(int), gap, (gap < 0.5).astype(int),
            np.abs(h - mu_h), l / (mu_l + 1e-9)]
    return np.column_stack(cols).astype(float)


# ──────────────────────────────────────────────────────────────────────────────
# 3. MODELS
# ──────────────────────────────────────────────────────────────────────────────
class NumpyAE:
    """17-12-6-12-17 ReLU autoencoder, plain mini-batch SGD (as in the paper)."""
    def __init__(self, n_feat, seed, lr=0.005, epochs=120, batch=64):
        r = np.random.default_rng(seed)
        self.r, self.lr, self.epochs, self.batch = r, lr, epochs, batch
        dims = [n_feat, 12, 6, 12, n_feat]
        self.W = [r.normal(0, 0.1, (dims[i], dims[i + 1])) for i in range(4)]
        self.b = [np.zeros(dims[i + 1]) for i in range(4)]
        self.losses = []

    def _fwd(self, x):
        hs = [x]
        for i in range(3):
            hs.append(np.maximum(0, hs[-1] @ self.W[i] + self.b[i]))
        hs.append(hs[-1] @ self.W[3] + self.b[3])
        return hs

    def fit(self, X):
        nb = int(np.ceil(len(X) / self.batch))
        for _ in range(self.epochs):
            idx = self.r.permutation(len(X)); tot = 0.0
            for s in range(0, len(X), self.batch):
                x = X[idx[s:s + self.batch]]
                hs = self._fwd(x); out = hs[-1]
                tot += np.mean((x - out) ** 2)
                d = 2 * (out - x) / len(x)
                gW, gb = [None] * 4, [None] * 4
                gW[3] = hs[3].T @ d; gb[3] = d.sum(0)
                for i in (2, 1, 0):
                    d = (d @ self.W[i + 1].T) * (hs[i + 1] > 0)
                    gW[i] = hs[i].T @ d; gb[i] = d.sum(0)
                for i in range(4):
                    self.W[i] -= self.lr * gW[i]; self.b[i] -= self.lr * gb[i]
            self.losses.append(tot / nb)       # mean per-feature MSE over mini-batches
        return self

    def score(self, X):
        return np.mean((X - self._fwd(X)[-1]) ** 2, axis=1)


def fit_models(Xtr_raw, ytr, seed, ae_train="all", scaler_on=None):
    scaler = StandardScaler().fit(Xtr_raw if scaler_on is None else scaler_on)
    Xs = scaler.transform(Xtr_raw)
    iso = IsolationForest(n_estimators=200, contamination=0.10, random_state=seed).fit(Xs)
    svm = OneClassSVM(kernel='rbf', nu=0.10, gamma='scale').fit(Xs)
    ae = NumpyAE(Xs.shape[1], seed)
    ae.fit(Xs if ae_train == "all" else Xs[ytr == 0])
    return dict(scaler=scaler, iso=iso, svm=svm, ae=ae)


def raw_scores(M, Xraw):
    Xs = M['scaler'].transform(Xraw)
    return dict(iso=-M['iso'].score_samples(Xs), svm=-M['svm'].score_samples(Xs), ae=M['ae'].score(Xs))


def make_norm(raw_tr):
    return {k: (v.min(), v.max()) for k, v in raw_tr.items()}


def apply_norm(raw, nrm):
    return {k: np.clip((v - nrm[k][0]) / (nrm[k][1] - nrm[k][0] + 1e-12), 0, 1) for k, v in raw.items()}


GRID_W = [(i / 20, j / 20, (20 - i - j) / 20) for i in range(21) for j in range(21 - i)]


def calibrate_weights(sv, yv):
    best, bw = None, None
    for w in GRID_W:
        e = w[0] * sv['iso'] + w[1] * sv['ae'] + w[2] * sv['svm']
        key = (round(average_precision_score(yv, e), 6), round(roc_auc_score(yv, e), 6), -max(w))
        if best is None or key > best:
            best, bw = key, w
    return bw


def ens(s, w):
    return w[0] * s['iso'] + w[1] * s['ae'] + w[2] * s['svm']


def tier_of(x):
    return np.where(x < 0.35, 0, np.where(x < 0.70, 1, 2))      # 0 low, 1 medium, 2 high


def metrics(y, s):
    return roc_auc_score(y, s), average_precision_score(y, s)


# ──────────────────────────────────────────────────────────────────────────────
# 4. ONE RUN (one seed, one protocol, one scenario)
# ──────────────────────────────────────────────────────────────────────────────
def run_seed(seed, protocol="clean", scen=None, ae_train="all", profile_mode="meanstd",
             per_op_cap=None, extras=False, df=None):
    scen = scen or {}
    if df is None:
        df = make_data(seed, **scen)
    y_all = df.label.values
    if protocol == "leaky":              # the ORIGINAL protocol: everything in-sample + label-0 profiles
        tab = build_profile(df, profile_mode, use_labels=True)
        X = features(df, tab)
        M = fit_models(X, y_all, seed, ae_train="normal")
        raw = raw_scores(M, X); nrm = make_norm(raw); s = apply_norm(raw, nrm)
        w = (0.45, 0.35, 0.20)
        e = ens(s, w)
        out = dict(seed=seed, y=y_all, at=df.atype.values, ens=e, **{k: s[k] for k in s}, w=w, nval=0)
        return _summ(out)
    idx = np.arange(len(df))
    tv, te = train_test_split(idx, test_size=0.20, stratify=df.atype.values, random_state=seed)
    tr, va = train_test_split(tv, test_size=0.25, stratify=df.atype.values[tv], random_state=seed)
    dtr, dva, dte = df.iloc[tr], df.iloc[va], df.iloc[te]
    use_lab = protocol == "clean_profilelabels"
    tab = build_profile(dtr, profile_mode, use_labels=use_lab, per_op_cap=per_op_cap)
    Xtr, Xva, Xte = features(dtr, tab), features(dva, tab), features(dte, tab)
    ytr, yva, yte = dtr.label.values, dva.label.values, dte.label.values
    M = fit_models(Xtr, ytr, seed, ae_train=("normal" if protocol == "clean_semi" else ae_train))
    raw_tr = raw_scores(M, Xtr); nrm = make_norm(raw_tr)
    s_va = apply_norm(raw_scores(M, Xva), nrm); s_te = apply_norm(raw_scores(M, Xte), nrm)
    w = calibrate_weights(s_va, yva)
    e_va, e_te = ens(s_va, w), ens(s_te, w)
    thr = float(np.quantile(e_va[yva == 0], 0.98))               # 2% FPR operating point on VAL
    out = dict(seed=seed, y=yte, at=dte.atype.values, ens=e_te, iso=s_te['iso'], ae=s_te['ae'],
               svm=s_te['svm'], w=w, nval=len(va), thr=thr, ae_losses=M['ae'].losses)
    res = _summ(out)
    res['thr'] = thr
    if extras:
        import shap
        expl = shap.TreeExplainer(M['iso'])
        sh_tr = -expl.shap_values(M['scaler'].transform(Xtr))
        sh_te = -expl.shap_values(M['scaler'].transform(Xte))
        res['shap_tr'] = np.abs(sh_tr).mean(0)
        res['shap_te'] = np.abs(sh_te).mean(0)
        res['shap_te_signed_geo'] = None
        # ablation: top-k by TRAIN-set IF-SHAP rank (no labels, no test data)
        order = np.argsort(-res['shap_tr'])
        abl = {}
        for k in (3, 5, 7, 10, 13, 17):
            cols = order[:k]
            sc = StandardScaler().fit(Xtr[:, cols])
            f = IsolationForest(n_estimators=200, contamination=0.10, random_state=seed).fit(sc.transform(Xtr[:, cols]))
            sk = -f.score_samples(sc.transform(Xte[:, cols]))
            abl[k] = metrics(yte, sk)
        res['ablation'] = abl
        res['rank_order'] = order
        res['train_stats'] = dict(n=len(tr))
    res['_models'] = None
    return res


def _summ(o):
    y = o['y']; r = dict(seed=o['seed'], w=o['w'], y=y, at=o['at'], ens=o['ens'], iso=o['iso'],
                         ae=o['ae'], svm=o['svm'], ae_losses=o.get('ae_losses'))
    for k in ('iso', 'ae', 'svm', 'ens'):
        r['auc_' + k], r['ap_' + k] = metrics(y, o[k])
    t = tier_of(o['ens'])
    r['tier'] = t
    thr = o.get('thr', None)
    if thr is not None:
        fl = o['ens'] > thr
        r['alert_recall'] = float(fl[y == 1].mean()); r['alert_fpr'] = float(fl[y == 0].mean())
        r['alert_prec'] = float(y[fl].mean()) if fl.sum() else np.nan
    r['med_recall'] = float((t[y == 1] >= 1).mean()); r['med_fpr'] = float((t[y == 0] >= 1).mean())
    r['high_recall'] = float((t[y == 1] == 2).mean()); r['high_fpr'] = float((t[y == 0] == 2).mean())
    return r


def _job(args):
    return run_seed(*args[0], **args[1])


def run_many(seeds, workers, **kw):
    jobs = [((s,), kw) for s in seeds]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            return list(ex.map(_job, jobs))
    return [_job(j) for j in jobs]


# ──────────────────────────────────────────────────────────────────────────────
# 5. AGGREGATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def ms(vals, nd=4):
    v = np.array(vals, float)
    return f"{np.nanmean(v):.{nd}f}", f"{np.nanstd(v, ddof=1):.{nd}f}"


def pooled(runs, key):
    return np.concatenate([r[key] for r in runs])


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 150})


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + '.png'), dpi=200, bbox_inches='tight'); plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 6. STAGES
# ──────────────────────────────────────────────────────────────────────────────
def stage_core(out, seeds, workers, T):
    print("[core] protocol comparison, main tables, figures", flush=True)
    # --- protocol comparison (R1-6 / R1-7 / label leakage) --------------------------------
    protos = [("Original (in-sample, label-0 profiles, global scaler)", dict(protocol="leaky")),
              ("Held-out, profiles from label-0 train rows", dict(protocol="clean_profilelabels")),
              ("Held-out, unlabeled profiles, AE on label-0 train rows", dict(protocol="clean_semi")),
              ("Held-out, fully unsupervised (main protocol)", dict(protocol="clean"))]
    prot_rows, main_runs = [], None
    for name, kw in protos:
        main_key = 'clean' if T.get('main', 'unsup') == 'unsup' else 'clean_semi'
        runs = run_many(seeds, workers, extras=(kw['protocol'] == main_key), **kw)
        if kw['protocol'] == main_key:
            main_runs = runs
        row = [name] + [ms([r[f'auc_{k}'] for r in runs], 4)[0] + " $\\pm$ " + ms([r[f'auc_{k}'] for r in runs], 4)[1]
                        for k in ('iso', 'ae', 'svm', 'ens')]
        row += [ms([r['ap_ens'] for r in runs], 4)[0] + " $\\pm$ " + ms([r['ap_ens'] for r in runs], 4)[1]]
        prot_rows.append(row)
        print("   ", name, "ENS AUROC", ms([r['auc_ens'] for r in runs])[0], "AP", ms([r['ap_ens'] for r in runs])[0], flush=True)
    T['protocol'] = prot_rows
    runs = main_runs
    r42 = [r for r in runs if r['seed'] == 42][0]

    # --- Table: dataset stats (seed 42 full dataset) --------------------------------------
    d = make_data(42)
    nrm_, an_ = d[d.label == 0], d[d.label == 1]
    rowsd = []
    for lab, col in [("Amount (USD)", 'amount'), ("Approval time (s)", 'approval'), ("Hour of initiation", 'hour'),
                     ("Signer count", 'signers'), ("IP reputation", 'ip')]:
        rowsd.append((lab, nrm_[col].mean(), nrm_[col].median(), an_[col].mean(), an_[col].median()))
    T['dataset'] = rowsd
    T['dataset_flags'] = dict(dev_norm=100 * nrm_.device_known.mean(), dev_an=100 * an_.device_known.mean(),
                              geo_norm=100 * (nrm_.country >= 5).mean(), geo_an=100 * (an_.country >= 5).mean())

    # --- Table: model performance (seed 42) & stability ----------------------------------
    T['seed42'] = {k: (r42['auc_' + k], r42['ap_' + k]) for k in ('iso', 'ae', 'svm', 'ens')}
    T['seed42_w'] = r42['w']
    T['stability'] = {k: (ms([r['auc_' + k] for r in runs]), ms([r['ap_' + k] for r in runs])) for k in ('iso', 'ae', 'svm', 'ens')}
    T['weights_all'] = [r['w'] for r in runs]
    wc = {}
    for nm_, w_ in [("Calibrated on validation (per seed)", None), ("Fixed (0.45, 0.35, 0.20)", (0.45, 0.35, 0.20)),
                    ("Equal (1/3 each)", (1/3, 1/3, 1/3))]:
        a_, p_ = [], []
        for r in runs:
            e_ = r['ens'] if w_ is None else w_[0]*r['iso'] + w_[1]*r['ae'] + w_[2]*r['svm']
            u, v = metrics(r['y'], e_); a_.append(u); p_.append(v)
        wc[nm_] = (ms(a_), ms(p_))
    T['weights_cmp'] = wc
    T['n_test'] = int(len(r42['y'])); T['n_test_an'] = int(r42['y'].sum())

    # --- tiers (pooled over seeds) --------------------------------------------------------
    y, t, at = pooled(runs, 'y'), pooled(runs, 'tier'), pooled(runs, 'at')
    T['tiers'] = {lab: [100 * np.mean(t[m] == k) for k in (0, 1, 2)] for lab, m in
                  [("Normal", y == 0), ("Anomalous", y == 1)]}
    T['tiers_n'] = (int((y == 0).sum()), int((y == 1).sum()))
    T['percat'] = []
    for c in range(1, 9):
        m = at == c
        hi, me, lo = [100 * np.mean(t[m] == k) for k in (2, 1, 0)]
        T['percat'].append((CATS[c - 1], hi, me, lo, hi + me))
    m = y == 1
    T['percat_all'] = tuple([100 * np.mean(t[m] == k) for k in (2, 1, 0)]) + (100 * np.mean(t[m] >= 1),)
    T['percat_n'] = int((at == 1).sum())
    # operating-point table
    T['opoint'] = {k: ms([r[k] for r in runs], 3) for k in ('alert_recall', 'alert_fpr', 'alert_prec', 'med_recall', 'med_fpr', 'high_recall', 'high_fpr')}
    # per-category AUROC vs normals (pooled) for the response letter
    en = pooled(runs, 'ens')
    T['percat_auc'] = [(CATS[c - 1], roc_auc_score(np.r_[np.zeros((y == 0).sum()), np.ones((at == c).sum())],
                                                   np.r_[en[y == 0], en[at == c]])) for c in range(1, 9)]

    # --- ablation ------------------------------------------------------------------------
    ranks = np.mean([np.argsort(np.argsort(-r['shap_tr'])) + 1 for r in runs], 0)
    mean_order = np.argsort(ranks)
    T['ablation'] = []
    for k in (3, 5, 7, 10, 13, 17):
        a = [r['ablation'][k][0] for r in runs]; p = [r['ablation'][k][1] for r in runs]
        T['ablation'].append((k, ms(a), ms(p)))
    T['ablation_order'] = [NAMES[i] for i in mean_order]

    # --- SHAP ranking tables ---------------------------------------------------------------
    sh = np.mean([r['shap_te'] for r in runs], 0)
    rk = (-sh).argsort().argsort() + 1
    T['shap_if'] = dict(val=sh, rank=rk, share=100 * sh / sh.sum())
    T['shap_if_rank_train'] = ranks
    pd.DataFrame(dict(x_id=XID, feature=NAMES, IF_TreeSHAP_test_mean=sh.round(4), IF_rank_test=rk,
                      IF_share_pct=(100 * sh / sh.sum()).round(1),
                      IF_rank_train_meanacrossseeds=ranks.round(1))).sort_values('x_id').to_csv(
        os.path.join(out, 'shap_ranking_if.csv'), index=False)
    T['_runs_main'] = runs

    # --- figures ---------------------------------------------------------------------------
    style()
    # Fig 1 ROC / PR (pooled test scores, 10 seeds)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for k, lab, col, lw in [('iso', 'Isolation Forest', BLUE, 1.5), ('ae', 'Autoencoder', GREEN, 1.5),
                            ('svm', 'One-Class SVM', ORANGE, 1.5), ('ens', 'Ensemble', RED, 2.2)]:
        s = pooled(runs, k); f, tp, _ = roc_curve(y, s)
        ax[0].plot(f, tp, color=col, lw=lw, label=f"{lab} (AUC={roc_auc_score(y, s):.3f})")
    ax[0].plot([0, 1], [0, 1], '--', color=GREY, lw=1)
    ax[0].set_xlabel('False positive rate'); ax[0].set_ylabel('True positive rate')
    ax[0].set_title('ROC curves, held-out test (pooled, 10 seeds)'); ax[0].legend(fontsize=9, loc='lower right')
    pr, rc, _ = precision_recall_curve(y, en)
    ax[1].plot(rc, pr, color=RED, lw=2, label=f"Ensemble (AP={average_precision_score(y, en):.3f})")
    ax[1].axhline(y.mean(), color=GREY, ls='--', lw=1, label='Baseline (random)')
    ax[1].set_xlabel('Recall'); ax[1].set_ylabel('Precision'); ax[1].set_title('Precision-recall curve (ensemble)')
    ax[1].legend(fontsize=9)
    for a in ax: a.grid(True, color=GRID, lw=0.5)
    plt.tight_layout(); save(fig, out, 'fig1_roc_pr')
    # Fig 3 risk dist
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(en[y == 0], bins=50, alpha=0.7, color=BLUE, label='Normal', density=True)
    ax[0].hist(en[y == 1], bins=50, alpha=0.7, color=RED, label='Anomalous', density=True)
    ax[0].axvline(0.35, color=ORANGE, ls='--', lw=1.2, label='Low/Medium threshold')
    ax[0].axvline(0.70, color=RED, ls='--', lw=1.2, label='Medium/High threshold')
    ax[0].set_xlabel('Ensemble risk score'); ax[0].set_ylabel('Density'); ax[0].set_title('Risk score distribution (held-out test)')
    ax[0].legend(fontsize=9); ax[0].grid(True, color=GRID, lw=0.5)
    cnt = [int((at == c).sum()) for c in range(1, 9)]
    ax[1].barh(CATS[::-1], cnt[::-1], color=BLUE, height=0.7)
    ax[1].set_xlabel('Count (test, pooled over seeds)'); ax[1].set_title('Anomaly types in the evaluation sets')
    ax[1].grid(True, axis='x', color=GRID, lw=0.5)
    plt.tight_layout(); save(fig, out, 'fig3_risk_dist')
    # Fig 6 per-category
    fig, ax = plt.subplots(figsize=(12, 5)); x = np.arange(8); wd = 0.27
    for j, (lab, col, k) in enumerate([('High', RED, 1), ('Medium', ORANGE, 2), ('Low', BLUE, 3)]):
        vals = [T['percat'][c][k] for c in range(8)]
        b = ax.bar(x + (j - 1) * wd, vals, wd, color=col, label=lab)
        if lab == 'High':
            for rect, v in zip(b, vals): ax.text(rect.get_x() + rect.get_width() / 2, v + 1, f"{v:.0f}", ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([c.split(' ', 1)[0] + '\n' + c.split(' ', 1)[1] for c in CATS], fontsize=9)
    ax.set_ylabel('Share of instances (%)'); ax.set_title('Risk tier by attack category (held-out test, pooled)')
    ax.legend(); ax.grid(True, axis='y', color=GRID, lw=0.5); plt.tight_layout(); save(fig, out, 'fig6_per_category')
    # Fig 7 ablation
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5)); ks = [a[0] for a in T['ablation']]
    for a_, idx_, lab in [(ax[0], 0, 'AUROC'), (ax[1], 1, 'Average precision')]:
        mu = [float(a[1 + idx_][0]) for a in T['ablation']]; sd = [float(a[1 + idx_][1]) for a in T['ablation']]
        a_.errorbar(ks, mu, yerr=sd, marker='o', color=BLUE, capsize=3)
        a_.axhline(mu[-1], color=GREY, ls='--', lw=1, label='All 17 features')
        a_.set_xlabel('Number of top-k features (TRAIN-set SHAP rank)'); a_.set_ylabel(lab)
        a_.set_xticks(ks); a_.grid(True, color=GRID, lw=0.5); a_.legend(fontsize=9)
    plt.tight_layout(); save(fig, out, 'fig7_ablation')
    # Fig 8 stability
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a_, m_, lab in [(ax[0], 'auc', 'AUROC'), (ax[1], 'ap', 'Average precision')]:
        data = [[r[f'{m_}_{k}'] for r in runs] for k in ('iso', 'ae', 'svm', 'ens')]
        a_.boxplot(data, tick_labels=['IF', 'AE', 'OCSVM', 'Ensemble'], patch_artist=True,
                   boxprops=dict(facecolor='#dbe9f8'), medianprops=dict(color=RED))
        a_.set_ylabel(lab); a_.grid(True, axis='y', color=GRID, lw=0.5)
    plt.tight_layout(); save(fig, out, 'fig8_stability')
    # Fig 9 AE loss (seed 42)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(r42['ae_losses'], color=BLUE, lw=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Training loss (MSE per feature, batch mean)')
    ax.set_title('Autoencoder training on the unlabeled training split'); ax.grid(True, color=GRID, lw=0.5)
    plt.tight_layout(); save(fig, out, 'fig9_ae_loss')
    T['ae_loss'] = (float(r42['ae_losses'][0]), float(r42['ae_losses'][-1]))


def stage_robust(out, seeds, workers, T):
    print("[robust] sensitivity scenarios", flush=True)
    scen = [("Baseline (paper generator, $k{=}t$ for A2)", {}),
        ("Legacy A2 ($k{=}1$, as in the submitted code)", dict(a2_k=1)),
        ("Milder anomalies, severity 0.5", dict(severity=0.5)),
        ("Milder anomalies, severity 0.25", dict(severity=0.25)),
        ("Benign look-alikes, 5\\% of normals", dict(benign=0.05)),
        ("Benign look-alikes, 10\\% of normals", dict(benign=0.10)),
        ("Adaptive evasion, strength 0.5", dict(evasion=0.5)),
        ("Adaptive evasion, strength 1.0", dict(evasion=1.0)),
        ("Combined: severity 0.5 + 5\\% benign + evasion 0.5", dict(severity=0.5, benign=0.05, evasion=0.5))]
    rows = []
    for name, kw in scen:
        runs = run_many(seeds, workers, scen=kw)
        pc = []
        at, t, y = pooled(runs, 'at'), pooled(runs, 'tier'), pooled(runs, 'y')
        for c in range(1, 9):
            pc.append(100 * np.mean(t[at == c] >= 1))
        rows.append(dict(name=name, auc=ms([r['auc_ens'] for r in runs]), ap=ms([r['ap_ens'] for r in runs]),
                         auc_if=ms([r['auc_iso'] for r in runs]), rec=ms([r['alert_recall'] for r in runs], 3),
                         fpr=ms([r['alert_fpr'] for r in runs], 3), medrec=ms([r['med_recall'] for r in runs], 3),
                         medfpr=ms([r['med_fpr'] for r in runs], 3), percat=pc))
        print("   ", name, "AUROC", rows[-1]['auc'][0], "AP", rows[-1]['ap'][0], "recall@2%FPR", rows[-1]['rec'][0], flush=True)
    T['robust'] = rows
    style()
    fig, ax = plt.subplots(1, 3, figsize=(16, 5)); lab = [r['name'].replace('\\%', '%').split(' (')[0].split(':')[0][:40] for r in rows]
    y_ = np.arange(len(rows))[::-1]
    for a_, key, ttl in [(ax[0], 'auc', 'Ensemble AUROC'), (ax[1], 'ap', 'Ensemble AP')]:
        a_.barh(y_, [float(r[key][0]) for r in rows], xerr=[float(r[key][1]) for r in rows], color=BLUE, height=0.65)
        a_.set_title(ttl); a_.grid(True, axis='x', color=GRID, lw=0.5)
        a_.set_yticks(y_); a_.set_yticklabels(lab if a_ is ax[0] else [''] * len(rows), fontsize=8)
    ax[2].barh(y_, [float(r['rec'][0]) for r in rows], xerr=[float(r['rec'][1]) for r in rows], color=RED, height=0.65)
    ax[2].set_title('Recall at 2% validation FPR'); ax[2].set_yticks(y_); ax[2].set_yticklabels([''] * len(rows))
    ax[2].grid(True, axis='x', color=GRID, lw=0.5)
    plt.tight_layout(); save(fig, out, 'fig10_robustness')


def stage_cold(out, seeds, workers, T):
    print("[cold] profile history length (cold start)", flush=True)
    rows = []
    for cap in (0, 3, 9, 10, 30, 100, None):
        runs = run_many(seeds, workers, per_op_cap=cap)
        rows.append(dict(cap=cap, auc=ms([r['auc_ens'] for r in runs]), ap=ms([r['ap_ens'] for r in runs]),
                         medrec=ms([r['med_recall'] for r in runs], 3), medfpr=ms([r['med_fpr'] for r in runs], 3),
                         rec=ms([r['alert_recall'] for r in runs], 3), fpr=ms([r['alert_fpr'] for r in runs], 3)))
        print("   cap", cap, "AUROC", rows[-1]['auc'][0], "medFPR", rows[-1]['medfpr'][0], flush=True)
    T['cold'] = rows
    # robust vs mean/std profile (contaminated training history)
    rr = run_many(seeds, workers, profile_mode="robust")
    T['profile_robust'] = dict(auc=ms([r['auc_ens'] for r in rr]), ap=ms([r['ap_ens'] for r in rr]),
                               medrec=ms([r['med_recall'] for r in rr], 3), medfpr=ms([r['med_fpr'] for r in rr], 3))
    style()
    fig, ax = plt.subplots(figsize=(7, 4)); lab = [("0" if r['cap'] == 0 else "all" if r['cap'] is None else str(r['cap'])) for r in rows]
    ax.errorbar(range(len(rows)), [float(r['auc'][0]) for r in rows], yerr=[float(r['auc'][1]) for r in rows], marker='o', color=BLUE, capsize=3)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(lab); ax.set_xlabel('History per operator used for profiles (tx)')
    ax.set_ylabel('Ensemble AUROC (test)'); ax.grid(True, color=GRID, lw=0.5); ax.axvline(3.5, ls=':', color=GREY)
    ax.text(3.55, ax.get_ylim()[0], ' $T_{min}$=10', fontsize=8, va='bottom')
    plt.tight_layout(); save(fig, out, 'fig11_coldstart')


def _update_job(seed):
    """Chronological stream with behavioural drift; compare static vs rolling profile updates."""
    drift = (0.0, 0.7, 0.3)          # amounts x2, approval latency +30% on legitimate behaviour after t=0.6
    d = make_data(seed, drift=drift, drift_from=0.6)
    d = d.sort_values('ts').reset_index(drop=True)
    # anomalies were generated with their own ts-free base; give them drifted base too (applied below)
    tr = d[d.ts < 0.6]; te = d[d.ts >= 0.6].reset_index(drop=True)
    # apply drift to anomaly bases that are not overwritten: amount/latency scale for A1-A3,A5-A8 handled implicitly
    tab0 = build_profile(tr)
    Xtr = features(tr, tab0)
    M = fit_models(Xtr, tr.label.values, seed)
    nrm = make_norm(raw_scores(M, Xtr)); w = (0.45, 0.35, 0.20)
    out = {}
    chunks = np.array_split(np.arange(len(te)), 8)
    for pol in ("static", "update_low_only", "update_all", "oracle_clean"):
        hist = tr.copy(); sc = np.zeros(len(te)); tab = tab0
        for ch in chunks:
            blk = te.iloc[ch]
            Xb = features(blk, tab)
            s = apply_norm(raw_scores(M, Xb), nrm); e = ens(s, w); sc[ch] = e
            if pol == "update_all": add = blk
            elif pol == "update_low_only": add = blk[e < 0.35]
            elif pol == "oracle_clean": add = blk[blk.label == 0]
            else: add = None
            if add is not None and len(add):
                hist = pd.concat([hist, add]); tab = build_profile(hist)
        y = te.label.values; t = tier_of(sc)
        out[pol] = (roc_auc_score(y, sc), average_precision_score(y, sc), float((t[y == 1] >= 1).mean()), float((t[y == 0] >= 1).mean()))
    return out


def stage_update(out, seeds, workers, T):
    print("[update] profile update policies under drift", flush=True)
    res = [_update_job(s) for s in seeds]
    T['update'] = {pol: [ms([r[pol][i] for r in res], 3) for i in range(4)] for pol in res[0]}
    for pol, v in T['update'].items():
        print("   ", pol, "AUROC", v[0][0], "AP", v[1][0], "medRec", v[2][0], "medFPR", v[3][0], flush=True)
    style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4)); pols = list(T['update'])
    cols = [GREY, GREEN, RED, BLUE]
    ax[0].bar(pols, [float(T['update'][p][0][0]) for p in pols], yerr=[float(T['update'][p][0][1]) for p in pols], color=cols)
    ax[0].set_title('AUROC under drift'); ax[1].bar(pols, [float(T['update'][p][3][0]) for p in pols], yerr=[float(T['update'][p][3][1]) for p in pols], color=cols)
    ax[1].set_title('False alert rate on normals (tier >= medium)')
    for a in ax: a.tick_params(axis='x', labelrotation=20); a.grid(True, axis='y', color=GRID, lw=0.5)
    plt.tight_layout(); save(fig, out, 'fig12_profile_update')


def stage_shap(out, T):
    """Ensemble Kernel SHAP + IF TreeSHAP figures (seed 42 held-out test)."""
    import shap
    print("[shap] Kernel SHAP on the full ensemble (seed 42)", flush=True)
    seed = 42
    df = make_data(seed); idx = np.arange(len(df))
    tv, te = train_test_split(idx, test_size=0.20, stratify=df.atype.values, random_state=seed)
    tr, va = train_test_split(tv, test_size=0.25, stratify=df.atype.values[tv], random_state=seed)
    dtr, dva, dte = df.iloc[tr], df.iloc[va], df.iloc[te].reset_index(drop=True)
    tab = build_profile(dtr)
    Xtr, Xva, Xte = features(dtr, tab), features(dva, tab), features(dte, tab)
    M = fit_models(Xtr, dtr.label.values, seed, ae_train=('all' if T.get('main', 'unsup') == 'unsup' else 'normal'))
    nrm = make_norm(raw_scores(M, Xtr))
    s_va = apply_norm(raw_scores(M, Xva), nrm); w = calibrate_weights(s_va, dva.label.values)

    def ens_fn(Xs):                      # input is in SCALED space
        raw = dict(iso=-M['iso'].score_samples(Xs), svm=-M['svm'].score_samples(Xs), ae=M['ae'].score(Xs))
        return ens(apply_norm(raw, nrm), w)
    Xs_te = M['scaler'].transform(Xte); Xs_tr = M['scaler'].transform(Xtr)
    e_te = ens_fn(Xs_te); yte = dte.label.values; atte = dte.atype.values
    assert np.allclose(e_te, ens(apply_norm(raw_scores(M, Xte), nrm), w))
    tier = tier_of(e_te)
    tre = shap.TreeExplainer(M['iso']); sh_if = -tre.shap_values(Xs_te)
    rng = np.random.default_rng(0)
    bg = Xs_tr[rng.choice(len(Xs_tr), 60, replace=False)]            # unlabeled background
    kex = shap.KernelExplainer(ens_fn, bg)
    g_an = np.where(yte == 1)[0]; g_no = rng.choice(np.where(yte == 0)[0], 80, replace=False)
    gi = np.sort(np.r_[g_an, g_no])
    kg = kex.shap_values(Xs_te[gi], nsamples=600, silent=True)
    yg = yte[gi]
    kmean = 0.10 * np.abs(kg[yg == 1]).mean(0) + 0.90 * np.abs(kg[yg == 0]).mean(0)   # re-weight to 10% prior
    # --- cases: A1 (high if available), A5 (medium if available), two normal-low
    def pick(mask, pref):
        c = np.where(mask)[0]
        for p in pref:
            cc = [i for i in c if tier[i] == p]
            if cc: return cc[0]
        return c[0]
    cases = [pick(atte == 1, [2, 1, 0]), pick(atte == 5, [1, 2, 0]), pick((yte == 0) & (tier == 0), [0]),
             np.where((yte == 0) & (tier == 0))[0][1]]
    kc = kex.shap_values(Xs_te[cases], nsamples=1500, silent=True)
    sh_rank = pd.DataFrame(dict(x_id=XID, feature=NAMES, Ens_KernelSHAP=kmean.round(4)))
    sh_rank.sort_values('x_id').to_csv(os.path.join(out, 'shap_ranking_ensemble.csv'), index=False)
    T['shap_ens'] = dict(vals=kmean, rank=(-kmean).argsort().argsort() + 1)
    T['shap_cases'] = [dict(i=int(i), at=int(atte[i]), score=float(e_te[i]), tier=int(tier[i]),
                            top=[(NAMES[j], float(kc[n][j])) for j in np.argsort(-np.abs(kc[n]))[:4]])
                       for n, i in enumerate(cases)]
    sh = T['shap_if']['val']                                     # 10-seed mean TreeSHAP on held-out test
    style()

    def bar(ax, v, ttl):
        o = np.argsort(v); top = set(np.argsort(-v)[:3])
        ax.barh([NAMES[i] for i in o], v[o], color=[RED if i in top else BLUE for i in o], height=0.7)
        ax.set_title(ttl, fontsize=10.5); ax.set_xlabel('Mean |SHAP value|'); ax.grid(True, axis='x', color=GRID, lw=0.5)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.5))
    bar(ax[0], sh, '(a) Isolation Forest (TreeSHAP, held-out test, mean of 10 seeds)')
    bar(ax[1], kmean, '(b) Full ensemble (Kernel SHAP, seed 42, prior-weighted)')
    ax[1].legend(handles=[mpatches.Patch(color=RED, label='Top-3 features'), mpatches.Patch(color=BLUE, label='Other features')],
                 fontsize=9, loc='lower right')
    plt.tight_layout(); save(fig, out, 'fig7_shap_global')
    lab = {1: 'A1 insider threat', 5: 'A5 rapid approvals', 0: 'Normal'}
    tn = ['low', 'medium', 'high']
    fig, ax = plt.subplots(2, 4, figsize=(17, 8))
    for row in range(2):
        for j, i in enumerate(cases):
            sv = sh_if[i] if row == 0 else kc[j]; o = np.argsort(np.abs(sv))[-8:]
            ax[row, j].barh([NAMES[k] for k in o], sv[o], color=[RED if sv[k] > 0 else BLUE for k in o], height=0.6)
            ax[row, j].axvline(0, color=GREY, lw=0.8); ax[row, j].tick_params(axis='y', labelsize=8)
            ax[row, j].grid(True, axis='x', color=GRID, lw=0.5)
            if row == 0:
                ax[row, j].set_title(f"{lab.get(atte[i], 'Normal')}\nEnsemble score {e_te[i]:.2f} ({tn[tier[i]]})", fontsize=9)
        ax[row, 0].set_ylabel(['IF component (TreeSHAP)', 'Ensemble (Kernel SHAP)'][row], fontsize=9)
    plt.suptitle('Transaction-level SHAP attributions (red: increases anomaly score; blue: decreases it)', y=1.0)
    plt.tight_layout(); save(fig, out, 'fig8_shap_cases')
    # sign sanity check
    g = np.where(atte == 7)[0]
    T['shap_sign_check'] = dict(ip_reputation=float(sh_if[g, FEATURES.index('ip_reputation')].mean()),
                                low_ip_rep=float(sh_if[g, FEATURES.index('low_ip_rep')].mean()))


# ──────────────────────────────────────────────────────────────────────────────
# 7. LaTeX OUTPUT
# ──────────────────────────────────────────────────────────────────────────────
def fnum(x, nd=1): return f"{x:.{nd}f}"


def write_latex(out, T):
    L = []
    A = L.append
    if 'dataset' in T:
        A("%% ===== Table: dataset statistics (seed 42, full dataset) =====")
        for lab, nm, nmed, am, amed in T['dataset']:
            nd = 0 if lab.startswith("Amount") else (3 if lab.startswith("IP") else (2 if lab.startswith("Signer") else 1))
            f = lambda v: (f"{v:,.0f}".replace(",", "\\,") if nd == 0 else f"{v:.{nd}f}")
            A(f"{lab} & {f(nm)} & {f(nmed)} & {f(am)} & {f(amed)} \\\\")
        d = T['dataset_flags']
        A(f"Device known (\\%) & {d['dev_norm']:.0f}\\% & --- & {d['dev_an']:.0f}\\% & --- \\\\")
        A(f"Country risk flag (\\%) & {d['geo_norm']:.0f}\\% & --- & {d['geo_an']:.1f}\\% & --- \\\\")
    if 'seed42' in T:
        A("\n%% ===== Table: model performance (seed 42, held-out test) =====")
        w = T['seed42_w']; nm = dict(iso='Isolation Forest (\\iforest)', ae='Autoencoder (\\ae)', svm='One-Class SVM (\\ocsvm)')
        wi = dict(iso=w[0], ae=w[1], svm=w[2])
        for k in ('iso', 'ae', 'svm'):
            A(f"{nm[k]} & {T['seed42'][k][0]:.4f} & {T['seed42'][k][1]:.4f} & {wi[k]:.2f} \\\\")
        A("\\midrule")
        A(f"\\textbf{{Ensemble}} & \\textbf{{{T['seed42']['ens'][0]:.4f}}} & \\textbf{{{T['seed42']['ens'][1]:.4f}}} & --- \\\\")
        A(f"%% test set: N={T['n_test']}, anomalies={T['n_test_an']}; weights per seed: {T['weights_all']}")
        A("\n%% ===== Table: multi-seed stability =====")
        nm['ens'] = '\\textbf{Ensemble}'
        for k in ('iso', 'ae', 'svm', 'ens'):
            (am, asd), (pm, psd) = T['stability'][k]
            A(f"{nm[k]} & {am} & {asd} & {pm} & {psd} \\\\")
        A("\n%% ===== Table: protocol comparison (NEW; supports the response to R1-6/R1-7) =====")
        for r in T['protocol']:
            A(" & ".join(r) + " \\\\")
        A("\n%% ===== Ensemble weighting comparison (test, mean and sd over seeds) =====")
        for nm_, (aa, pp) in T['weights_cmp'].items():
            A(f"{nm_} & {aa[0]} $\\pm$ {aa[1]} & {pp[0]} $\\pm$ {pp[1]} \\\\")
        A("\n%% ===== Table: tiers (pooled over seeds) =====")
        for lab, v in T['tiers'].items():
            n = T['tiers_n'][0 if lab == "Normal" else 1]
            ns = f"{n:,}".replace(",", "\\,")
            A(f"{lab} (pooled test, $n{{=}}{ns}$) & {v[0]:.1f} & {v[1]:.1f} & {v[2]:.1f} \\\\")
        A("\n%% ===== Table: per category =====")
        for c, hi, me, lo, de in T['percat']:
            A(f"{c} & {hi:.1f} & {me:.1f} & {lo:.1f} & {de:.1f} \\\\")
        A("\\midrule")
        h, m_, l, dd = T['percat_all']
        A(f"\\textit{{All anomalies}} & {h:.1f} & {m_:.1f} & {l:.1f} & {dd:.1f} \\\\")
        A(f"%% n per category (pooled): {T['percat_n']}")
        A("\n%% ===== Per-category AUROC vs normals (pooled) =====")
        for c, a in T['percat_auc']: A(f"%% {c}: {a:.4f}")
        A("\n%% ===== Operating points (mean, sd over seeds) =====")
        for k, v in T['opoint'].items(): A(f"%% {k}: {v[0]} ({v[1]})")
        A("\n%% ===== Ablation (top-k by TRAIN-set IF-SHAP rank) =====")
        for k, (a, p) in [(k, (a, p)) for k, a, p in T['ablation']]:
            A(f"{k} & {a[0]} $\\pm$ {a[1]} & {p[0]} $\\pm$ {p[1]} \\\\")
        A("%% mean-rank order: " + ", ".join(T['ablation_order']))
        A("\n%% ===== Table 5: SHAP rank (IF, held-out test mean of seeds) in x_1..x_17 order =====")
        rk = T['shap_if']['rank']; byx = {XID[i]: int(rk[i]) for i in range(17)}
        A("%% " + ", ".join(str(byx[j]) for j in range(1, 18)))
        A(f"%% top-3 share of IF SHAP mass: {100*np.sort(T['shap_if']['share'])[::-1][:3].sum()/100:.1f}%")
        A(f"%% AE loss first/last: {T['ae_loss'][0]:.3f} / {T['ae_loss'][1]:.3f}")
    if 'robust' in T:
        A("\n%% ===== Table: robustness scenarios (NEW) =====")
        for r in T['robust']:
            A(f"{r['name']} & {r['auc'][0]} $\\pm$ {r['auc'][1]} & {r['ap'][0]} $\\pm$ {r['ap'][1]} & "
              f"{r['rec'][0]} & {r['fpr'][0]} & {r['medrec'][0]} & {r['medfpr'][0]} \\\\")
        A("\n%% per-category tier>=medium detection (%) per scenario: A1..A8")
        for r in T['robust']: A("%% " + r['name'] + ": " + ", ".join(f"{v:.0f}" for v in r['percat']))
    if 'cold' in T:
        A("\n%% ===== Table: cold start / profile history (NEW) =====")
        for r in T['cold']:
            lab = "all (about 109)" if r['cap'] is None else str(r['cap'])
            A(f"{lab} & {r['auc'][0]} $\\pm$ {r['auc'][1]} & {r['ap'][0]} & {r['rec'][0]} & {r['fpr'][0]} & {r['medrec'][0]} & {r['medfpr'][0]} \\\\")
        p = T['profile_robust']
        A(f"%% robust (median/MAD) profile: AUROC {p['auc'][0]}, AP {p['ap'][0]}, medRec {p['medrec'][0]}, medFPR {p['medfpr'][0]}")
    if 'update' in T:
        A("\n%% ===== Table: profile update under drift (NEW) =====")
        for pol, v in T['update'].items():
            A(f"{pol} & {v[0][0]} $\\pm$ {v[0][1]} & {v[1][0]} & {v[2][0]} & {v[3][0]} \\\\")
    if 'shap_ens' in T:
        A("\n%% ===== SHAP =====")
        A("%% ensemble rank by x_id: " + ", ".join(str(int(T['shap_ens']['rank'][XID.index(j)])) for j in range(1, 18)))
        for c in T['shap_cases']:
            A(f"%% case atype={c['at']} score={c['score']:.2f} tier={c['tier']} top={[(n, round(v, 3)) for n, v in c['top']]}")
        A(f"%% sign check (A7 mean IF-SHAP, should be >0): {T['shap_sign_check']}")
    open(os.path.join(out, 'latex_tables.tex'), 'w').write("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results'); ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--main', default='unsup', choices=['unsup', 'semi']); ap.add_argument('--stages', default='core,robust,cold,update,shap'); ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = SEEDS_ALL[:a.seeds]; st = a.stages.split(','); T = {'main': a.main}
    t0 = time.time()
    if 'core' in st: stage_core(a.out, seeds, a.workers, T)
    if 'robust' in st: stage_robust(a.out, seeds, a.workers, T)
    if 'cold' in st: stage_cold(a.out, seeds, a.workers, T)
    if 'update' in st: stage_update(a.out, seeds, a.workers, T)
    if 'shap' in st and 'core' in st: stage_shap(a.out, T)
    T.pop('_runs_main', None)
    write_latex(a.out, T)
    json.dump(T, open(os.path.join(a.out, 'summary.json'), 'w'), default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o), indent=1)
    print(f"\nDONE in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
