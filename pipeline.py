"""
BehaCust: Behavioral Anomaly Detection for Institutional Digital Asset Custody
==============================================================================
Synthetic dataset generation + Isolation Forest + Autoencoder ensemble + SHAP
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
import shap
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

COLORS = {
    'normal': '#378ADD',
    'anomaly': '#E24B4A',
    'low': '#639922',
    'medium': '#BA7517',
    'high': '#E24B4A',
    'grid': '#e8e8e0',
    'bg': '#ffffff',
}

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────

OPERATORS = [f'op_{i:02d}' for i in range(1, 12)]
COUNTRIES_NORMAL = ['DE', 'GB', 'CH', 'NL', 'FR']
COUNTRIES_ANOMALY = ['RU', 'KP', 'IR', 'CN', 'BY']
DEVICES_NORMAL = [f'device_{i}' for i in range(1, 8)]
DEVICES_NEW = [f'new_device_{i}' for i in range(1, 6)]

def generate_normal_transactions(n=1800):
    records = []
    for _ in range(n):
        op = np.random.choice(OPERATORS)
        hour = int(np.clip(np.random.normal(13, 3), 7, 20))
        amount = np.random.lognormal(mean=8.5, sigma=1.2)
        amount = np.clip(amount, 500, 500_000)
        signer_count = np.random.choice([3, 4, 5], p=[0.5, 0.35, 0.15])
        approval_time_sec = abs(np.random.normal(120, 40))
        device = np.random.choice(DEVICES_NORMAL)
        country = np.random.choice(COUNTRIES_NORMAL)
        ip_reputation = np.random.uniform(0.7, 1.0)
        prev_tx_gap_hours = abs(np.random.normal(6, 2))

        records.append({
            'operator': op,
            'hour': hour,
            'amount_usd': round(amount, 2),
            'signer_count': signer_count,
            'approval_time_sec': round(approval_time_sec, 1),
            'device_known': 1,
            'country': country,
            'ip_reputation': round(ip_reputation, 3),
            'prev_tx_gap_hours': round(prev_tx_gap_hours, 2),
            'weekend': int(np.random.random() < 0.28),
            'label': 0,
            'anomaly_type': 'normal',
        })
    return records

def generate_anomalies(n=200):
    records = []
    threat_types = [
        'insider_threat',
        'compromised_admin',
        'unusual_time',
        'abnormal_amount',
        'rapid_approval',
        'new_device',
        'geo_anomaly',
        'collusion',
    ]
    per_type = n // len(threat_types)

    for threat in threat_types:
        for _ in range(per_type):
            op = np.random.choice(OPERATORS)
            base = {
                'operator': op,
                'hour': int(np.clip(np.random.normal(13, 3), 7, 20)),
                'amount_usd': round(np.clip(np.random.lognormal(8.5, 1.2), 500, 500_000), 2),
                'signer_count': np.random.choice([3, 4, 5], p=[0.5, 0.35, 0.15]),
                'approval_time_sec': round(abs(np.random.normal(120, 40)), 1),
                'device_known': 1,
                'country': np.random.choice(COUNTRIES_NORMAL),
                'ip_reputation': round(np.random.uniform(0.7, 1.0), 3),
                'prev_tx_gap_hours': round(abs(np.random.normal(6, 2)), 2),
                'weekend': int(np.random.random() < 0.28),
                'label': 1,
                'anomaly_type': threat,
            }

            if threat == 'insider_threat':
                base['hour'] = np.random.choice([1, 2, 3, 4, 22, 23])
                base['amount_usd'] = round(np.random.uniform(800_000, 5_000_000), 2)
                base['device_known'] = 0

            elif threat == 'compromised_admin':
                base['signer_count'] = 1
                base['approval_time_sec'] = round(np.random.uniform(1, 8), 1)
                base['amount_usd'] = round(np.random.uniform(500_000, 3_000_000), 2)

            elif threat == 'unusual_time':
                base['hour'] = np.random.choice([0, 1, 2, 3, 4, 23])
                base['weekend'] = 1

            elif threat == 'abnormal_amount':
                base['amount_usd'] = round(np.random.uniform(2_000_000, 10_000_000), 2)

            elif threat == 'rapid_approval':
                base['approval_time_sec'] = round(np.random.uniform(1, 5), 1)
                base['signer_count'] = 5

            elif threat == 'new_device':
                base['device_known'] = 0
                base['amount_usd'] = round(np.random.uniform(300_000, 2_000_000), 2)

            elif threat == 'geo_anomaly':
                base['country'] = np.random.choice(COUNTRIES_ANOMALY)
                base['ip_reputation'] = round(np.random.uniform(0.0, 0.3), 3)

            elif threat == 'collusion':
                base['approval_time_sec'] = round(np.random.uniform(2, 10), 1)
                base['signer_count'] = np.random.choice([4, 5])
                base['prev_tx_gap_hours'] = round(np.random.uniform(0.01, 0.5), 3)

            records.append(base)
    return records

print("Generating synthetic custody dataset...")
normal_recs = generate_normal_transactions(1800)
anomaly_recs = generate_anomalies(200)
all_recs = normal_recs + anomaly_recs
np.random.shuffle(all_recs)

df = pd.DataFrame(all_recs).reset_index(drop=True)
df['tx_id'] = [f'TX{str(i).zfill(4)}' for i in range(len(df))]

print(f"Dataset: {len(df)} transactions | Normal: {(df.label==0).sum()} | Anomaly: {(df.label==1).sum()}")
df.to_csv('/home/claude/custody_anomaly/custody_dataset.csv', index=False)
print("Saved: custody_dataset.csv")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

print("\nFeature engineering...")

high_risk_countries = set(COUNTRIES_ANOMALY)
df['country_risk'] = df['country'].apply(lambda c: 1 if c in high_risk_countries else 0)
df['off_hours'] = df['hour'].apply(lambda h: 1 if h < 7 or h > 20 else 0)
df['log_amount'] = np.log1p(df['amount_usd'])
df['low_signer_flag'] = (df['signer_count'] < 3).astype(int)
df['rapid_approval'] = (df['approval_time_sec'] < 10).astype(int)
df['low_ip_rep'] = (df['ip_reputation'] < 0.4).astype(int)
df['high_freq'] = (df['prev_tx_gap_hours'] < 0.5).astype(int)

op_stats = df[df['label'] == 0].groupby('operator').agg(
    op_mean_amount=('amount_usd', 'mean'),
    op_std_amount=('amount_usd', 'std'),
    op_mean_hour=('hour', 'mean'),
    op_mean_latency=('approval_time_sec', 'mean'),
).reset_index()
op_stats['op_std_amount'] = op_stats['op_std_amount'].fillna(1)

df = df.merge(op_stats, on='operator', how='left')
df['amount_zscore'] = (df['amount_usd'] - df['op_mean_amount']) / df['op_std_amount']
df['hour_dev'] = abs(df['hour'] - df['op_mean_hour'])
df['latency_ratio'] = df['approval_time_sec'] / (df['op_mean_latency'] + 1e-9)

FEATURES = [
    'log_amount', 'amount_zscore', 'hour', 'off_hours', 'weekend',
    'signer_count', 'low_signer_flag', 'approval_time_sec', 'rapid_approval',
    'device_known', 'ip_reputation', 'low_ip_rep', 'country_risk',
    'prev_tx_gap_hours', 'high_freq', 'hour_dev', 'latency_ratio',
]

X = df[FEATURES].fillna(0).values
y = df['label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────
# 3. MODELS
# ─────────────────────────────────────────────

print("Training Isolation Forest...")
iso = IsolationForest(n_estimators=200, contamination=0.10, random_state=42)
iso.fit(X_scaled)
iso_scores = -iso.score_samples(X_scaled)
iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())

print("Training One-Class SVM...")
ocsvm = OneClassSVM(kernel='rbf', nu=0.10, gamma='scale')
ocsvm.fit(X_scaled)
svm_scores = -ocsvm.score_samples(X_scaled)
svm_scores = (svm_scores - svm_scores.min()) / (svm_scores.max() - svm_scores.min())

print("Training Autoencoder (numpy)...")
def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def mse(a, b): return np.mean((a - b) ** 2, axis=1)

n_feat = X_scaled.shape[1]
rng = np.random.default_rng(42)
lr = 0.005
epochs = 120
batch_size = 64

W1 = rng.normal(0, 0.1, (n_feat, 12))
b1 = np.zeros(12)
W2 = rng.normal(0, 0.1, (12, 6))
b2 = np.zeros(6)
W3 = rng.normal(0, 0.1, (6, 12))
b3 = np.zeros(12)
W4 = rng.normal(0, 0.1, (12, n_feat))
b4 = np.zeros(n_feat)

X_normal = X_scaled[y == 0]
ae_losses = []

for epoch in range(epochs):
    idx = rng.permutation(len(X_normal))
    epoch_loss = 0
    for start in range(0, len(X_normal), batch_size):
        batch = X_normal[idx[start:start+batch_size]]
        h1 = relu(batch @ W1 + b1)
        h2 = relu(h1 @ W2 + b2)
        h3 = relu(h2 @ W3 + b3)
        out = h3 @ W4 + b4
        loss = np.mean((batch - out) ** 2)
        epoch_loss += loss

        d_out = 2 * (out - batch) / len(batch)
        dW4 = h3.T @ d_out; db4 = d_out.sum(0)
        d_h3 = d_out @ W4.T
        d_h3 *= (h3 > 0)
        dW3 = h2.T @ d_h3; db3 = d_h3.sum(0)
        d_h2 = d_h3 @ W3.T
        d_h2 *= (h2 > 0)
        dW2 = h1.T @ d_h2; db2 = d_h2.sum(0)
        d_h1 = d_h2 @ W2.T
        d_h1 *= (h1 > 0)
        dW1 = batch.T @ d_h1; db1 = d_h1.sum(0)

        for W, dW in [(W1,dW1),(W2,dW2),(W3,dW3),(W4,dW4)]:
            W -= lr * dW
        for b, db in [(b1,db1),(b2,db2),(b3,db3),(b4,db4)]:
            b -= lr * db

    ae_losses.append(epoch_loss)

def ae_score(X_in):
    h1 = relu(X_in @ W1 + b1)
    h2 = relu(h1 @ W2 + b2)
    h3 = relu(h2 @ W3 + b3)
    out = h3 @ W4 + b4
    return mse(X_in, out)

ae_raw = ae_score(X_scaled)
ae_scores = (ae_raw - ae_raw.min()) / (ae_raw.max() - ae_raw.min())

ensemble_scores = 0.45 * iso_scores + 0.35 * ae_scores + 0.20 * svm_scores

df['risk_score'] = ensemble_scores
df['risk_level'] = pd.cut(
    ensemble_scores,
    bins=[0, 0.35, 0.70, 1.0],
    labels=['low', 'medium', 'high'],
    include_lowest=True,
)

auc_iso = roc_auc_score(y, iso_scores)
auc_ae  = roc_auc_score(y, ae_scores)
auc_svm = roc_auc_score(y, svm_scores)
auc_ens = roc_auc_score(y, ensemble_scores)
ap_ens  = average_precision_score(y, ensemble_scores)

print(f"\nAUROC  →  IF: {auc_iso:.3f}  AE: {auc_ae:.3f}  SVM: {auc_svm:.3f}  Ensemble: {auc_ens:.3f}")
print(f"Avg Precision (Ensemble): {ap_ens:.3f}")

# ─────────────────────────────────────────────
# 4. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────

print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(iso)
shap_vals = explainer.shap_values(X_scaled)

# ─────────────────────────────────────────────
# 5. FIGURES
# ─────────────────────────────────────────────

print("Generating figures...")

# ── Fig 1: ROC curves ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for scores, label, color, lw in [
    (iso_scores, f'Isolation Forest  (AUC={auc_iso:.3f})', '#378ADD', 1.5),
    (ae_scores,  f'Autoencoder        (AUC={auc_ae:.3f})',  '#639922', 1.5),
    (svm_scores, f'One-Class SVM      (AUC={auc_svm:.3f})', '#BA7517', 1.5),
    (ensemble_scores, f'Ensemble           (AUC={auc_ens:.3f})', '#E24B4A', 2.2),
]:
    fpr, tpr, _ = roc_curve(y, scores)
    ax.plot(fpr, tpr, label=label, color=color, lw=lw)
ax.plot([0,1],[0,1], '--', color='#aaa', lw=1)
ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
ax.set_title('ROC curves — model comparison', fontweight='500')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, color=COLORS['grid'], linewidth=0.5)

ax = axes[1]
prec, rec, _ = precision_recall_curve(y, ensemble_scores)
ax.plot(rec, prec, color='#E24B4A', lw=2, label=f'Ensemble  (AP={ap_ens:.3f})')
ax.axhline(y.mean(), color='#aaa', ls='--', lw=1, label='Baseline (random)')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision–recall curve (ensemble)', fontweight='500')
ax.legend(fontsize=9)
ax.grid(True, color=COLORS['grid'], linewidth=0.5)

plt.tight_layout()
plt.savefig('/home/claude/custody_anomaly/fig1_roc_pr.png', bbox_inches='tight', dpi=150)
plt.close()

# ── Fig 2: SHAP summary ──
fig, ax = plt.subplots(figsize=(10, 7))
feat_names_display = [
    'Log amount', 'Amount z-score', 'Hour of day', 'Off-hours flag', 'Weekend',
    'Signer count', 'Low signer flag', 'Approval time (s)', 'Rapid approval',
    'Device known', 'IP reputation', 'Low IP rep', 'Country risk',
    'Prev tx gap (h)', 'High freq flag', 'Hour deviation', 'Latency ratio',
]
mean_abs_shap = np.abs(shap_vals).mean(0)
order = np.argsort(mean_abs_shap)
colors_bar = ['#E24B4A' if mean_abs_shap[i] > np.percentile(mean_abs_shap, 70) else '#378ADD' for i in order]
bars = ax.barh(
    [feat_names_display[i] for i in order],
    mean_abs_shap[order],
    color=colors_bar, height=0.7, edgecolor='none',
)
ax.set_xlabel('Mean |SHAP value|  (impact on anomaly score)')
ax.set_title('Feature importance — SHAP (Isolation Forest)', fontweight='500')
ax.grid(True, axis='x', color=COLORS['grid'], linewidth=0.5)
patch_high = mpatches.Patch(color='#E24B4A', label='Top-3 features')
patch_rest = mpatches.Patch(color='#378ADD', label='Other features')
ax.legend(handles=[patch_high, patch_rest], fontsize=9)
plt.tight_layout()
plt.savefig('/home/claude/custody_anomaly/fig2_shap_summary.png', bbox_inches='tight', dpi=150)
plt.close()

# ── Fig 3: Risk score distribution ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.hist(ensemble_scores[y==0], bins=50, alpha=0.7, color=COLORS['normal'], label='Normal', density=True)
ax.hist(ensemble_scores[y==1], bins=50, alpha=0.7, color=COLORS['anomaly'], label='Anomaly', density=True)
ax.axvline(0.35, color='#BA7517', ls='--', lw=1.2, label='Low/Medium threshold')
ax.axvline(0.70, color='#E24B4A', ls='--', lw=1.2, label='Medium/High threshold')
ax.set_xlabel('Ensemble risk score'); ax.set_ylabel('Density')
ax.set_title('Risk score distribution', fontweight='500')
ax.legend(fontsize=9)
ax.grid(True, color=COLORS['grid'], linewidth=0.5)

ax = axes[1]
threat_types = df[df['label']==1]['anomaly_type'].value_counts()
colors_threat = ['#E24B4A','#D85A30','#BA7517','#854F0B','#639922','#0F6E56','#185FA5','#534AB7']
bars = ax.barh(threat_types.index, threat_types.values, color=colors_threat[:len(threat_types)], height=0.7, edgecolor='none')
ax.set_xlabel('Count'); ax.set_title('Anomaly types in dataset', fontweight='500')
ax.grid(True, axis='x', color=COLORS['grid'], linewidth=0.5)
plt.tight_layout()
plt.savefig('/home/claude/custody_anomaly/fig3_risk_dist.png', bbox_inches='tight', dpi=150)
plt.close()

# ── Fig 4: Case study — two transactions ──
high_risk_cases = df[(df['label']==1) & (df['risk_level']=='high')].head(2)
low_risk_cases  = df[(df['label']==0) & (df['risk_level']=='low')].head(2)
case_rows = pd.concat([high_risk_cases, low_risk_cases])

case_idx = [df.index.get_loc(i) for i in case_rows.index]
case_shap = shap_vals[case_idx]

fig, axes = plt.subplots(1, min(4, len(case_idx)), figsize=(14, 5))
if len(case_idx) == 1: axes = [axes]

for ax_i, (idx_in_case, orig_idx) in enumerate(zip(range(len(case_idx)), case_rows.index)):
    row = df.loc[orig_idx]
    sv = case_shap[ax_i]
    order = np.argsort(np.abs(sv))[-8:]
    colors_case = ['#E24B4A' if sv[j] > 0 else '#378ADD' for j in order]
    axes[ax_i].barh(
        [feat_names_display[j] for j in order],
        sv[order],
        color=colors_case, height=0.6, edgecolor='none',
    )
    axes[ax_i].axvline(0, color='#aaa', lw=0.8)
    label_str = 'ANOMALY' if row['label']==1 else 'NORMAL'
    type_str = row['anomaly_type']
    axes[ax_i].set_title(f'{label_str}\n{type_str}\nScore: {row["risk_score"]:.2f}', fontsize=9, fontweight='500')
    axes[ax_i].grid(True, axis='x', color=COLORS['grid'], linewidth=0.5)
    axes[ax_i].tick_params(axis='y', labelsize=8)

plt.suptitle('SHAP waterfall — individual transaction explanations', fontweight='500', y=1.02)
plt.tight_layout()
plt.savefig('/home/claude/custody_anomaly/fig4_shap_cases.png', bbox_inches='tight', dpi=150)
plt.close()

# ── Fig 5: Autoencoder training loss ──
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ae_losses, color='#378ADD', lw=1.5)
ax.set_xlabel('Epoch'); ax.set_ylabel('Reconstruction loss (MSE)')
ax.set_title('Autoencoder training — convergence on normal transactions', fontweight='500')
ax.grid(True, color=COLORS['grid'], linewidth=0.5)
plt.tight_layout()
plt.savefig('/home/claude/custody_anomaly/fig5_ae_loss.png', bbox_inches='tight', dpi=150)
plt.close()

# ── Summary stats table ──
summary = pd.DataFrame({
    'Model': ['Isolation Forest', 'Autoencoder', 'One-Class SVM', 'Ensemble (IF+AE+SVM)'],
    'AUROC': [round(auc_iso,4), round(auc_ae,4), round(auc_svm,4), round(auc_ens,4)],
    'Avg Precision': [
        round(average_precision_score(y, iso_scores),4),
        round(average_precision_score(y, ae_scores),4),
        round(average_precision_score(y, svm_scores),4),
        round(ap_ens,4),
    ],
    'Weight in ensemble': ['45%', '35%', '20%', '—'],
})
summary.to_csv('/home/claude/custody_anomaly/results_summary.csv', index=False)
print("\nResults summary:")
print(summary.to_string(index=False))

print("\nAll figures saved to /home/claude/custody_anomaly/")
print("Done.")
