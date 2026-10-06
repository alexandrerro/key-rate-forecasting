
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
import os
import warnings
warnings.filterwarnings('ignore')


from paths import FIG_DIR
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.figsize': (14, 6), 'figure.dpi': 120,
    'savefig.dpi': 150, 'savefig.bbox': 'tight',
})



print("=" * 60)
print("БЛОК 1: СБОР ПРОГНОЗОВ ВСЕХ МОДЕЛЕЙ")
print("=" * 60)

prediction_files = {
    "baseline": f"{FIG_DIR}/baseline_all_predictions.csv",
    "statsforecast": f"{FIG_DIR}/sf_all_predictions.csv",
    "prophet": f"{FIG_DIR}/prophet_all_predictions.csv",
    "ml": f"{FIG_DIR}/ml_all_predictions.csv",
    "nn": f"{FIG_DIR}/nn_all_predictions.csv",
    "foundation": f"{FIG_DIR}/fm_all_predictions.csv",
}

all_preds = {} 

for source, path in prediction_files.items():
    if not os.path.exists(path):
        print(f"  [!] {source}: файл не найден ({path})")
        continue

    df = pd.read_csv(path, parse_dates=["ds"] if "ds" in pd.read_csv(path, nrows=0).columns else [])

    date_col = "ds" if "ds" in df.columns else "date"
    true_col = "y_true" if "y_true" in df.columns else "y"

    if date_col not in df.columns:
        print(f"  [!] {source}: нет столбца даты")
        continue

    df[date_col] = pd.to_datetime(df[date_col])

    pred_cols = [c for c in df.columns if c.startswith("pred_") or
                 c in ["TimeGPT", "Chronos-Bolt", "NBEATS", "NHITS", "PatchTST"]]

    for col in pred_cols:
        if col.startswith("pred_"):
            model_name = col.replace("pred_", "")
        else:
            model_name = col

        model_df = pd.DataFrame({
            "ds": df[date_col].values,
            "y_true": df[true_col].values,
            "pred": df[col].values,
        }).dropna()

        all_preds[model_name] = model_df
        print(f"  {source:15s} → {model_name:17s}: {len(model_df)} прогнозов")

print(f"\nВсего моделей загружено: {len(all_preds)}")


print("\n" + "=" * 60)
print("БЛОК 2: МЕТРИКИ ВСЕХ МОДЕЛЕЙ")
print("=" * 60)

def calc_metrics(y_true, y_pred, name):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return None
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    mape = np.mean(np.abs((yt - yp) / yt)) * 100 if np.all(yt != 0) else np.nan
    return {"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE_%": round(mape, 2), "n_forecasts": int(mask.sum())}

metrics_list = []
for model_name, df_pred in all_preds.items():
    m = calc_metrics(df_pred["y_true"].values, df_pred["pred"].values, model_name)
    if m:
        metrics_list.append(m)

metrics_df = pd.DataFrame(metrics_list).sort_values("MAE").reset_index(drop=True)

print(f"\n  СВОДНАЯ ТАБЛИЦА ({len(metrics_df)} моделей, h=1):")
print("  " + "=" * 70)
print(f"  {'#':>3s} {'Модель':20s} | {'MAE':>8s} | {'RMSE':>8s} | {'MAPE %':>8s} | {'N':>3s}")
print("  " + "-" * 70)
for i, row in metrics_df.iterrows():
    print(f"  {i+1:3d} {row['model']:20s} | {row['MAE']:8.4f} | {row['RMSE']:8.4f} | "
          f"{row['MAPE_%']:7.2f}% | {int(row['n_forecasts']):3d}")

metrics_df.to_csv(f"{FIG_DIR}/metrics_final_all.csv", index=False)
print(f"\n  [OK] metrics_final_all.csv")


print("\n" + "=" * 60)
print("БЛОК 3: ТЕСТ DIEBOLD-MARIANO")
print("=" * 60)

def diebold_mariano_test(y_true, pred_1, pred_2, loss='absolute'):
    e1 = y_true - pred_1
    e2 = y_true - pred_2

    if loss == 'absolute':
        d = np.abs(e1) - np.abs(e2)
    elif loss == 'squared':
        d = e1**2 - e2**2
    else:
        raise ValueError("loss must be 'absolute' or 'squared'")

    T = len(d)
    d_mean = np.mean(d)
    K = max(1, int(np.floor(T ** (1/3))))

    gamma = np.zeros(K + 1)
    for k in range(K + 1):
        gamma[k] = np.mean((d[k:] - d_mean) * (d[:T-k] - d_mean))

    V = gamma[0] + 2 * np.sum([(1 - k/(K+1)) * gamma[k] for k in range(1, K+1)])
    V = V / T

    if V <= 0:
        return np.nan, np.nan

    dm_stat = d_mean / np.sqrt(V)
    p_value = 2 * (1 - sp_stats.norm.cdf(np.abs(dm_stat)))

    return dm_stat, p_value

key_models = [
    "Naive", "AutoTheta", "Drift", "AutoETS", "AutoARIMA",
    "ARIMA", "TimeGPT", "Prophet_exog", "Prophet-exog",
    "LightGBM", "XGBoost",
]

alt_names = {
    "Prophet_exog": "Prophet-exog",
    "Prophet_base": "Prophet-base",
}

available_models = []
for m in key_models:
    name = alt_names.get(m, m)
    if name in all_preds:
        available_models.append(name)
    elif m in all_preds:
        available_models.append(m)

for m in ["NBEATS", "NHITS", "PatchTST", "TimeGPT", "Chronos-Bolt"]:
    if m in all_preds and m not in available_models:
        available_models.append(m)

seen = set()
available_models = [m for m in available_models if not (m in seen or seen.add(m))]

print(f"\n  Модели для DM-теста ({len(available_models)}):")
for m in available_models:
    print(f"    {m} ({len(all_preds[m])} прогнозов)")

dm_results = []
n_models = len(available_models)

print(f"\n  Попарные DM-тесты ({n_models * (n_models - 1) // 2} пар)...")

dm_matrix = pd.DataFrame(np.nan, index=available_models, columns=available_models)
pval_matrix = pd.DataFrame(np.nan, index=available_models, columns=available_models)

for i in range(n_models):
    for j in range(i + 1, n_models):
        m1_name = available_models[i]
        m2_name = available_models[j]

        df1 = all_preds[m1_name]
        df2 = all_preds[m2_name]

        merged = df1.merge(df2, on="ds", suffixes=("_1", "_2"))

        if len(merged) < 10:
            continue

        y_true = merged["y_true_1"].values
        pred_1 = merged["pred_1"].values
        pred_2 = merged["pred_2"].values

        dm_stat, p_value = diebold_mariano_test(y_true, pred_1, pred_2, loss='absolute')

        dm_matrix.loc[m1_name, m2_name] = dm_stat
        dm_matrix.loc[m2_name, m1_name] = -dm_stat
        pval_matrix.loc[m1_name, m2_name] = p_value
        pval_matrix.loc[m2_name, m1_name] = p_value

        if p_value < 0.01:
            sig = "***"
        elif p_value < 0.05:
            sig = "**"
        elif p_value < 0.10:
            sig = "*"
        else:
            sig = ""

        better = m1_name if dm_stat < 0 else m2_name

        dm_results.append({
            "model_1": m1_name,
            "model_2": m2_name,
            "DM_stat": round(dm_stat, 3),
            "p_value": round(p_value, 4),
            "significance": sig,
            "better": better,
            "n_common": len(merged),
        })

dm_df = pd.DataFrame(dm_results)

print(f"\n  КЛЮЧЕВЫЕ СРАВНЕНИЯ (тест Diebold-Mariano):")
print(f"  {'Модель 1':17s} vs {'Модель 2':17s} | {'DM':>7s} | {'p-value':>8s} | {'Знач.':>5s} | {'Лучше'}")
print(f"  {'-'*75}")

for _, row in dm_df.iterrows():
    print(f"  {row['model_1']:17s} vs {row['model_2']:17s} | {row['DM_stat']:+7.3f} | "
          f"{row['p_value']:8.4f} | {row['significance']:>5s} | {row['better']}")

dm_df.to_csv(f"{FIG_DIR}/dm_test_results.csv", index=False)
pval_matrix.to_csv(f"{FIG_DIR}/dm_pvalue_matrix.csv")
print(f"\n  [OK] dm_test_results.csv")
print(f"  [OK] dm_pvalue_matrix.csv")



print("\n" + "=" * 60)
print("БЛОК 4: ВИЗУАЛИЗАЦИЯ")
print("=" * 60)

# Матрица p-values DM-теста 
fig, ax = plt.subplots(figsize=(12, 10))

pval_data = pval_matrix.values.astype(float)
mask = np.isnan(pval_data)

im = ax.imshow(pval_data, cmap='RdYlGn', vmin=0, vmax=0.15, aspect='auto')

for i in range(len(available_models)):
    for j in range(len(available_models)):
        if i == j:
            ax.text(j, i, "—", ha='center', va='center', fontsize=9)
        elif not np.isnan(pval_data[i, j]):
            val = pval_data[i, j]
            sig = "***" if val < 0.01 else "**" if val < 0.05 else "*" if val < 0.10 else ""
            color = 'white' if val < 0.05 else 'black'
            ax.text(j, i, f'{val:.3f}{sig}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold' if val < 0.05 else 'normal')

ax.set_xticks(range(len(available_models)))
ax.set_yticks(range(len(available_models)))
ax.set_xticklabels(available_models, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(available_models, fontsize=9)

plt.colorbar(im, ax=ax, label='p-value (зелёный = различия НЕ значимы)')
ax.set_title('Тест Diebold-Mariano: матрица p-values\n'
             '(*** p<0.01, ** p<0.05, * p<0.10)')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/comparison_01_dm_matrix.png')
plt.close()
print("  [OK] comparison_01_dm_matrix.png")

model_classes = {
    "Naive": "Baseline", "Drift": "Baseline", "ARIMA": "Baseline", "Decomp+Trend": "Baseline",
    "AutoARIMA": "StatsForecast", "AutoETS": "StatsForecast",
    "CES": "StatsForecast", "AutoTheta": "StatsForecast",
    "Prophet-base": "Prophet", "Prophet-exog": "Prophet", "Prophet_base": "Prophet", "Prophet_exog": "Prophet",
    "LightGBM": "ML (бустинг)", "XGBoost": "ML (бустинг)",
    "NBEATS": "NeuralForecast", "NHITS": "NeuralForecast", "PatchTST": "NeuralForecast",
    "TimeGPT": "Foundation", "Chronos-Bolt": "Foundation", "TimesFM": "Foundation",
}

class_colors = {
    "Baseline": "#bdc3c7",
    "StatsForecast": "#3498db",
    "Prophet": "#e67e22",
    "ML (бустинг)": "#2ecc71",
    "NeuralForecast": "#e74c3c",
    "Foundation": "#9b59b6",
}

fig, ax = plt.subplots(figsize=(12, max(8, len(metrics_df) * 0.5)))

sorted_metrics = metrics_df.sort_values("MAE")
models = sorted_metrics["model"].values
maes = sorted_metrics["MAE"].values

bar_colors = []
for m in models:
    cls = model_classes.get(m, "Другое")
    bar_colors.append(class_colors.get(cls, "#888888"))

bars = ax.barh(range(len(models)), maes, color=bar_colors, alpha=0.85)

ax.set_yticks(range(len(models)))
ax.set_yticklabels(models)
ax.invert_yaxis()
ax.set_xlabel("MAE (п.п.)")
ax.set_title("Ranking всех моделей по MAE (h=1)")
ax.grid(True, alpha=0.3, axis='x')

for bar, val in zip(bars, maes):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=cls) for cls, c in class_colors.items()
                   if any(model_classes.get(m) == cls for m in models)]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/comparison_02_ranking.png')
plt.close()
print("  [OK] comparison_02_ranking.png")


fig, ax = plt.subplots(figsize=(10, 8))

for _, row in metrics_df.iterrows():
    cls = model_classes.get(row['model'], "Другое")
    color = class_colors.get(cls, "#888888")
    ax.scatter(row['MAE'], row['RMSE'], c=color, s=100, zorder=5, alpha=0.8)
    ax.annotate(row['model'], (row['MAE'], row['RMSE']),
                textcoords="offset points", xytext=(8, 5), fontsize=8)

ax.set_xlabel("MAE (п.п.)")
ax.set_ylabel("RMSE (п.п.)")
ax.set_title("MAE vs RMSE: trade-off между средней и экстремальной ошибкой")
ax.grid(True, alpha=0.3)

legend_elements = [Patch(facecolor=c, label=cls) for cls, c in class_colors.items()]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/comparison_03_mae_vs_rmse.png')
plt.close()
print("  [OK] comparison_03_mae_vs_rmse.png")


print("\n" + "=" * 60)
print("БЛОК 5: ВЫЧИСЛИТЕЛЬНАЯ СЛОЖНОСТЬ")
print("=" * 60)

complexity = [
    {"model": "Naive",       "time_s": 0,    "params": 0,        "complexity": "O(1)"},
    {"model": "Drift",       "time_s": 0,    "params": 2,        "complexity": "O(n)"},
    {"model": "AutoARIMA",   "time_s": 55,   "params": 5,        "complexity": "O(n·p·q)"},
    {"model": "AutoETS",     "time_s": 55,   "params": 10,       "complexity": "O(n·m)"},
    {"model": "AutoTheta",   "time_s": 55,   "params": 3,        "complexity": "O(n)"},
    {"model": "CES",         "time_s": 55,   "params": 4,        "complexity": "O(n)"},
    {"model": "Prophet",     "time_s": 16,   "params": 50,       "complexity": "O(n·k)"},
    {"model": "LightGBM",    "time_s": 5,    "params": 3000,     "complexity": "O(n·d·T)"},
    {"model": "XGBoost",     "time_s": 5,    "params": 3000,     "complexity": "O(n·d·T·log(n))"},
    {"model": "N-BEATS",     "time_s": 171,  "params": 2400000,  "complexity": "O(n·L·H)"},
    {"model": "N-HiTS",      "time_s": 171,  "params": 2400000,  "complexity": "O(n·L·H/Πk)"},
    {"model": "PatchTST",    "time_s": 171,  "params": 42600,    "complexity": "O(n·P²·d)"},
    {"model": "TimeGPT",     "time_s": 84,   "params": 0,        "complexity": "O(API call)"},
]

# ручные замеры для одной своей машины

comp_df = pd.DataFrame(complexity)

print(f"\n  {'Модель':15s} | {'Время (с)':>10s} | {'Параметры':>12s} | {'Сложность'}")
print(f"  {'-'*60}")
for _, row in comp_df.iterrows():
    params_str = f"{row['params']:,}" if row['params'] > 0 else "—"
    print(f"  {row['model']:15s} | {row['time_s']:>8d}s | {params_str:>12s} | {row['complexity']}")

comp_df.to_csv(f"{FIG_DIR}/computational_complexity.csv", index=False)
print(f"\n  [OK] computational_complexity.csv")



print("\n" + "=" * 60)
print("ИТОГОВАЯ СВОДКА")
print("=" * 60)

best = metrics_df.iloc[0]
n_total = len(metrics_df)

n_significant = 0
n_not_significant = 0
if len(dm_df) > 0:
    n_significant = (dm_df['p_value'] < 0.05).sum()
    n_not_significant = (dm_df['p_value'] >= 0.05).sum()

naive_vs_theta = dm_df[(dm_df['model_1'] == 'Naive') & (dm_df['model_2'] == 'AutoTheta') |
                        (dm_df['model_1'] == 'AutoTheta') & (dm_df['model_2'] == 'Naive')]

print(f"""
ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ РАБОТЫ

Сравнено {n_total} моделей четырёх классов:
  • Baseline:       Naive, Drift, ARIMA, Decomp+Trend
  • StatsForecast:  AutoARIMA, AutoETS, CES, AutoTheta
  • Prophet:        base (без регрессоров), exog (с регрессорами)
  • ML:             LightGBM, XGBoost
  • NeuralForecast: N-BEATS, N-HiTS, PatchTST
  • Foundation:     TimeGPT (zero-shot)

ЛУЧШАЯ МОДЕЛЬ: {best['model']} (MAE = {best['MAE']:.4f} п.п.)

ТЕСТ DIEBOLD-MARIANO:
  Пар сравнений: {len(dm_df)}
  Значимые различия (p < 0.05): {n_significant}
  Незначимые различия (p ≥ 0.05): {n_not_significant}
""")

if len(naive_vs_theta) > 0:
    row = naive_vs_theta.iloc[0]
    print(f"  Naive vs AutoTheta: DM = {row['DM_stat']:+.3f}, p = {row['p_value']:.4f}")
    if row['p_value'] >= 0.05:
        print(f"  → Различие НЕ значимо: Naive и AutoTheta статистически эквивалентны")
    else:
        print(f"  → Различие значимо: {row['better']} достоверно лучше")

print("Все сохранённые файлы:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith('comparison_') or f.startswith('dm_') or f == 'metrics_final_all.csv' or f == 'computational_complexity.csv':
        path = os.path.join(FIG_DIR, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {path} ({size_kb:.0f} KB)")
