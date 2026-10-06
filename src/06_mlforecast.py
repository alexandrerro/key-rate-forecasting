import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import time
import warnings
from scipy import stats
from sklearn.inspection import permutation_importance as sklearn_perm_importance

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    print("[OK] LightGBM")
except ImportError:
    print("[!] LightGBM не установлен: pip install lightgbm")
    exit(1)

try:
    import xgboost as xgb
    print("[OK] XGBoost")
except ImportError:
    print("[!] XGBoost не установлен: pip install xgboost")
    xgb = None

try:
    import shap
    HAS_SHAP = True
    print("[OK] SHAP")
except ImportError:
    HAS_SHAP = False
    print("[!] SHAP не установлен (pip install shap). Feature importance будет без SHAP.")

from paths import DATA_DIR, FIG_DIR
DATA_PATH = DATA_DIR / "dataset_monthly.csv"
os.makedirs(FIG_DIR, exist_ok=True)

MIN_TRAIN_SIZE = 60
HORIZON = 1

plt.rcParams.update({
    'figure.figsize': (14, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


print("\n" + "=" * 60)
print("БЛОК 1: FEATURE ENGINEERING")
print("=" * 60)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

TARGET = "key_rate"

EXOG_VARS = ["cpi_mom", "m2", "usd_rub", "brent",
             "zcyc_1y", "zcyc_5y", "zcyc_10y",
             "spread_10y_1y", "spread_5y_1y", 
             "ruonia",             
             "spread_ruonia_keyrate", 
             ]
EXOG_VARS = [v for v in EXOG_VARS if v in df.columns]

print(f"Целевая переменная: {TARGET}")
print(f"Экзогенные переменные: {len(EXOG_VARS)}")
LAGS = [1, 2, 3, 6, 12]
for lag in LAGS:
    df[f"lag_{lag}"] = df[TARGET].shift(lag)

WINDOWS = [3, 6, 12]
for w in WINDOWS:
    df[f"rolling_mean_{w}"] = df[TARGET].shift(1).rolling(window=w, min_periods=1).mean()
    df[f"rolling_std_{w}"] = df[TARGET].shift(1).rolling(window=w, min_periods=1).std()

df["diff_1"] = df[TARGET].shift(1) - df[TARGET].shift(2)
df["diff_3"] = df[TARGET].shift(1) - df[TARGET].shift(4) 
df["diff_6"] = df[TARGET].shift(1) - df[TARGET].shift(7)  

EXOG_LAGS = [1, 3]
for var in EXOG_VARS:
    for lag in EXOG_LAGS:
        df[f"{var}_lag{lag}"] = df[var].shift(lag)

df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter

feature_cols = []
for col in df.columns:
    if col in ["date", TARGET] or col in EXOG_VARS:
        continue
    if any(col.startswith(p) for p in ["lag_", "rolling_", "diff_"]):
        feature_cols.append(col)
    elif "_lag" in col:
        feature_cols.append(col)
    elif col in ["month", "quarter"]:
        feature_cols.append(col)

df_ml = df[["date", TARGET] + feature_cols].dropna().reset_index(drop=True)

print(f"\nПризнаков создано: {len(feature_cols)}")
print(f"Наблюдений после удаления NaN: {len(df_ml)}")
print(f"Период: {df_ml['date'].min():%Y-%m} — {df_ml['date'].max():%Y-%m}")
print(f"\nСписок признаков:")
for i, f in enumerate(feature_cols):
    print(f"  {i+1:2d}. {f}")

print("\n" + "=" * 60)
print("БЛОК 2: НАСТРОЙКА МОДЕЛЕЙ")
print("=" * 60)

lgb_params = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 15,          
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "min_child_samples": 5,
    "subsample": 0.8,    
    "colsample_bytree": 0.8, 
    "reg_alpha": 0.1,        
    "reg_lambda": 0.1,        
    "random_state": 42,
    "verbose": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbosity": 0,
}

print(f"LightGBM: {lgb_params['n_estimators']} деревьев, "
      f"max_depth={lgb_params['max_depth']}, lr={lgb_params['learning_rate']}")
if xgb is not None:
    print(f"XGBoost:  {xgb_params['n_estimators']} деревьев, "
          f"max_depth={xgb_params['max_depth']}, lr={xgb_params['learning_rate']}")


print("\n" + "=" * 60)
print("БЛОК 3: EXPANDING WINDOW CV")
print("=" * 60)

y_all = df_ml[TARGET].values
X_all = df_ml[feature_cols].values
dates_all = df_ml["date"].values
n = len(df_ml)

val_start = MIN_TRAIN_SIZE
val_end = n - HORIZON + 1
n_val_steps = val_end - val_start

print(f"Наблюдений (после feature engineering): {n}")
print(f"Шагов валидации: {n_val_steps}")
print(f"Первый прогноз: {pd.Timestamp(dates_all[val_start]):%Y-%m}")
print(f"Последний прогноз: {pd.Timestamp(dates_all[val_end-1]):%Y-%m}")

print(f"\n  LightGBM: expanding window ({n_val_steps} шагов)...")
start_time = time.time()

preds_lgb = []
last_lgb_model = None 

for i in range(n_val_steps):
    t = val_start + i

    X_train = X_all[:t]
    y_train = y_all[:t]
    X_test = X_all[t:t+1]
    y_test = y_all[t]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)[0]

    preds_lgb.append({
        "ds": dates_all[t],
        "y_true": y_test,
        "pred_LightGBM": pred,
    })

    last_lgb_model = model

    if (i + 1) % 30 == 0 or (i + 1) == n_val_steps:
        elapsed = time.time() - start_time
        print(f"    [{i+1}/{n_val_steps}] elapsed: {elapsed:.1f}s")

elapsed_lgb = time.time() - start_time
print(f"  LightGBM завершён за {elapsed_lgb:.1f}s")

print("\n" + "=" * 60)
print("ИНТЕРВАЛЬНЫЕ ОЦЕНКИ: LightGBM Quantile Regression")
print("=" * 60)

lgb_params_lower = lgb_params.copy()
lgb_params_lower["objective"] = "quantile"
lgb_params_lower["alpha"] = 0.025 

lgb_params_upper = lgb_params.copy()
lgb_params_upper["objective"] = "quantile"
lgb_params_upper["alpha"] = 0.975 

lgb_params_lower.pop("metric", None)
lgb_params_upper.pop("metric", None)

print(f"  Нижний квантиль: α = 0.025")
print(f"  Верхний квантиль: α = 0.975")
print(f"  Expanding window: {n_val_steps} шагов")

start_time = time.time()

preds_ci = []

for i in range(n_val_steps):
    t = val_start + i

    X_train = X_all[:t]
    y_train = y_all[:t]
    X_test = X_all[t:t+1]
    y_test = y_all[t]

    model_lo = lgb.LGBMRegressor(**lgb_params_lower)
    model_lo.fit(X_train, y_train)
    pred_lo = model_lo.predict(X_test)[0]

    model_hi = lgb.LGBMRegressor(**lgb_params_upper)
    model_hi.fit(X_train, y_train)
    pred_hi = model_hi.predict(X_test)[0]

    pred_point = preds_lgb[i]["pred_LightGBM"]

    preds_ci.append({
        "ds": dates_all[t],
        "y_true": y_test,
        "pred": pred_point,
        "lower_95": pred_lo,
        "upper_95": pred_hi,
    })

    if (i + 1) % 30 == 0 or (i + 1) == n_val_steps:
        elapsed = time.time() - start_time
        print(f"    [{i+1}/{n_val_steps}] elapsed: {elapsed:.1f}s")

elapsed = time.time() - start_time
print(f"  Квантильная регрессия завершена за {elapsed:.1f}s")

df_ci = pd.DataFrame(preds_ci)

y_true_ci = df_ci["y_true"].values
lo_ci = df_ci["lower_95"].values
hi_ci = df_ci["upper_95"].values

inside = (y_true_ci >= lo_ci) & (y_true_ci <= hi_ci)
coverage_95 = inside.mean()

avg_width = (hi_ci - lo_ci).mean()

avg_rate = y_true_ci.mean()
norm_width = avg_width / avg_rate * 100

print(f"\n  РЕЗУЛЬТАТЫ:")
print(f"  Coverage (95% CI):     {coverage_95:.1%} (идеал: 95.0%)")
print(f"  Средняя ширина CI:    {avg_width:.2f} п.п.")
print(f"  Норм. ширина CI:     {norm_width:.1f}% от средней ставки")

if coverage_95 < 0.90:
    print(f"  → Интервалы СЛИШКОМ УЗКИЕ (модель overconfident)")
elif coverage_95 > 0.99:
    print(f"  → Интервалы СЛИШКОМ ШИРОКИЕ (модель conservative)")
else:
    print(f"  → Калибровка адекватная")

df_ci.to_csv(f"{FIG_DIR}/ml_prediction_intervals.csv", index=False)

fig, ax = plt.subplots(figsize=(16, 7))

ax.plot(df_ci["ds"], df_ci["y_true"],
        color='black', linewidth=2, label='Факт', zorder=5)

ax.plot(df_ci["ds"], df_ci["pred"],
        color='#2ecc71', linewidth=1.3, alpha=0.8, label='LightGBM (прогноз)')

ax.fill_between(df_ci["ds"],
                df_ci["lower_95"], df_ci["upper_95"],
                color='#2ecc71', alpha=0.15,
                label=f'95% CI (coverage={coverage_95:.1%})')

outside = ~inside
if outside.any():
    ax.scatter(df_ci["ds"][outside], df_ci["y_true"][outside],
               color='red', s=40, zorder=6, label=f'Вне CI ({outside.sum()} точек)')

ax.set_title(f'LightGBM: прогноз с 95% доверительным интервалом (quantile regression)')
ax.set_ylabel('Ключевая ставка, % годовых')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/ml_11_prediction_intervals.png')
plt.close()
print(f"\n  [OK] ml_11_prediction_intervals.png")
print(f"  [OK] ml_prediction_intervals.csv")

preds_xgb = []
last_xgb_model = None

if xgb is not None:
    print(f"\n  XGBoost: expanding window ({n_val_steps} шагов)...")
    start_time = time.time()

    for i in range(n_val_steps):
        t = val_start + i

        X_train = X_all[:t]
        y_train = y_all[:t]
        X_test = X_all[t:t+1]
        y_test = y_all[t]

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]

        preds_xgb.append({
            "ds": dates_all[t],
            "y_true": y_test,
            "pred_XGBoost": pred,
        })

        last_xgb_model = model

        if (i + 1) % 30 == 0 or (i + 1) == n_val_steps:
            elapsed = time.time() - start_time
            print(f"    [{i+1}/{n_val_steps}] elapsed: {elapsed:.1f}s")

    elapsed_xgb = time.time() - start_time
    print(f"  XGBoost завершён за {elapsed_xgb:.1f}s")

print("\n" + "=" * 60)
print("БЛОК 4: FEATURE IMPORTANCE И SHAP")
print("=" * 60)

importances = last_lgb_model.feature_importances_
fi_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances,
}).sort_values("importance", ascending=False)

print("\n  TOP-15 признаков по важности (LightGBM gain):")
for i, row in fi_df.head(15).iterrows():
    bar = "█" * int(row['importance'] / fi_df['importance'].max() * 30)
    print(f"    {row['feature']:25s}: {row['importance']:8.0f}  {bar}")

fig, ax = plt.subplots(figsize=(10, 8))
top_n = min(20, len(fi_df))
top_fi = fi_df.head(top_n)

ax.barh(range(top_n), top_fi["importance"].values, color='#2980b9', alpha=0.8)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_fi["feature"].values)
ax.invert_yaxis()
ax.set_xlabel("Feature Importance (gain)")
ax.set_title("LightGBM: Top-20 признаков по важности")
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/ml_01_feature_importance.png')
plt.close()
print(f"\n  [OK] ml_01_feature_importance.png")

fi_df.to_csv(f"{FIG_DIR}/ml_feature_importance.csv", index=False)

if HAS_SHAP:
    print("\n  Расчёт SHAP-значений (это может занять ~1 мин)...")

    X_train_full = X_all[:val_end]
    explainer = shap.TreeExplainer(last_lgb_model)
    shap_values = explainer.shap_values(X_train_full)

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_train_full,
                      feature_names=feature_cols,
                      max_display=20, show=False)
    plt.title("SHAP: вклад признаков в прогноз ключевой ставки")
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/ml_02_shap_summary.png', bbox_inches='tight')
    plt.close()
    print("  [OK] ml_02_shap_summary.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_train_full,
                      feature_names=feature_cols,
                      plot_type="bar", max_display=20, show=False)
    plt.title("SHAP: средний абсолютный вклад признаков")
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/ml_03_shap_bar.png', bbox_inches='tight')
    plt.close()
    print("  [OK] ml_03_shap_bar.png")

    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    print("\n  TOP-10 признаков по SHAP:")
    for _, row in shap_df.head(10).iterrows():
        print(f"    {row['feature']:25s}: {row['mean_abs_shap']:.4f}")

    shap_df.to_csv(f"{FIG_DIR}/ml_shap_values.csv", index=False)

else:
    print("  SHAP не доступен — пропускаем")

print("\n" + "=" * 60)
print("БЛОК 5: МЕТРИКИ И СРАВНЕНИЕ")
print("=" * 60)

def calc_metrics(y_true, y_pred, name):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    mape = np.mean(np.abs((yt - yp) / yt)) * 100 if np.all(yt != 0) else np.nan
    return {"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE_%": round(mape, 2), "n_forecasts": int(mask.sum())}

ml_metrics = []

df_lgb = pd.DataFrame(preds_lgb)
m_lgb = calc_metrics(df_lgb["y_true"].values, df_lgb["pred_LightGBM"].values, "LightGBM")
ml_metrics.append(m_lgb)
print(f"  LightGBM: MAE={m_lgb['MAE']:.4f}, RMSE={m_lgb['RMSE']:.4f}, MAPE={m_lgb['MAPE_%']:.2f}%")

if preds_xgb:
    df_xgb = pd.DataFrame(preds_xgb)
    m_xgb = calc_metrics(df_xgb["y_true"].values, df_xgb["pred_XGBoost"].values, "XGBoost")
    ml_metrics.append(m_xgb)
    print(f"  XGBoost:  MAE={m_xgb['MAE']:.4f}, RMSE={m_xgb['RMSE']:.4f}, MAPE={m_xgb['MAPE_%']:.2f}%")

ml_metrics_df = pd.DataFrame(ml_metrics)

prev_path = f"{FIG_DIR}/metrics_all_sprint2.csv"
if os.path.exists(prev_path):
    prev_metrics = pd.read_csv(prev_path)
    all_metrics = pd.concat([prev_metrics, ml_metrics_df], ignore_index=True)
else:
    all_metrics = ml_metrics_df

all_metrics = all_metrics.sort_values("MAE").reset_index(drop=True)

print(f"\n  ПОЛНАЯ СВОДНАЯ ТАБЛИЦА (все модели):")
print("  " + "=" * 65)
print(f"  {'#':>3s} {'Модель':17s} | {'MAE':>8s} | {'RMSE':>8s} | {'MAPE %':>8s} | {'N':>3s}")
print("  " + "-" * 65)

naive_mae = all_metrics[all_metrics['model'] == 'Naive']['MAE'].values
naive_mae = naive_mae[0] if len(naive_mae) > 0 else None

for i, row in all_metrics.iterrows():
    marker = ""
    if naive_mae is not None and row['MAE'] < naive_mae:
        marker = " бьёт Naive"
    print(f"  {i+1:3d} {row['model']:17s} | {row['MAE']:8.4f} | {row['RMSE']:8.4f} | "
          f"{row['MAPE_%']:7.2f}% | {int(row['n_forecasts']):3d}{marker}")

all_metrics.to_csv(f"{FIG_DIR}/metrics_all_with_ml.csv", index=False)
ml_metrics_df.to_csv(f"{FIG_DIR}/ml_metrics.csv", index=False)
print(f"\n  [OK] metrics_all_with_ml.csv")



print("\n" + "=" * 60)
print("БЛОК 4.3: КОРРЕЛЯЦИИ (Пирсон, Спирмен, Кендалл)")
print("=" * 60)

corr_results = []

for feat in feature_cols:
    x = df_ml[feat].values
    y_target = df_ml[TARGET].values

    mask = np.isfinite(x) & np.isfinite(y_target)
    x_clean = x[mask]
    y_clean = y_target[mask]

    if len(x_clean) < 10:
        continue

    r_pearson, p_pearson = stats.pearsonr(x_clean, y_clean)

    r_spearman, p_spearman = stats.spearmanr(x_clean, y_clean)

    r_kendall, p_kendall = stats.kendalltau(x_clean, y_clean)

    corr_results.append({
        "feature": feat,
        "pearson_r": round(r_pearson, 4),
        "pearson_p": round(p_pearson, 4),
        "spearman_r": round(r_spearman, 4),
        "spearman_p": round(p_spearman, 4),
        "kendall_tau": round(r_kendall, 4),
        "kendall_p": round(p_kendall, 4),
        "abs_pearson": round(abs(r_pearson), 4),
    })

corr_df = pd.DataFrame(corr_results).sort_values("abs_pearson", ascending=False)

print(f"\n  TOP-15 признаков по |корреляции| с ключевой ставкой:")
print(f"  {'Признак':25s} | {'Пирсон':>8s} | {'Спирмен':>8s} | {'Кендалл':>8s}")
print(f"  {'-'*60}")
for _, row in corr_df.head(15).iterrows():
    sig = "***" if row['pearson_p'] < 0.001 else "**" if row['pearson_p'] < 0.01 else "*" if row['pearson_p'] < 0.05 else ""
    print(f"  {row['feature']:25s} | {row['pearson_r']:+7.3f}{sig:3s} | "
          f"{row['spearman_r']:+7.3f} | {row['kendall_tau']:+7.3f}")

corr_df.to_csv(f"{FIG_DIR}/ml_correlations_all.csv", index=False)
print(f"\n  [OK] ml_correlations_all.csv")

fig, ax = plt.subplots(figsize=(12, 8))
top_corr = corr_df.head(15).iloc[::-1]

y_pos = np.arange(len(top_corr))
height = 0.25

ax.barh(y_pos - height, top_corr["pearson_r"].values,
        height=height, color='#2980b9', alpha=0.8, label='Пирсон (линейная)')
ax.barh(y_pos, top_corr["spearman_r"].values,
        height=height, color='#e74c3c', alpha=0.8, label='Спирмен (монотонная)')
ax.barh(y_pos + height, top_corr["kendall_tau"].values,
        height=height, color='#27ae60', alpha=0.8, label='Кендалл (ранговая)')

ax.set_yticks(y_pos)
ax.set_yticklabels(top_corr["feature"].values)
ax.set_xlabel("Корреляция с ключевой ставкой")
ax.set_title("Сравнение трёх мер корреляции: TOP-15 признаков")
ax.legend(loc='lower right')
ax.axvline(x=0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/ml_08_correlations_comparison.png')
plt.close()
print("  [OK] ml_08_correlations_comparison.png")


print("\n" + "=" * 60)
print("БЛОК 4.4: PERMUTATION IMPORTANCE")
print("=" * 60)

X_train_pi = X_all[:val_end]
y_train_pi = y_all[:val_end]

print(f"  Расчёт permutation importance на {X_train_pi.shape[0]} наблюдениях...")
print(f"  (10 повторений для каждого признака для устойчивости)")

perm_result = sklearn_perm_importance(
    last_lgb_model,
    X_train_pi,
    y_train_pi,
    n_repeats=10,       
    random_state=42,
    scoring='neg_mean_absolute_error', 
)

perm_df = pd.DataFrame({
    "feature": feature_cols,
    "perm_importance_mean": np.round(perm_result.importances_mean, 4),
    "perm_importance_std": np.round(perm_result.importances_std, 4),
}).sort_values("perm_importance_mean", ascending=False)

print(f"\n  TOP-15 признаков по Permutation Importance:")
print(f"  {'Признак':25s} | {'PI mean':>10s} | {'PI std':>10s} | {'Вывод'}")
print(f"  {'-'*65}")
for _, row in perm_df.head(15).iterrows():
    if row['perm_importance_mean'] > 2 * row['perm_importance_std']:
        verdict = "ЗНАЧИМ"
    elif row['perm_importance_mean'] > row['perm_importance_std']:
        verdict = "вероятно значим"
    elif row['perm_importance_mean'] > 0:
        verdict = "слабый сигнал"
    else:
        verdict = "не значим"
    print(f"  {row['feature']:25s} | {row['perm_importance_mean']:10.4f} | "
          f"{row['perm_importance_std']:10.4f} | {verdict}")

perm_df.to_csv(f"{FIG_DIR}/ml_permutation_importance.csv", index=False)
print(f"\n  [OK] ml_permutation_importance.csv")

fig, ax = plt.subplots(figsize=(10, 8))
top_perm = perm_df.head(20).iloc[::-1]

colors_pi = ['#2ecc71' if v > 0 else '#e74c3c' for v in top_perm['perm_importance_mean'].values]
ax.barh(range(len(top_perm)), top_perm["perm_importance_mean"].values,
        xerr=top_perm["perm_importance_std"].values,
        color=colors_pi, alpha=0.8, capsize=3)
ax.set_yticks(range(len(top_perm)))
ax.set_yticklabels(top_perm["feature"].values)
ax.set_xlabel("Permutation Importance (ухудшение MAE при перемешивании)")
ax.set_title("Permutation Importance: TOP-20 признаков (LightGBM)")
ax.axvline(x=0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/ml_09_permutation_importance.png')
plt.close()
print("  [OK] ml_09_permutation_importance.png")

print("\n" + "=" * 60)
print("БЛОК 4.5: СВОДНАЯ ТАБЛИЦА (5 методов)")
print("=" * 60)

fi_ranked = fi_df.reset_index(drop=True)
fi_ranked["rank_builtin"] = range(1, len(fi_ranked) + 1)

if HAS_SHAP:
    shap_ranked = shap_df.reset_index(drop=True)
    shap_ranked["rank_shap"] = range(1, len(shap_ranked) + 1)
else:
    shap_ranked = pd.DataFrame({"feature": feature_cols, "rank_shap": [None]*len(feature_cols)})

corr_ranked = corr_df.reset_index(drop=True)
corr_ranked["rank_pearson"] = range(1, len(corr_ranked) + 1)

spearman_ranked = corr_df.sort_values("spearman_r", key=abs, ascending=False).reset_index(drop=True)
spearman_ranked["rank_spearman"] = range(1, len(spearman_ranked) + 1)

perm_ranked = perm_df.reset_index(drop=True)
perm_ranked["rank_perm"] = range(1, len(perm_ranked) + 1)

summary = pd.DataFrame({"feature": feature_cols})
summary = summary.merge(fi_ranked[["feature", "rank_builtin"]], on="feature", how="left")
if HAS_SHAP:
    summary = summary.merge(shap_ranked[["feature", "rank_shap"]], on="feature", how="left")
summary = summary.merge(corr_ranked[["feature", "rank_pearson"]], on="feature", how="left")
summary = summary.merge(spearman_ranked[["feature", "rank_spearman"]], on="feature", how="left")
summary = summary.merge(perm_ranked[["feature", "rank_perm"]], on="feature", how="left")

rank_cols = [c for c in summary.columns if c.startswith("rank_")]
summary["mean_rank"] = summary[rank_cols].mean(axis=1)
summary = summary.sort_values("mean_rank")

print(f"\n  СВОДНАЯ ТАБЛИЦА РАНГОВ (5 методов, TOP-15 по среднему рангу):")
header = f"  {'Признак':25s} | {'BI':>4s} | {'SHAP':>4s} | {'Pears':>5s} | {'Spear':>5s} | {'Perm':>4s} | {'Сред.':>5s}"
print(header)
print(f"  {'-'*70}")
for _, row in summary.head(15).iterrows():
    vals = []
    for c in rank_cols:
        v = row[c]
        vals.append(f"{int(v):4d}" if pd.notna(v) else "   -")
    print(f"  {row['feature']:25s} | {' | '.join(vals)} | {row['mean_rank']:5.1f}")

kbd_in_top10 = summary.head(10)
kbd_count = sum(1 for f in kbd_in_top10["feature"] if "zcyc" in f or "spread" in f)
non_kbd_count = 10 - kbd_count

print(f"\n  В TOP-10 по среднему рангу:")
print(f"    КБД-признаки:     {kbd_count} из 10")
print(f"    Не-КБД признаки:  {non_kbd_count} из 10")

summary.to_csv(f"{FIG_DIR}/ml_importance_summary_5methods.csv", index=False)
print(f"\n  [OK] ml_importance_summary_5methods.csv")

fig, ax = plt.subplots(figsize=(14, 8))

top_summary = summary.head(15).iloc[::-1]
y_pos = np.arange(len(top_summary))

method_labels = {
    "rank_builtin": ("Built-in (gain)", "#2980b9"),
    "rank_shap": ("SHAP", "#e74c3c"),
    "rank_pearson": ("Пирсон", "#27ae60"),
    "rank_spearman": ("Спирмен", "#f39c12"),
    "rank_perm": ("Permutation", "#9b59b6"),
}

width = 0.15
for i, (col, (label, color)) in enumerate(method_labels.items()):
    if col in top_summary.columns:
        vals = top_summary[col].values
        offset = (i - 2) * width
        ax.barh(y_pos + offset, vals, height=width,
                color=color, alpha=0.8, label=label)

ax.set_yticks(y_pos)
ax.set_yticklabels(top_summary["feature"].values)
ax.set_xlabel("Ранг (1 = самый важный)")
ax.set_title("Сводная оценка важности: 5 методов (TOP-15 признаков)")
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_xaxis()  # Ранг 1 справа (ближе к оси Y)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/ml_10_importance_5methods.png')
plt.close()
print("  [OK] ml_10_importance_5methods.png")

print("\n" + "=" * 60)
print("ABLATION STUDY: ВКЛАД КБД ОФЗ В ТОЧНОСТЬ ПРОГНОЗА")
print("=" * 60)

kbd_features = [f for f in feature_cols if any(
    f.startswith(prefix) or prefix in f
    for prefix in ["zcyc_", "spread_"]
)]
non_kbd_features = [f for f in feature_cols if f not in kbd_features]

print(f"\n  Всего признаков:     {len(feature_cols)}")
print(f"  Признаков КБД:      {len(kbd_features)}")
print(f"  Признаков без КБД:  {len(non_kbd_features)}")
print(f"\n  Убираемые признаки (КБД):")
for f in kbd_features:
    print(f"    - {f}")
print(f"\n  Остающиеся признаки:")
for f in non_kbd_features:
    print(f"    + {f}")

X_all_no_kbd = df_ml[non_kbd_features].values

print(f"\n  LightGBM БЕЗ КБД: expanding window ({n_val_steps} шагов)...")
start_time = time.time()

preds_no_kbd = []

for i in range(n_val_steps):
    t = val_start + i

    X_train = X_all_no_kbd[:t]
    y_train = y_all[:t]
    X_test = X_all_no_kbd[t:t+1]
    y_test = y_all[t]

    model_no_kbd = lgb.LGBMRegressor(**lgb_params)
    model_no_kbd.fit(X_train, y_train)
    pred = model_no_kbd.predict(X_test)[0]

    preds_no_kbd.append({
        "ds": dates_all[t],
        "y_true": y_test,
        "pred_LightGBM_no_KBD": pred,
    })

elapsed_no_kbd = time.time() - start_time
print(f"  Завершено за {elapsed_no_kbd:.1f}s")

df_no_kbd = pd.DataFrame(preds_no_kbd)
m_no_kbd = calc_metrics(
    df_no_kbd["y_true"].values,
    df_no_kbd["pred_LightGBM_no_KBD"].values,
    "LightGBM (без КБД)"
)
print(f"\n  {'='*60}")
print(f"  РЕЗУЛЬТАТ ABLATION STUDY")
print(f"  {'='*60}")
print(f"  {'Конфигурация':30s} | {'MAE':>8s} | {'RMSE':>8s} | {'MAPE%':>8s}")
print(f"  {'-'*60}")
print(f"  {'LightGBM (все признаки)':30s} | {m_lgb['MAE']:8.4f} | {m_lgb['RMSE']:8.4f} | {m_lgb['MAPE_%']:7.2f}%")
print(f"  {'LightGBM (без КБД)':30s} | {m_no_kbd['MAE']:8.4f} | {m_no_kbd['RMSE']:8.4f} | {m_no_kbd['MAPE_%']:7.2f}%")
print(f"  {'-'*60}")

mae_diff = m_no_kbd['MAE'] - m_lgb['MAE']
mae_pct = (mae_diff / m_lgb['MAE']) * 100

if mae_diff > 0:
    print(f"  Удаление КБД УХУДШИЛО прогноз на {mae_diff:.4f} п.п. ({mae_pct:+.1f}%)")
else:
    print(f"  Удаление КБД УЛУЧШИЛО прогноз на {abs(mae_diff):.4f} п.п. ({mae_pct:.1f}%)")

ablation_df = pd.DataFrame([m_lgb, m_no_kbd])
ablation_df.to_csv(f"{FIG_DIR}/ml_ablation_kbd.csv", index=False)
print(f"\n  [OK] ml_ablation_kbd.csv")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.plot(df_lgb["ds"], df_lgb["y_true"],
        color='black', linewidth=2, label='Факт', zorder=5)
ax.plot(df_lgb["ds"], df_lgb["pred_LightGBM"],
        color='#2ecc71', linewidth=1.3, alpha=0.8,
        label=f'С КБД (MAE={m_lgb["MAE"]:.3f})')
ax.plot(df_no_kbd["ds"], df_no_kbd["pred_LightGBM_no_KBD"],
        color='#e74c3c', linewidth=1.3, alpha=0.8, linestyle='--',
        label=f'Без КБД (MAE={m_no_kbd["MAE"]:.3f})')
ax.set_title('Ablation: прогнозы с КБД и без КБД')
ax.set_ylabel('Ключевая ставка, % годовых')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

ax = axes[1]
models_ab = ['С КБД\n(все признаки)', 'Без КБД\n(zcyc/spread убраны)']
maes_ab = [m_lgb['MAE'], m_no_kbd['MAE']]
colors_ab = ['#2ecc71', '#e74c3c']
bars = ax.bar(models_ab, maes_ab, color=colors_ab, alpha=0.85, width=0.5)
for bar, val in zip(bars, maes_ab):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.4f}', ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('MAE (п.п.)')
ax.set_title(f'Вклад КБД: {mae_pct:+.1f}% к MAE')
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Ablation Study: вклад КБД ОФЗ в точность LightGBM', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/ml_07_ablation_kbd.png')
plt.close()
print("  [OK] ml_07_ablation_kbd.png")

print("\n" + "=" * 60)
print("БЛОК 6: ВИЗУАЛИЗАЦИЯ")
print("=" * 60)

fig, ax = plt.subplots(figsize=(16, 7))

ax.plot(df_lgb["ds"], df_lgb["y_true"],
        color='black', linewidth=2, label='Факт', zorder=5)
ax.plot(df_lgb["ds"], df_lgb["pred_LightGBM"],
        color='#2ecc71', linewidth=1.3, alpha=0.8, label='LightGBM')

if preds_xgb:
    ax.plot(df_xgb["ds"], df_xgb["pred_XGBoost"],
            color='#3498db', linewidth=1.3, alpha=0.8, label='XGBoost')

ax.set_title(f'ML-модели: прогнозы vs факт (expanding window, h={HORIZON})')
ax.set_ylabel('Ключевая ставка, % годовых')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/ml_04_forecasts.png')
plt.close()
print("  [OK] ml_04_forecasts.png")

n_plots = 1 + (1 if preds_xgb else 0)
fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 5))
if n_plots == 1:
    axes = [axes]

errors_lgb = df_lgb["y_true"] - df_lgb["pred_LightGBM"]
axes[0].bar(df_lgb["ds"], errors_lgb, color='#2ecc71', alpha=0.7, width=25)
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].set_title("LightGBM — ошибки")
axes[0].set_ylabel("Ошибка (п.п.)")
axes[0].grid(True, alpha=0.3)
mae_lgb = np.mean(np.abs(errors_lgb))
bias_lgb = errors_lgb.mean()
axes[0].text(0.02, 0.95, f'MAE={mae_lgb:.3f}, bias={bias_lgb:+.3f}',
             transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

if preds_xgb:
    errors_xgb = df_xgb["y_true"] - df_xgb["pred_XGBoost"]
    axes[1].bar(df_xgb["ds"], errors_xgb, color='#3498db', alpha=0.7, width=25)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_title("XGBoost — ошибки")
    axes[1].set_ylabel("Ошибка (п.п.)")
    axes[1].grid(True, alpha=0.3)
    mae_xgb = np.mean(np.abs(errors_xgb))
    bias_xgb = errors_xgb.mean()
    axes[1].text(0.02, 0.95, f'MAE={mae_xgb:.3f}, bias={bias_xgb:+.3f}',
                 transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle(f'Ошибки ML-моделей (h={HORIZON})', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/ml_05_errors.png')
plt.close()
print("  [OK] ml_05_errors.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 8))
metrics_to_plot = ["MAE", "RMSE", "MAPE_%"]
metric_labels = ["MAE (п.п.)", "RMSE (п.п.)", "MAPE (%)"]

all_colors = {
    "Naive": "#bdc3c7", "Drift": "#bdc3c7", "ARIMA": "#95a5a6", "Decomp+Trend": "#95a5a6",
    "AutoARIMA": "#2980b9", "AutoETS": "#e74c3c", "CES": "#9b59b6", "AutoTheta": "#f39c12",
    "Prophet-base": "#8e44ad", "Prophet-exog": "#e67e22",
    "LightGBM": "#2ecc71", "XGBoost": "#3498db",
}

for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
    sorted_df = all_metrics.sort_values(metric)
    models = sorted_df["model"].values
    values = sorted_df[metric].values

    bar_colors = [all_colors.get(m, '#888888') for m in models]
    bars = ax.barh(models, values, color=bar_colors, alpha=0.85)
    ax.set_xlabel(label)
    ax.set_title(label)
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        fmt = f'{val:.3f}' if metric != "MAPE_%" else f'{val:.1f}%'
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                fmt, va='center', fontsize=9)

plt.suptitle(f'Все модели Спринтов 1-2 (h={HORIZON})', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/ml_06_comparison_all.png')
plt.close()
print("  [OK] ml_06_comparison_all.png")


all_preds = df_lgb[["ds", "y_true", "pred_LightGBM"]].copy()
if preds_xgb:
    all_preds["pred_XGBoost"] = df_xgb["pred_XGBoost"].values

all_preds.to_csv(f"{FIG_DIR}/ml_all_predictions.csv", index=False)
print("  [OK] ml_all_predictions.csv")


print("\n" + "=" * 60)
print("ИТОГОВАЯ СВОДКА СПРИНТА 2")
print("=" * 60)

best = all_metrics.iloc[0]
lgb_beaten_naive = naive_mae is not None and m_lgb['MAE'] < naive_mae

naive_text = f"{naive_mae:.4f}" if naive_mae is not None else "N/A"
if lgb_beaten_naive and naive_mae:
    diff_text = f"LightGBM улучшил Naive на {(1 - m_lgb['MAE']/naive_mae)*100:.1f}%"
elif naive_mae:
    diff_text = f"LightGBM проиграл Naive на {(m_lgb['MAE']/naive_mae - 1)*100:.1f}%"
else:
    diff_text = "Сравнение с Naive недоступно"

top_3_shap = ", ".join([f"'{f}'" for f in shap_df['feature'].head(3).values]) if HAS_SHAP else "N/A"

print(f"""
ПОЛНАЯ ТАБЛИЦА МЕТРИК (все {len(all_metrics)} моделей):

{all_metrics.to_string(index=False)}

ЛУЧШАЯ модель: {best['model']} (MAE = {best['MAE']:.4f})

LightGBM MAE = {m_lgb['MAE']:.4f}  {'★ ПОБИЛ Naive!' if lgb_beaten_naive else ''}
Naive    MAE = {naive_text}

""")

print("Все сохранённые файлы ML:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith('ml_'):
        path = os.path.join(FIG_DIR, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {path} ({size_kb:.0f} KB)")