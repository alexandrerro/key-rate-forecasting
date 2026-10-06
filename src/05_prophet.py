import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import time
import warnings

warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
except ImportError:
    try:
        from fbprophet import Prophet
    except ImportError:
        print("[!] Prophet не установлен.")
        print("    Установите: pip install prophet")
        print("    Или через conda: conda install -c conda-forge prophet")
        exit(1)

from paths import DATA_DIR, FIG_DIR
DATA_PATH = DATA_DIR / "dataset_monthly.csv"

MIN_TRAIN_SIZE = 60
HORIZON = 1

EXOG_VARS = [
    "zcyc_1y", "spread_10y_1y", "cpi_mom", "brent", "usd_rub", "m2",
    "ruonia",                   
    "spread_ruonia_keyrate",   
]

EXOG_LAG = 1

plt.rcParams.update({
    'figure.figsize': (14, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

print("=" * 60)
print("БЛОК 1: ПОДГОТОВКА ДАННЫХ ДЛЯ PROPHET")
print("=" * 60)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

prophet_base = df[["date", "key_rate"]].dropna().copy()
prophet_base = prophet_base.rename(columns={"date": "ds", "key_rate": "y"})

print(f"Базовый датасет: {len(prophet_base)} наблюдений")
print(f"Период: {prophet_base['ds'].min():%Y-%m} — {prophet_base['ds'].max():%Y-%m}")

prophet_exog = df[["date", "key_rate"] + EXOG_VARS].copy()
prophet_exog = prophet_exog.rename(columns={"date": "ds", "key_rate": "y"})

exog_lagged_names = []
for var in EXOG_VARS:
    lagged_name = f"{var}_lag{EXOG_LAG}"
    prophet_exog[lagged_name] = prophet_exog[var].shift(EXOG_LAG)
    exog_lagged_names.append(lagged_name)

prophet_exog = prophet_exog.drop(columns=EXOG_VARS)
prophet_exog = prophet_exog.dropna().reset_index(drop=True)

common_dates = set(prophet_exog["ds"]) & set(prophet_base["ds"])
prophet_base = prophet_base[prophet_base["ds"].isin(common_dates)].reset_index(drop=True)
prophet_exog = prophet_exog[prophet_exog["ds"].isin(common_dates)].reset_index(drop=True)

print(f"Датасет с экзогенными: {len(prophet_exog)} наблюдений")
print(f"Экзогенные переменные (с лагом {EXOG_LAG}):")
for var in exog_lagged_names:
    n_valid = prophet_exog[var].notna().sum()
    print(f"  {var}: {n_valid} значений")

n_total = len(prophet_base)
n_val_steps = n_total - MIN_TRAIN_SIZE - HORIZON + 1
print(f"\nШагов валидации: {n_val_steps}")


print("\n" + "=" * 60)
print("БЛОК 2: PROPHET-BASE (без регрессоров)")
print("=" * 60)

print(f"  Запуск expanding window CV ({n_val_steps} шагов)...")
start_time = time.time()

preds_base = []

for i in range(n_val_steps):
    t = MIN_TRAIN_SIZE + i 

    train = prophet_base.iloc[:t].copy()
    actual = prophet_base.iloc[t]["y"]
    actual_date = prophet_base.iloc[t]["ds"]

    try:
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            interval_width=0.95,
        )
        model.fit(train)
 
        future = model.make_future_dataframe(periods=HORIZON, freq="MS")
        forecast = model.predict(future)
        pred = forecast.iloc[-1]["yhat"]
        pred_lower = forecast.iloc[-1]["yhat_lower"]
        pred_upper = forecast.iloc[-1]["yhat_upper"] 
 
    except Exception as e:
        pred = train["y"].iloc[-1]
        pred_lower = pred - 1.0
        pred_upper = pred + 1.0
 
    preds_base.append({
        "ds": actual_date,
        "y_true": actual,
        "pred_Prophet_base": pred,
        "lower_Prophet_base": pred_lower,
        "upper_Prophet_base": pred_upper, 
    })

    if (i + 1) % 20 == 0 or (i + 1) == n_val_steps:
        elapsed = time.time() - start_time
        print(f"    [{i+1}/{n_val_steps}] elapsed: {elapsed:.0f}s")

elapsed = time.time() - start_time
print(f"  Prophet-base завершён за {elapsed:.0f}s")


print("\n" + "=" * 60)
print("БЛОК 3: PROPHET-EXOG (с регрессорами)")
print("=" * 60)

print(f"  Регрессоры: {', '.join(exog_lagged_names)}")
print(f"  Запуск expanding window CV ({n_val_steps} шагов)...")
start_time = time.time()

preds_exog = []

for i in range(n_val_steps):
    t = MIN_TRAIN_SIZE + i

    train = prophet_exog.iloc[:t].copy()
    actual = prophet_exog.iloc[t]["y"]
    actual_date = prophet_exog.iloc[t]["ds"]

    try:
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            interval_width=0.95,
        )
 
        for var in exog_lagged_names:
            model.add_regressor(var)
 
        model.fit(train)
 
        future = prophet_exog.iloc[t:t+HORIZON][["ds"] + exog_lagged_names].copy()
        forecast = model.predict(future)
        pred = forecast.iloc[-1]["yhat"]
        pred_lower = forecast.iloc[-1]["yhat_lower"]
        pred_upper = forecast.iloc[-1]["yhat_upper"]
 
    except Exception as e:
        pred = train["y"].iloc[-1]
        pred_lower = pred - 1.0
        pred_upper = pred + 1.0
 
    preds_exog.append({
        "ds": actual_date,
        "y_true": actual,
        "pred_Prophet_exog": pred,
        "lower_Prophet_exog": pred_lower,
        "upper_Prophet_exog": pred_upper, 
    })

    if (i + 1) % 20 == 0 or (i + 1) == n_val_steps:
        elapsed = time.time() - start_time
        print(f"    [{i+1}/{n_val_steps}] elapsed: {elapsed:.0f}s")

elapsed = time.time() - start_time
print(f"  Prophet-exog завершён за {elapsed:.0f}s")

print("\n" + "=" * 60)
print("БЛОК 4: МЕТРИКИ И СРАВНЕНИЕ")
print("=" * 60)

def calc_metrics(y_true, y_pred, name):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    mape = np.mean(np.abs((yt - yp) / yt)) * 100 if np.all(yt != 0) else np.nan
    return {"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE_%": round(mape, 2), "n_forecasts": int(mask.sum())}

df_base = pd.DataFrame(preds_base)
df_exog = pd.DataFrame(preds_exog)

prophet_metrics = []

m1 = calc_metrics(df_base["y_true"].values, df_base["pred_Prophet_base"].values, "Prophet-base")
prophet_metrics.append(m1)
print(f"  Prophet-base: MAE={m1['MAE']:.4f}, RMSE={m1['RMSE']:.4f}, MAPE={m1['MAPE_%']:.2f}%")

m2 = calc_metrics(df_exog["y_true"].values, df_exog["pred_Prophet_exog"].values, "Prophet-exog")
prophet_metrics.append(m2)
print(f"  Prophet-exog: MAE={m2['MAE']:.4f}, RMSE={m2['RMSE']:.4f}, MAPE={m2['MAPE_%']:.2f}%")

prophet_metrics_df = pd.DataFrame(prophet_metrics)

prev_metrics_path = f"{FIG_DIR}/metrics_baseline_sf.csv"
if os.path.exists(prev_metrics_path):
    prev_metrics = pd.read_csv(prev_metrics_path)
    all_metrics = pd.concat([prev_metrics, prophet_metrics_df], ignore_index=True)
else:
    all_metrics = prophet_metrics_df

all_metrics = all_metrics.sort_values("MAE").reset_index(drop=True)

print(f"\n  СВОДНАЯ ТАБЛИЦА (все модели):")
print("  " + "=" * 65)
print(f"  {'#':>3s} {'Модель':17s} | {'MAE':>8s} | {'RMSE':>8s} | {'MAPE %':>8s} | {'N':>3s}")
print("  " + "-" * 65)

naive_mae = all_metrics[all_metrics['model'] == 'Naive']['MAE'].values
naive_mae = naive_mae[0] if len(naive_mae) > 0 else None

for i, row in all_metrics.iterrows():
    marker = ""
    if naive_mae is not None and row['MAE'] < naive_mae:
        marker = " < Naive"
    print(f"  {i+1:3d} {row['model']:17s} | {row['MAE']:8.4f} | {row['RMSE']:8.4f} | "
          f"{row['MAPE_%']:7.2f}% | {int(row['n_forecasts']):3d}{marker}")

all_metrics.to_csv(f"{FIG_DIR}/metrics_all_sprint2.csv", index=False)
prophet_metrics_df.to_csv(f"{FIG_DIR}/prophet_metrics.csv", index=False)
print(f"\n  [OK] metrics_all_sprint2.csv")


print("\n" + "=" * 60)
print("ИНТЕРВАЛЬНЫЕ ОЦЕНКИ PROPHET")
print("=" * 60)
 
df_base = pd.DataFrame(preds_base)
df_exog = pd.DataFrame(preds_exog)

for name, df_pred in [("Prophet-base", df_base), ("Prophet-exog", df_exog)]:
    lo_col = f"lower_{name.replace('-', '_')}"
    hi_col = f"upper_{name.replace('-', '_')}"
    
    if lo_col in df_pred.columns and hi_col in df_pred.columns:
        y_true = df_pred["y_true"].values
        lo = df_pred[lo_col].values
        hi = df_pred[hi_col].values
        
        inside = (y_true >= lo) & (y_true <= hi)
        coverage = inside.mean()
        avg_width = (hi - lo).mean()
        
        print(f"  {name}:")
        print(f"    Coverage (95% CI): {coverage:.1%}")
        print(f"    Средняя ширина CI: {avg_width:.2f} п.п.")
        print(f"    Идеал: 95.0%")
        if coverage < 0.90:
            print(f"    → Интервалы слишком узкие (overconfident)")
        elif coverage > 0.99:
            print(f"    → Интервалы слишком широкие (conservative)")
        else:
            print(f"    → Калибровка адекватная")
 
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
 
for ax, (name, df_pred, color) in zip(axes, [
    ("Prophet-base", df_base, '#9b59b6'),
    ("Prophet-exog", df_exog, '#e67e22'),
]):
    ax.plot(df_pred["ds"], df_pred["y_true"],
            color='black', linewidth=2, label='Факт', zorder=5)
    
    pred_col = f"pred_{name.replace('-', '_')}"
    lo_col = f"lower_{name.replace('-', '_')}"
    hi_col = f"upper_{name.replace('-', '_')}"
    
    if pred_col in df_pred.columns:
        ax.plot(df_pred["ds"], df_pred[pred_col],
                color=color, linewidth=1.3, alpha=0.8, label=name)
    
    if lo_col in df_pred.columns and hi_col in df_pred.columns:
        ax.fill_between(df_pred["ds"],
                        df_pred[lo_col], df_pred[hi_col],
                        color=color, alpha=0.15, label='95% CI')
        
        y_true = df_pred["y_true"].values
        inside = (y_true >= df_pred[lo_col].values) & (y_true <= df_pred[hi_col].values)
        cov = inside.mean()
        width = (df_pred[hi_col] - df_pred[lo_col]).mean()
        
        ax.text(0.02, 0.95,
                f'Coverage: {cov:.1%}\nШирина CI: {width:.2f} п.п.',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(name)
    ax.set_ylabel('% годовых')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
 
plt.suptitle('Prophet: прогнозы с 95% доверительными интервалами', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/prophet_04_prediction_intervals_all.png')
plt.close()
print("  [OK] prophet_04_prediction_intervals_all.png")
 


print("\n" + "=" * 60)
print("БЛОК 5: ВИЗУАЛИЗАЦИЯ")
print("=" * 60)

fig, ax = plt.subplots(figsize=(16, 7))

ax.plot(df_base["ds"], df_base["y_true"],
        color='black', linewidth=2, label='Факт', zorder=5)
ax.plot(df_base["ds"], df_base["pred_Prophet_base"],
        color='#9b59b6', linewidth=1.3, alpha=0.8, label='Prophet-base')
ax.plot(df_exog["ds"], df_exog["pred_Prophet_exog"],
        color='#e67e22', linewidth=1.3, alpha=0.8, label='Prophet-exog')

ax.set_title(f'Prophet: прогнозы vs факт (expanding window, h={HORIZON})')
ax.set_ylabel('Ключевая ставка, % годовых')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/prophet_01_forecasts.png')
plt.close()
print("  [OK] prophet_01_forecasts.png")


fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, (name, df_pred, color) in zip(axes, [
    ("Prophet-base", df_base, '#9b59b6'),
    ("Prophet-exog", df_exog, '#e67e22'),
]):
    pred_col = [c for c in df_pred.columns if c.startswith("pred_")][0]
    errors = df_pred["y_true"] - df_pred[pred_col]

    ax.bar(df_pred["ds"], errors, color=color, alpha=0.7, width=25)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title(f'{name} — ошибки прогноза')
    ax.set_ylabel('Ошибка (п.п.)')
    ax.grid(True, alpha=0.3)

    mae = np.mean(np.abs(errors))
    bias = errors.mean()
    ax.text(0.02, 0.95, f'MAE={mae:.3f}, bias={bias:+.3f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle(f'Ошибки прогнозов Prophet (h={HORIZON})', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/prophet_02_errors.png')
plt.close()
print("  [OK] prophet_02_errors.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
metrics_to_plot = ["MAE", "RMSE", "MAPE_%"]
metric_labels = ["MAE (п.п.)", "RMSE (п.п.)", "MAPE (%)"]

all_colors = {
    "Naive": "#bdc3c7", "Drift": "#bdc3c7", "ARIMA": "#95a5a6", "Decomp+Trend": "#95a5a6",
    "AutoARIMA": "#2980b9", "AutoETS": "#e74c3c", "CES": "#27ae60", "AutoTheta": "#f39c12",
    "Prophet-base": "#9b59b6", "Prophet-exog": "#e67e22",
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

plt.suptitle(f'Все модели: Baseline + StatsForecast + Prophet (h={HORIZON})',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/prophet_03_comparison_all.png')
plt.close()
print("  [OK] prophet_03_comparison_all.png")

all_preds = df_base[["ds", "y_true", "pred_Prophet_base"]].copy()
all_preds = all_preds.merge(
    df_exog[["ds", "pred_Prophet_exog"]],
    on="ds", how="left"
)
all_preds.to_csv(f"{FIG_DIR}/prophet_all_predictions.csv", index=False)
print("  [OK] prophet_all_predictions.csv")

print("\n" + "=" * 60)
print("ИТОГОВАЯ СВОДКА")
print("=" * 60)

best = all_metrics.iloc[0]
exog_mae = m2['MAE']
base_mae = m1['MAE']
exog_better = exog_mae < base_mae

exog_vs_naive = ""
if naive_mae is not None:
    diff = ((exog_mae - naive_mae) / naive_mae) * 100
    if exog_mae < naive_mae:
        exog_vs_naive = f"Prophet-exog ПОБИЛ Naive на {abs(diff):.1f}%"
    else:
        exog_vs_naive = f"Prophet-exog проиграл Naive на {abs(diff):.1f}%"

print(f"""
ПОЛНАЯ ТАБЛИЦА МЕТРИК:

{all_metrics.to_string(index=False)}

ЛУЧШАЯ модель общая: {best['model']} (MAE = {best['MAE']:.4f})

Prophet-base:  MAE = {base_mae:.4f}
Prophet-exog:  MAE = {exog_mae:.4f}
Экзогенные регрессоры {'ПОМОГЛИ' if exog_better else 'НЕ ПОМОГЛИ'}: \
{'улучшение' if exog_better else 'ухудшение'} на {abs(exog_mae - base_mae):.4f} п.п.
{exog_vs_naive}

""")

print("Сохранённые файлы:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith('prophet_') or f == 'metrics_all_sprint2.csv':
        path = os.path.join(FIG_DIR, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {path} ({size_kb:.0f} KB)")
