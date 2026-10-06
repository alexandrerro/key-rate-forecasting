
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import time
import warnings

warnings.filterwarnings('ignore')

# Импорт StatsForecast
try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, AutoCES
except ImportError as e:
    print(f"[!] Ошибка импорта: {e}")
    print("    Установите/обновите: pip install statsforecast")
    exit(1)


from paths import DATA_DIR, FIG_DIR
DATA_PATH = DATA_DIR / "dataset_monthly.csv"


MIN_TRAIN_SIZE = 60    
HORIZON = 1            
STEP_SIZE = 1         

plt.rcParams.update({
    'figure.figsize': (14, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})



print("=" * 60)
print("БЛОК 1: ПОДГОТОВКА ДАННЫХ В ФОРМАТЕ NIXTLA")
print("=" * 60)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

nixtla_df = df[["date", "key_rate"]].dropna().copy()
nixtla_df = nixtla_df.rename(columns={"date": "ds", "key_rate": "y"})
nixtla_df["unique_id"] = "key_rate"

nixtla_df = nixtla_df[["unique_id", "ds", "y"]]

nixtla_df["ds"] = pd.to_datetime(nixtla_df["ds"])

nixtla_df = nixtla_df.sort_values("ds").reset_index(drop=True)

print(f"Формат Nixtla: {nixtla_df.shape[0]} наблюдений")
print(f"Период: {nixtla_df['ds'].min():%Y-%m} — {nixtla_df['ds'].max():%Y-%m}")
print(f"Столбцы: {list(nixtla_df.columns)}")
print(f"\nПример данных:")
print(nixtla_df.head(3).to_string(index=False))

nixtla_df.to_csv(f"{FIG_DIR}/nixtla_key_rate.csv", index=False)
print(f"\n[OK] nixtla_key_rate.csv — переиспользуйте в 05, 06, 07, 08")



print("\n" + "=" * 60)
print("БЛОК 2: ОПРЕДЕЛЕНИЕ МОДЕЛЕЙ")
print("=" * 60)


models = [

    AutoARIMA(season_length=1),

    AutoETS(season_length=1, model='ZZZ'),

    AutoCES(season_length=1),

    AutoTheta(season_length=1),
]

model_names = ["AutoARIMA", "AutoETS", "CES", "AutoTheta"]
print(f"Модели: {', '.join(model_names)}")
print(f"Сезонность: отключена (season_length=1, обосновано EDA)")

 
print("\n" + "=" * 60)
print("БЛОК 3: КРОСС-ВАЛИДАЦИЯ С ИНТЕРВАЛЬНЫМИ ОЦЕНКАМИ")
print("=" * 60)
 
CONFIDENCE_LEVELS = [90, 95]
 
sf = StatsForecast(
    models=models,
    freq="MS",
    n_jobs=1,
)
 
n_total = len(nixtla_df)
n_windows = n_total - MIN_TRAIN_SIZE - HORIZON + 1
 
print(f"\nПараметры валидации:")
print(f"  Наблюдений: {n_total}")
print(f"  Начальное окно: {MIN_TRAIN_SIZE} месяцев")
print(f"  Горизонт: h={HORIZON}")
print(f"  Шагов валидации: {n_windows}")
print(f"  Уровни доверия: {CONFIDENCE_LEVELS}%")
 
print(f"\nЗапуск кросс-валидации...")
start_time = time.time()
 
cv_results = sf.cross_validation(
    df=nixtla_df,
    h=HORIZON,
    step_size=STEP_SIZE,
    n_windows=n_windows,
    level=CONFIDENCE_LEVELS, 
)
 
elapsed = time.time() - start_time
print(f"Готово за {elapsed:.1f} секунд")
 
print(f"\nСтолбцы результата: {list(cv_results.columns)}")
 
# Убираем лишние столбцы
if 'unique_id' in cv_results.columns:
    cv_results = cv_results.drop(columns=['unique_id'])
if 'cutoff' in cv_results.columns:
    cv_results = cv_results.drop(columns=['cutoff'])
 
print(f"\n  COVERAGE (попадание факта в интервал):")
print(f"  {'Модель':15s} | {'90% CI':>10s} | {'95% CI':>10s}")
print(f"  {'-'*42}")
 
coverage_results = []
 
for model_name in model_names:
    row = {"model": model_name}
    
    for level in CONFIDENCE_LEVELS:
        lo_col = f"{model_name}-lo-{level}"
        hi_col = f"{model_name}-hi-{level}"
        
        if lo_col in cv_results.columns and hi_col in cv_results.columns:
            y_true = cv_results["y"].values
            lo = cv_results[lo_col].values
            hi = cv_results[hi_col].values
            
            inside = ((y_true >= lo) & (y_true <= hi))
            cov = inside.mean()
            
            width = (hi - lo).mean()
            
            row[f"coverage_{level}"] = round(cov, 4)
            row[f"width_{level}"] = round(width, 4)
        else:
            row[f"coverage_{level}"] = None
            row[f"width_{level}"] = None
    
    coverage_results.append(row)
    
    cov90 = row.get('coverage_90', None)
    cov95 = row.get('coverage_95', None)
    c90_str = f"{cov90:.1%}" if cov90 is not None else "N/A"
    c95_str = f"{cov95:.1%}" if cov95 is not None else "N/A"
    print(f"  {model_name:15s} | {c90_str:>10s} | {c95_str:>10s}")
 
coverage_df = pd.DataFrame(coverage_results)
coverage_df.to_csv(f"{FIG_DIR}/sf_coverage.csv", index=False)
print(f"\n  [OK] sf_coverage.csv")
 
 
colors_sf = {
    "AutoARIMA": "#2980b9",
    "AutoETS": "#e74c3c",
    "CES": "#27ae60",
    "AutoTheta": "#f39c12",
}
 
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()
 
for idx, model_name in enumerate(model_names):
    ax = axes[idx]
    
    ax.plot(cv_results["ds"], cv_results["y"],
            color='black', linewidth=2, label='Факт', zorder=5)
    if model_name in cv_results.columns:
        ax.plot(cv_results["ds"], cv_results[model_name],
                color=colors_sf.get(model_name, '#888888'),
                linewidth=1.3, alpha=0.8, label=f'{model_name}')
    
    lo95 = f"{model_name}-lo-95"
    hi95 = f"{model_name}-hi-95"
    if lo95 in cv_results.columns and hi95 in cv_results.columns:
        ax.fill_between(cv_results["ds"],
                        cv_results[lo95], cv_results[hi95],
                        color=colors_sf.get(model_name, '#888888'),
                        alpha=0.15, label='95% CI')
        
        y_true = cv_results["y"].values
        inside = (y_true >= cv_results[lo95].values) & (y_true <= cv_results[hi95].values)
        cov = inside.mean()
        width = (cv_results[hi95] - cv_results[lo95]).mean()
        
        ax.text(0.02, 0.95,
                f'Coverage: {cov:.1%}\nШирина CI: {width:.2f} п.п.',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(f'{model_name}')
    ax.set_ylabel('% годовых')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
 
plt.suptitle('StatsForecast: прогнозы с 95% доверительными интервалами',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/sf_04_prediction_intervals_all.png')
plt.close()
print("  [OK] sf_04_prediction_intervals_all.png")


print("\n" + "=" * 60)
print("БЛОК 4: МЕТРИКИ И СРАВНЕНИЕ С BASELINE")
print("=" * 60)

def calc_metrics(y_true, y_pred, name):
    """Рассчитывает MAE, RMSE, MAPE для пары (факт, прогноз)."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    mape = np.mean(np.abs((yt - yp) / yt)) * 100 if np.all(yt != 0) else np.nan
    return {"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE_%": round(mape, 2), "n_forecasts": int(mask.sum())}

sf_metrics = []
y_true = cv_results["y"].values

for model_name in model_names:
    col_name = model_name
    if col_name not in cv_results.columns:
        for c in cv_results.columns:
            if model_name.lower() in c.lower():
                col_name = c
                break
    
    if col_name in cv_results.columns:
        y_pred = cv_results[col_name].values
        metrics = calc_metrics(y_true, y_pred, model_name)
        sf_metrics.append(metrics)
        print(f"  {model_name}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, MAPE={metrics['MAPE_%']:.2f}%")
    else:
        print(f"  [!] Столбец '{model_name}' не найден в результатах")
        print(f"      Доступные столбцы: {list(cv_results.columns)}")

sf_metrics_df = pd.DataFrame(sf_metrics)

baseline_path = f"{FIG_DIR}/baseline_metrics.csv"
if os.path.exists(baseline_path):
    baseline_df = pd.read_csv(baseline_path)
    all_metrics = pd.concat([baseline_df, sf_metrics_df], ignore_index=True)
else:
    print(f"\n  [!] Файл {baseline_path} не найден — сравнение только StatsForecast")
    all_metrics = sf_metrics_df

all_metrics = all_metrics.sort_values("MAE").reset_index(drop=True)

print(f"\n  СВОДНАЯ ТАБЛИЦА (baseline + StatsForecast):")
print("  " + "=" * 60)
print(f"  {'#':>3s} {'Модель':15s} | {'MAE':>8s} | {'RMSE':>8s} | {'MAPE %':>8s} | {'N'}")
print("  " + "-" * 60)
for i, row in all_metrics.iterrows():
    marker = ""
    if os.path.exists(baseline_path):
        naive_mae = baseline_df[baseline_df['model'] == 'Naive']['MAE'].values
        if len(naive_mae) > 0 and row['MAE'] < naive_mae[0]:
            marker = " ✓ бьёт Naive"
    print(f"  {i+1:3d} {row['model']:15s} | {row['MAE']:8.4f} | {row['RMSE']:8.4f} | "
          f"{row['MAPE_%']:7.2f}% | {int(row['n_forecasts'])}{marker}")

all_metrics.to_csv(f"{FIG_DIR}/metrics_baseline_sf.csv", index=False)
print(f"\n  [OK] metrics_baseline_sf.csv")

sf_metrics_df.to_csv(f"{FIG_DIR}/sf_metrics.csv", index=False)


print("\n" + "=" * 60)
print("БЛОК 5: ВИЗУАЛИЗАЦИЯ")
print("=" * 60)

colors_sf = {
    "AutoARIMA": "#2980b9",
    "ETS": "#e74c3c",
    "CES": "#27ae60",
    "Theta": "#f39c12",
}
fig, ax = plt.subplots(figsize=(16, 7))

ax.plot(cv_results["ds"], cv_results["y"],
        color='black', linewidth=2, label='Факт', zorder=5)

for model_name in model_names:
    if model_name in cv_results.columns:
        ax.plot(cv_results["ds"], cv_results[model_name],
                color=colors_sf.get(model_name, '#888888'),
                linewidth=1.2, alpha=0.8, label=model_name)

ax.set_title(f'StatsForecast: прогнозы vs факт (expanding window, h={HORIZON})')
ax.set_ylabel('Ключевая ставка, % годовых')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/sf_01_forecasts.png')
plt.close()
print("  [OK] sf_01_forecasts.png")

n_models = len([m for m in model_names if m in cv_results.columns])
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, model_name in enumerate(model_names):
    if model_name not in cv_results.columns or idx >= len(axes):
        continue
    ax = axes[idx]
    errors = cv_results["y"] - cv_results[model_name]

    ax.bar(cv_results["ds"], errors,
           color=colors_sf.get(model_name, '#888888'), alpha=0.7, width=25)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title(f'{model_name} — ошибки прогноза')
    ax.set_ylabel('Ошибка (п.п.)')
    ax.grid(True, alpha=0.3)

    mae = np.mean(np.abs(errors.dropna()))
    bias = errors.mean()
    ax.text(0.02, 0.95, f'MAE={mae:.3f}, bias={bias:+.3f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

for idx in range(n_models, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle(f'Ошибки прогнозов StatsForecast (h={HORIZON})', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/sf_02_errors.png')
plt.close()
print("  [OK] sf_02_errors.png")


fig, axes = plt.subplots(1, 3, figsize=(16, 6))
metrics_to_plot = ["MAE", "RMSE", "MAPE_%"]
metric_labels = ["MAE (п.п.)", "RMSE (п.п.)", "MAPE (%)"]

all_colors = {
    "Naive": "#bdc3c7", "Drift": "#bdc3c7", "ARIMA": "#95a5a6", "Decomp+Trend": "#95a5a6",
    **colors_sf
}

for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
    models_sorted = all_metrics.sort_values(metric)
    model_list = models_sorted["model"].values
    values = models_sorted[metric].values

    bar_colors = [all_colors.get(m, '#888888') for m in model_list]
    bars = ax.barh(model_list, values, color=bar_colors, alpha=0.85)
    ax.set_xlabel(label)
    ax.set_title(label)
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        fmt = f'{val:.3f}' if metric != "MAPE_%" else f'{val:.1f}%'
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                fmt, va='center', fontsize=9)

plt.suptitle(f'Baseline + StatsForecast (h={HORIZON})', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/sf_03_comparison_all.png')
plt.close()
print("  [OK] sf_03_comparison_all.png")

sf_preds = cv_results[["ds", "y"]].copy()
sf_preds = sf_preds.rename(columns={"y": "y_true"})
for model_name in model_names:
    if model_name in cv_results.columns:
        sf_preds[f"pred_{model_name}"] = cv_results[model_name].values

sf_preds.to_csv(f"{FIG_DIR}/sf_all_predictions.csv", index=False)
print("  [OK] sf_all_predictions.csv")


print("\n" + "=" * 60)
print("ИТОГОВАЯ СВОДКА")
print("=" * 60)

best = all_metrics.iloc[0]
best_sf = sf_metrics_df.sort_values("MAE").iloc[0] if len(sf_metrics_df) > 0 else None

naive_beaten = False
if os.path.exists(baseline_path):
    naive_mae = baseline_df[baseline_df['model'] == 'Naive']['MAE'].values
    if len(naive_mae) > 0:
        naive_mae = naive_mae[0]
        sf_best_mae = best_sf['MAE'] if best_sf is not None else float('inf')
        naive_beaten = sf_best_mae < naive_mae

print(f"""
ПОЛНАЯ ТАБЛИЦА МЕТРИК:

{all_metrics.to_string(index=False)}

ЛУЧШАЯ модель общая:       {best['model']} (MAE = {best['MAE']:.4f})
ЛУЧШАЯ модель StatsForecast: {best_sf['model'] if best_sf is not None else 'N/A'} (MAE = {best_sf['MAE'] if best_sf is not None else 0:.4f})
Naive baseline побит:       {'ДА' if naive_beaten else 'НЕТ — Naive всё ещё лучший'}
""")

# Список файлов
print("Сохранённые файлы:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith('sf_') or f == 'metrics_baseline_sf.csv' or f == 'nixtla_key_rate.csv':
        path = os.path.join(FIG_DIR, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {path} ({size_kb:.0f} KB)")

        
# Условие: Факт находится МЕЖДУ 90% и 95% границами (снизу или сверху)
grey_zone = cv_results[
    ((cv_results['y'] >= cv_results[f'{model_name}-lo-95']) & 
     (cv_results['y'] < cv_results[f'{model_name}-lo-90'])) 
    |
    ((cv_results['y'] > cv_results[f'{model_name}-hi-90']) & 
     (cv_results['y'] <= cv_results[f'{model_name}-hi-95']))
]

print("Месяцы, попавшие между 90% и 95% интервалами:")
print(grey_zone[['ds', 'y', model_name]])
