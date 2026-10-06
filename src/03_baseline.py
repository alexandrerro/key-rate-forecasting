import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import os
import warnings
import time

warnings.filterwarnings('ignore')

# Попробуем импортировать pmdarima для AutoARIMA
try:
    from pmdarima import auto_arima
    HAS_PMDARIMA = True
except ImportError:
    print("[!] pmdarima не установлен. Установите: pip install pmdarima")
    print("    AutoARIMA будет заменён на ручной ARIMA(1,1,0).")
    from statsmodels.tsa.arima.model import ARIMA
    HAS_PMDARIMA = False



from paths import DATA_DIR, FIG_DIR
DATA_PATH = DATA_DIR / "dataset_monthly.csv"

# Папка для результатов
os.makedirs(FIG_DIR, exist_ok=True)

# Параметры валидации
MIN_TRAIN_SIZE = 60    # Минимальное окно обучения (60 мес = 5 лет)
HORIZON = 1            # Горизонт прогноза (1 = one-step-ahead)

plt.rcParams.update({
    'figure.figsize': (14, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})



print("=" * 60)
print("БЛОК 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
print("=" * 60)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.set_index("date").sort_index()

# Целевая переменная — ключевая ставка ЦБ РФ
TARGET = "key_rate"
y = df[TARGET].dropna()

print(f"Целевая переменная: {TARGET}")
print(f"Наблюдений: {len(y)}")
print(f"Период: {y.index.min():%Y-%m} — {y.index.max():%Y-%m}")
print(f"Минимальное окно обучения: {MIN_TRAIN_SIZE} месяцев")
print(f"Горизонт прогноза: {HORIZON} мес")
print(f"Количество шагов валидации: {len(y) - MIN_TRAIN_SIZE - HORIZON + 1}")


def calc_mae(y_true, y_pred):
    """Средняя абсолютная ошибка (в тех же единицах, что и ставка — п.п.)."""
    return np.mean(np.abs(y_true - y_pred))

def calc_rmse(y_true, y_pred):
    """Корень среднеквадратичной ошибки — штрафует большие ошибки сильнее."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calc_mape(y_true, y_pred):
    """Средняя абсолютная процентная ошибка (%). 
    Осторожно: не определена при y_true = 0."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100



print("\n" + "=" * 60)
print("БЛОК 2: EXPANDING WINDOW CROSS-VALIDATION")
print("=" * 60)

y_values = y.values
y_dates = y.index
n = len(y_values)


val_start = MIN_TRAIN_SIZE
val_end = n - HORIZON + 1

print(f"\nВалидация:")
print(f"  Первый прогноз для даты:  {y_dates[val_start]:%Y-%m}")
print(f"  Последний прогноз для даты: {y_dates[val_end - 1]:%Y-%m}")
print(f"  Всего шагов валидации: {val_end - val_start}")


forecasts = {
    "Naive": [],
    "Drift": [],
    "ARIMA": [],
    "Decomp+Trend": [],
}



print("\n" + "=" * 60)
print("БЛОК 3: НАИВНЫЕ МОДЕЛИ (Naive, Drift)")
print("=" * 60)


print("\n  Naive forecast (ŷ = y_last)...")
for t in range(val_start, val_end):
    train = y_values[:t]
    actual = y_values[t]
    pred = train[-1]  # последнее значение
    forecasts["Naive"].append({
        "date": y_dates[t],
        "actual": actual,
        "pred": pred,
    })


print("  Drift forecast (линейная экстраполяция)...")
for t in range(val_start, val_end):
    train = y_values[:t]
    actual = y_values[t]
    n_train = len(train)
    # Дрифт = среднее изменение за период
    drift = (train[-1] - train[0]) / (n_train - 1) if n_train > 1 else 0
    pred = train[-1] + HORIZON * drift
    forecasts["Drift"].append({
        "date": y_dates[t],
        "actual": actual,
        "pred": pred,
    })

print(f"  Наивные модели: {len(forecasts['Naive'])} прогнозов каждая")




print("\n" + "=" * 60)
print("БЛОК 4: ARIMA (автоматический подбор параметров)")
print("=" * 60)


print("\n  Это самый медленный этап — ARIMA переобучается на каждом шаге.")
print(f"  Всего шагов: {val_end - val_start}")

arima_start_time = time.time()
arima_orders = []  # Сохраняем выбранные (p,d,q) для анализа

for i, t in enumerate(range(val_start, val_end)):
    train = y_values[:t]
    actual = y_values[t]

    try:
        if HAS_PMDARIMA:
            model = auto_arima(
                train,
                start_p=0, max_p=5,
                start_q=0, max_q=5,
                d=None,           
                max_d=2,
                seasonal=False,   # нет сезонности 
                stepwise=True,    # быстрый пошаговый поиск
                suppress_warnings=True,
                error_action='ignore',
            )
            pred = model.predict(n_periods=HORIZON)[-1]
            order = model.order
        else:
            model = ARIMA(train, order=(1, 1, 0))
            fit = model.fit()
            pred = fit.forecast(steps=HORIZON)[-1]
            order = (1, 1, 0)

        arima_orders.append(order)

    except Exception as e:
        pred = train[-1]
        order = None

    forecasts["ARIMA"].append({
        "date": y_dates[t],
        "actual": actual,
        "pred": pred,
    })

    if (i + 1) % 20 == 0 or (i + 1) == (val_end - val_start):
        elapsed = time.time() - arima_start_time
        print(f"    [{i+1}/{val_end - val_start}] "
              f"elapsed: {elapsed:.0f}s, "
              f"last order: {order}")

if arima_orders:
    valid_orders = [o for o in arima_orders if o is not None]
    if valid_orders:
        from collections import Counter
        order_counts = Counter(valid_orders)
        most_common = order_counts.most_common(3)
        print(f"\n  Наиболее частые порядки ARIMA:")
        for order, count in most_common:
            print(f"    ARIMA{order}: {count} раз ({count/len(valid_orders)*100:.0f}%)")


print("\n" + "=" * 60)
print("БЛОК 5: ДЕКОМПОЗИЦИЯ + ЭКСТРАПОЛЯЦИЯ ТРЕНДА")
print("=" * 60)


print("\n  STL-декомпозиция + линейная экстраполяция тренда...")

for i, t in enumerate(range(val_start, val_end)):
    train = y_values[:t]
    actual = y_values[t]

    try:
        train_series = pd.Series(train, index=y_dates[:t])
        stl = STL(train_series, period=12, robust=True)
        result = stl.fit()

        trend = result.trend.values
        n_trend = min(12, len(trend))
        x = np.arange(n_trend)
        y_trend = trend[-n_trend:]

        coeffs = np.polyfit(x, y_trend, deg=1)
        pred = np.polyval(coeffs, n_trend + HORIZON - 1)

    except Exception as e:
        pred = train[-1]

    forecasts["Decomp+Trend"].append({
        "date": y_dates[t],
        "actual": actual,
        "pred": pred,
    })

print(f"  Декомпозиция: {len(forecasts['Decomp+Trend'])} прогнозов")



print("\n" + "=" * 60)
print("БЛОК 6: МЕТРИКИ И СРАВНЕНИЕ")
print("=" * 60)

results_summary = []

for model_name, preds in forecasts.items():
    pred_df = pd.DataFrame(preds)
    y_true = pred_df["actual"].values
    y_pred = pred_df["pred"].values

    mae = calc_mae(y_true, y_pred)
    rmse = calc_rmse(y_true, y_pred)
    mape = calc_mape(y_true, y_pred)

    results_summary.append({
        "model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE_%": round(mape, 2),
        "n_forecasts": len(preds),
    })

results_df = pd.DataFrame(results_summary).sort_values("MAE")

print("\n  СВОДНАЯ ТАБЛИЦА МЕТРИК (expanding window, h=1)")
print("  " + "=" * 55)
print(f"  {'Модель':15s} | {'MAE':>8s} | {'RMSE':>8s} | {'MAPE %':>8s} | {'N'}")
print("  " + "-" * 55)
for _, row in results_df.iterrows():
    print(f"  {row['model']:15s} | {row['MAE']:8.4f} | {row['RMSE']:8.4f} | "
          f"{row['MAPE_%']:7.2f}% | {row['n_forecasts']}")

print(f"\n  MAE показывает среднюю ошибку прогноза в процентных пунктах.")
print(f"  Например, MAE = 0.50 означает, что в среднем прогноз")
print(f"  отклоняется от факта на 0.5 п.п. (ставка 8% vs прогноз 8.5%).")

results_df.to_csv(f"{FIG_DIR}/baseline_metrics.csv", index=False)
print(f"\n  [OK] baseline_metrics.csv")



print("\n" + "=" * 60)
print("БЛОК 7: ВИЗУАЛИЗАЦИЯ")
print("=" * 60)


fig, ax = plt.subplots(figsize=(16, 7))

# Фактические значения
pred_df = pd.DataFrame(forecasts["Naive"])
ax.plot(pred_df["date"], pred_df["actual"],
        color='black', linewidth=2, label='Факт (key_rate)', zorder=5)

# Прогнозы каждой модели
colors = {"Naive": "#e74c3c", "Drift": "#f39c12",
          "ARIMA": "#2980b9", "Decomp+Trend": "#27ae60"}
for model_name, preds in forecasts.items():
    pred_df = pd.DataFrame(preds)
    ax.plot(pred_df["date"], pred_df["pred"],
            color=colors[model_name], linewidth=1.2,
            alpha=0.8, label=f'{model_name}')

ax.set_title(f'Baseline: прогнозы vs факт (expanding window, h={HORIZON})')
ax.set_ylabel('Ключевая ставка, % годовых')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/baseline_01_forecasts.png')
plt.close()
print("  [OK] baseline_01_forecasts.png")


fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, (model_name, preds) in enumerate(forecasts.items()):
    ax = axes[idx]
    pred_df = pd.DataFrame(preds)
    errors = pred_df["actual"] - pred_df["pred"]

    ax.bar(pred_df["date"], errors, color=colors[model_name], alpha=0.7, width=25)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title(f'{model_name} — ошибки прогноза')
    ax.set_ylabel('Ошибка (факт - прогноз), п.п.')
    ax.grid(True, alpha=0.3)

    mae = calc_mae(pred_df["actual"].values, pred_df["pred"].values)
    bias = errors.mean()
    ax.text(0.02, 0.95, f'MAE={mae:.3f}, bias={bias:+.3f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle(f'Ошибки прогнозов baseline-моделей (h={HORIZON})', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/baseline_02_errors.png')
plt.close()
print("  [OK] baseline_02_errors.png")


fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics_to_plot = ["MAE", "RMSE", "MAPE_%"]
metric_labels = ["MAE (п.п.)", "RMSE (п.п.)", "MAPE (%)"]

for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
    models = results_df["model"].values
    values = results_df[metric].values

    bar_colors = [colors.get(m, '#888888') for m in models]
    bars = ax.barh(models, values, color=bar_colors, alpha=0.8)
    ax.set_xlabel(label)
    ax.set_title(label)
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}' if metric != "MAPE_%" else f'{val:.1f}%',
                va='center', fontsize=10)

plt.suptitle(f'Сравнение baseline-моделей (h={HORIZON})', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/baseline_03_comparison.png')
plt.close()
print("  [OK] baseline_03_comparison.png")


all_preds = pd.DataFrame(forecasts["Naive"])[["date", "actual"]].copy()
all_preds = all_preds.rename(columns={"actual": "y_true"})
for model_name, preds in forecasts.items():
    pred_df = pd.DataFrame(preds)
    all_preds[f"pred_{model_name}"] = pred_df["pred"].values

all_preds.to_csv(f"{FIG_DIR}/baseline_all_predictions.csv", index=False)
print("  [OK] baseline_all_predictions.csv")



print("\n" + "=" * 60)
print("ИТОГОВАЯ СВОДКА")
print("=" * 60)

best_model = results_df.iloc[0]
worst_model = results_df.iloc[-1]

print(f"""
РЕЗУЛЬТАТЫ BASELINE (expanding window, h={HORIZON}):

{results_df.to_string(index=False)}

ЛУЧШАЯ модель по MAE: {best_model['model']} (MAE = {best_model['MAE']:.4f} п.п.)
ХУДШАЯ модель по MAE: {worst_model['model']} (MAE = {worst_model['MAE']:.4f} п.п.)
""")

print("Сохранённые файлы:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith('baseline_'):
        path = os.path.join(FIG_DIR, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {path} ({size_kb:.0f} KB)")
