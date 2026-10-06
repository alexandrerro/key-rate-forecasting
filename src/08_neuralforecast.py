
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import time
import warnings

warnings.filterwarnings('ignore')

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS, NHITS, PatchTST
    print("[OK] NeuralForecast")
except ImportError as e:
    print(f"[!] Ошибка импорта: {e}")
    print("    Попробуйте: pip install neuralforecast")
    exit(1)


from paths import DATA_DIR, FIG_DIR
DATA_PATH = DATA_DIR / "dataset_monthly.csv"
os.makedirs(FIG_DIR, exist_ok=True)

MIN_TRAIN_SIZE = 60
HORIZON = 1

# Гиперпараметры нейросетей — адаптированы для КОРОТКОГО ряда

INPUT_SIZE = 12           # Размер входного окна (12 мес = 1 год)
MAX_STEPS = 300           # Максимум шагов обучения (мало — защита от переобучения)
LEARNING_RATE = 1e-3      # Скорость обучения
BATCH_SIZE = 32           # Размер батча (почти весь датасет)
RANDOM_SEED = 42

plt.rcParams.update({
    'figure.figsize': (14, 6), 'axes.titlesize': 14,
    'axes.labelsize': 12, 'figure.dpi': 120,
    'savefig.dpi': 150, 'savefig.bbox': 'tight',
})


print("\n" + "=" * 60)
print("БЛОК 1: ПОДГОТОВКА ДАННЫХ")
print("=" * 60)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# Формат Nixtla
nixtla_df = df[["date", "key_rate"]].dropna().copy()
nixtla_df = nixtla_df.rename(columns={"date": "ds", "key_rate": "y"})
nixtla_df["unique_id"] = "key_rate"
nixtla_df = nixtla_df[["unique_id", "ds", "y"]].sort_values("ds").reset_index(drop=True)
nixtla_df["ds"] = pd.to_datetime(nixtla_df["ds"])

n_total = len(nixtla_df)
n_windows = n_total - MIN_TRAIN_SIZE - HORIZON + 1

print(f"Наблюдений: {n_total}")
print(f"Период: {nixtla_df['ds'].min():%Y-%m} — {nixtla_df['ds'].max():%Y-%m}")
print(f"Шагов валидации: {n_windows}")
print(f"Input size: {INPUT_SIZE} мес, Horizon: {HORIZON} мес")
print(f"Max steps: {MAX_STEPS}, Batch size: {BATCH_SIZE}")


print("\n" + "=" * 60)
print("БЛОК 2: ОПРЕДЕЛЕНИЕ МОДЕЛЕЙ")
print("=" * 60)

models = [
    NBEATS(
        h=HORIZON,
        input_size=INPUT_SIZE,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        scaler_type='standard',
        random_seed=RANDOM_SEED,
        accelerator='cpu',
        enable_progress_bar=False,
        stack_types=['identity', 'identity', 'identity'],  # generic stacks вместо trend/seasonality
    ),

    NHITS(
        h=HORIZON,
        input_size=INPUT_SIZE,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        scaler_type='standard',
        random_seed=RANDOM_SEED,
        accelerator='cpu',
        enable_progress_bar=False,
    ),

    PatchTST(
        h=HORIZON,
        input_size=INPUT_SIZE,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        scaler_type='standard',
        random_seed=RANDOM_SEED,
        accelerator='cpu',
        enable_progress_bar=False,
        patch_len=4,
        stride=2,
        hidden_size=32,
        n_heads=4,
        encoder_layers=2,
    ),
]

model_names = ["NBEATS", "NHITS", "PatchTST"]
print(f"Модели: {', '.join(model_names)}")
print(f"Все модели: CPU, {MAX_STEPS} шагов, input_size={INPUT_SIZE}")


print("\n" + "=" * 60)
print("БЛОК 3: КРОСС-ВАЛИДАЦИЯ (expanding window)")
print("=" * 60)

nf = NeuralForecast(models=models, freq="MS")

print(f"\nЗапуск кросс-валидации ({n_windows} окон)...")
print(f"Это может занять 5-20 минут (нейросети обучаются на каждом окне)")
start_time = time.time()

try:
    cv_results = nf.cross_validation(
        df=nixtla_df,
        n_windows=n_windows,
        step_size=1,
    )
    elapsed = time.time() - start_time
    print(f"Готово за {elapsed:.0f}s ({elapsed/60:.1f} мин)")

except Exception as e:
    print(f"\n[!] Ошибка при кросс-валидации: {e}")
    print("    Попробуем запустить модели по одной...")

    cv_results = None
    for i, (model, name) in enumerate(zip(models, model_names)):
        print(f"\n  Запуск {name}...")
        try:
            nf_single = NeuralForecast(models=[model], freq="MS")
            cv_single = nf_single.cross_validation(
                df=nixtla_df,
                n_windows=n_windows,
                step_size=1,
            )
            if cv_results is None:
                cv_results = cv_single
            else:
                for col in cv_single.columns:
                    if col not in ["unique_id", "ds", "cutoff", "y"]:
                        cv_results[col] = cv_single[col].values
            print(f"  {name} OK ({time.time() - start_time:.0f}s)")
        except Exception as e2:
            print(f"  {name} FAILED: {e2}")

    elapsed = time.time() - start_time
    print(f"\nОбщее время: {elapsed:.0f}s")

if cv_results is None:
    print("\n[!] Ни одна нейросетевая модель не сработала.")
    print("    Проверьте установку: pip install neuralforecast pytorch-lightning")
    exit(1)

if 'unique_id' in cv_results.columns:
    cv_results = cv_results.drop(columns=['unique_id'])
if 'cutoff' in cv_results.columns:
    cv_results = cv_results.drop(columns=['cutoff'])

print(f"\nРезультат:")
print(f"  Shape: {cv_results.shape}")
print(f"  Столбцы: {list(cv_results.columns)}")
print(f"\nПоследние 5 строк:")
print(cv_results.tail().to_string(index=False))


print("\n" + "=" * 60)
print("БЛОК 4: МЕТРИКИ И СРАВНЕНИЕ")
print("=" * 60)

def calc_metrics(y_true, y_pred, name):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {"model": name, "MAE": np.nan, "RMSE": np.nan,
                "MAPE_%": np.nan, "n_forecasts": 0}
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    mape = np.mean(np.abs((yt - yp) / yt)) * 100 if np.all(yt != 0) else np.nan
    return {"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE_%": round(mape, 2), "n_forecasts": int(mask.sum())}

nn_metrics = []
y_true = cv_results["y"].values

for model_name in model_names:
    col = None
    for c in cv_results.columns:
        if model_name in c:
            col = c
            break

    if col and col in cv_results.columns:
        y_pred = cv_results[col].values
        m = calc_metrics(y_true, y_pred, model_name)
        nn_metrics.append(m)
        print(f"  {model_name}: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, MAPE={m['MAPE_%']:.2f}%")
    else:
        print(f"  [!] {model_name}: столбец не найден в результатах")

nn_metrics_df = pd.DataFrame(nn_metrics)

prev_path = f"{FIG_DIR}/metrics_all_with_ml.csv"
if os.path.exists(prev_path):
    prev_metrics = pd.read_csv(prev_path)
    all_metrics = pd.concat([prev_metrics, nn_metrics_df], ignore_index=True)
else:
    all_metrics = nn_metrics_df

all_metrics = all_metrics.sort_values("MAE").reset_index(drop=True)

print(f"\n  ПОЛНАЯ СВОДНАЯ ТАБЛИЦА (все модели, h={HORIZON}):")
print("  " + "=" * 65)
print(f"  {'#':>3s} {'Модель':17s} | {'MAE':>8s} | {'RMSE':>8s} | {'MAPE %':>8s} | {'N':>3s}")
print("  " + "-" * 65)

naive_mae = all_metrics[all_metrics['model'] == 'Naive']['MAE'].values
naive_mae = naive_mae[0] if len(naive_mae) > 0 else None

for i, row in all_metrics.iterrows():
    marker = ""
    if naive_mae and row['MAE'] < naive_mae and row['model'] != 'Naive':
        marker = " ★"
    print(f"  {i+1:3d} {row['model']:17s} | {row['MAE']:8.4f} | {row['RMSE']:8.4f} | "
          f"{row['MAPE_%']:7.2f}% | {int(row['n_forecasts']):3d}{marker}")

all_metrics.to_csv(f"{FIG_DIR}/metrics_all_with_nn.csv", index=False)
nn_metrics_df.to_csv(f"{FIG_DIR}/nn_metrics.csv", index=False)
print(f"\n  [OK] metrics_all_with_nn.csv")


print("\n" + "=" * 60)
print("БЛОК 5: ВИЗУАЛИЗАЦИЯ")
print("=" * 60)

colors_nn = {
    "NBEATS": "#e74c3c",
    "NHITS": "#2ecc71",
    "PatchTST": "#9b59b6",
}

fig, ax = plt.subplots(figsize=(16, 7))

ax.plot(cv_results["ds"], cv_results["y"],
        color='black', linewidth=2, label='Факт', zorder=5)

for model_name in model_names:
    for c in cv_results.columns:
        if model_name in c and c not in ["ds", "y"]:
            ax.plot(cv_results["ds"], cv_results[c],
                    color=colors_nn.get(model_name, '#888888'),
                    linewidth=1.2, alpha=0.8, label=model_name)
            break

ax.set_title(f'NeuralForecast: прогнозы vs факт (h={HORIZON})')
ax.set_ylabel('Ключевая ставка, % годовых')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/nn_01_forecasts.png')
plt.close()
print("  [OK] nn_01_forecasts.png")

n_nn_models = len([m for m in model_names if any(m in c for c in cv_results.columns)])
if n_nn_models > 0:
    fig, axes = plt.subplots(1, min(n_nn_models, 3), figsize=(6*min(n_nn_models, 3), 5))
    if n_nn_models == 1:
        axes = [axes]

    idx = 0
    for model_name in model_names:
        col = None
        for c in cv_results.columns:
            if model_name in c and c not in ["ds", "y"]:
                col = c
                break
        if col is None:
            continue

        ax = axes[idx]
        errors = cv_results["y"] - cv_results[col]
        ax.bar(cv_results["ds"], errors,
               color=colors_nn.get(model_name, '#888888'), alpha=0.7, width=25)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_title(f'{model_name} — ошибки')
        ax.set_ylabel('п.п.')
        ax.grid(True, alpha=0.3)

        mae = np.mean(np.abs(errors.dropna()))
        bias = errors.mean()
        ax.text(0.02, 0.95, f'MAE={mae:.3f}\nbias={bias:+.3f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        idx += 1

    plt.suptitle(f'Ошибки нейросетей (h={HORIZON})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/nn_02_errors.png')
    plt.close()
    print("  [OK] nn_02_errors.png")

fig, axes = plt.subplots(1, 3, figsize=(20, 9))
metrics_to_plot = ["MAE", "RMSE", "MAPE_%"]
metric_labels = ["MAE (п.п.)", "RMSE (п.п.)", "MAPE (%)"]

all_colors = {
    "Naive": "#bdc3c7", "Drift": "#bdc3c7", "ARIMA": "#95a5a6", "Decomp+Trend": "#95a5a6",
    "AutoARIMA": "#2980b9", "AutoETS": "#3498db", "CES": "#1abc9c", "AutoTheta": "#f39c12",
    "Prophet-base": "#8e44ad", "Prophet-exog": "#e67e22",
    "LightGBM": "#2ecc71", "XGBoost": "#27ae60",
    **colors_nn,
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
                fmt, va='center', fontsize=8)

plt.suptitle(f'Все модели: Baseline + StatsForecast + Prophet + ML + NeuralForecast (h={HORIZON})',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/nn_03_comparison_all.png')
plt.close()
print("  [OK] nn_03_comparison_all.png")

nn_preds = cv_results[["ds", "y"]].copy().rename(columns={"y": "y_true"})
for model_name in model_names:
    for c in cv_results.columns:
        if model_name in c and c not in ["ds", "y"]:
            nn_preds[f"pred_{model_name}"] = cv_results[c].values
            break

nn_preds.to_csv(f"{FIG_DIR}/nn_all_predictions.csv", index=False)
print("  [OK] nn_all_predictions.csv")


print("\n" + "=" * 60)
print("ИТОГОВАЯ СВОДКА СПРИНТА 3 (NeuralForecast)")
print("=" * 60)

best = all_metrics.iloc[0]
best_nn = nn_metrics_df.sort_values("MAE").iloc[0] if len(nn_metrics_df) > 0 else None

best_nn_model = best_nn['model'] if best_nn is not None else 'N/A'
best_nn_mae_text = f"{best_nn['MAE']:.4f}" if best_nn is not None else "N/A"
naive_mae_text = f"{naive_mae:.4f}" if naive_mae is not None else "N/A"

print(f"""
ПОЛНАЯ ТАБЛИЦА ({len(all_metrics)} моделей):

{all_metrics.to_string(index=False)}

ЛУЧШАЯ модель общая:        {best['model']} (MAE = {best['MAE']:.4f})
ЛУЧШАЯ нейросеть:           {best_nn['model'] if best_nn is not None else 'N/A'} (MAE = {best_nn['MAE']:.4f})
""")

# Список файлов
print("Сохранённые файлы:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith('nn_'):
        path = os.path.join(FIG_DIR, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {path} ({size_kb:.0f} KB)")