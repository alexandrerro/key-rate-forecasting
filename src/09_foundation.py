import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import time
import warnings
warnings.filterwarnings('ignore')


from paths import DATA_DIR, FIG_DIR
DATA_PATH = DATA_DIR / "dataset_monthly.csv"
os.makedirs(FIG_DIR, exist_ok=True)

MIN_TRAIN_SIZE = 60
HORIZON = 1

TIMEGPT_API_KEY = os.environ["NIXTLA_API_KEY"]

plt.rcParams.update({
    'figure.figsize': (14, 6), 'figure.dpi': 120,
    'savefig.dpi': 150, 'savefig.bbox': 'tight',
})



HAS_TIMEGPT = False
try:
    from nixtla import NixtlaClient
    if TIMEGPT_API_KEY:
        HAS_TIMEGPT = True
        print("[OK] TimeGPT (nixtla + API key)")
    else:
        print("    Вставьте в переменную TIMEGPT_API_KEY в начале скрипта")
except ImportError:
    print("[!] nixtla не установлена: pip install nixtla")

HAS_CHRONOS = False
try:
    import torch
    from chronos import ChronosPipeline
    HAS_CHRONOS = True
    print("[OK] Chronos-Bolt")
except ImportError:
    print("[!] Chronos не установлен: pip install chronos-forecasting torch")

if not HAS_TIMEGPT and not HAS_CHRONOS:
    print("\n[!] Ни одна foundation-модель не доступна.")
    print("\n    Скрипт продолжит работу с тем, что доступно.")



print("\n" + "=" * 60)
print("БЛОК 1: ПОДГОТОВКА ДАННЫХ")
print("=" * 60)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

TARGET = "key_rate"

nixtla_df = df[["date", TARGET]].dropna().copy()
nixtla_df = nixtla_df.rename(columns={"date": "ds", TARGET: "y"})
nixtla_df["unique_id"] = "key_rate"
nixtla_df = nixtla_df[["unique_id", "ds", "y"]].sort_values("ds").reset_index(drop=True)
nixtla_df["ds"] = pd.to_datetime(nixtla_df["ds"])

n_total = len(nixtla_df)
n_val_steps = n_total - MIN_TRAIN_SIZE - HORIZON + 1

print(f"Наблюдений: {n_total}")
print(f"Шагов валидации: {n_val_steps}")

y_values = nixtla_df["y"].values
y_dates = nixtla_df["ds"].values


preds_timegpt = []

if HAS_TIMEGPT:
    print("\n" + "=" * 60)
    print("БЛОК 2: TimeGPT (Nixtla API)")
    print("=" * 60)

    client = NixtlaClient(api_key=TIMEGPT_API_KEY)

    try:
        client.validate_api_key()
        print("  API ключ валиден")
    except Exception as e:
        print(f"  [!] API ключ невалиден: {e}")
        HAS_TIMEGPT = False

if HAS_TIMEGPT:
    print(f"\n  Expanding window CV ({n_val_steps} шагов)...")
    start_time = time.time()

    for i in range(n_val_steps):
        t = MIN_TRAIN_SIZE + i
        train = nixtla_df.iloc[:t].copy()
        actual = nixtla_df.iloc[t]["y"]
        actual_date = nixtla_df.iloc[t]["ds"]

        try:
            forecast = client.forecast(
                df=train,
                h=HORIZON,
                freq="MS",
                model="timegpt-1",
            )
            pred = forecast["TimeGPT"].values[-1]
        except Exception as e:
            pred = train["y"].iloc[-1]  # fallback

        preds_timegpt.append({
            "ds": actual_date,
            "y_true": actual,
            "pred": pred,
        })

        if (i + 1) % 20 == 0 or (i + 1) == n_val_steps:
            elapsed = time.time() - start_time
            print(f"    [{i+1}/{n_val_steps}] elapsed: {elapsed:.0f}s")

    elapsed = time.time() - start_time
    print(f"  TimeGPT завершён за {elapsed:.0f}s")
else:
    print("\n  [SKIP] TimeGPT пропущен (нет API ключа или библиотеки)")


preds_chronos = []

if HAS_CHRONOS:
    print("\n" + "=" * 60)
    print("БЛОК 3: Chronos-Bolt (локальный)")
    print("=" * 60)
    
    MODEL_NAME = "amazon/chronos-bolt-small"
    print(f"  Загрузка модели: {MODEL_NAME}")
    print(f"  (при первом запуске скачивается ~150MB)")

    try:
        pipeline = ChronosPipeline.from_pretrained(
            MODEL_NAME,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        print(f"  Модель загружена")

        print(f"\n  Expanding window CV ({n_val_steps} шагов)...")
        start_time = time.time()

        for i in range(n_val_steps):
            t = MIN_TRAIN_SIZE + i
            train_values = y_values[:t]
            actual = y_values[t]
            actual_date = y_dates[t]

            try:
                # Chronos принимает tensor
                context = torch.tensor(train_values, dtype=torch.float32).unsqueeze(0)
                forecast = pipeline.predict(context, prediction_length=HORIZON)
                # forecast shape: (1, num_samples, horizon) — берём медиану
                pred = forecast.median(dim=1).values[0, -1].item()
            except Exception as e:
                pred = train_values[-1]

            preds_chronos.append({
                "ds": actual_date,
                "y_true": actual,
                "pred": pred,
            })

            if (i + 1) % 20 == 0 or (i + 1) == n_val_steps:
                elapsed = time.time() - start_time
                print(f"    [{i+1}/{n_val_steps}] elapsed: {elapsed:.0f}s")

        elapsed = time.time() - start_time
        print(f"  Chronos-Bolt завершён за {elapsed:.0f}s")

    except Exception as e:
        print(f"  [!] Ошибка загрузки Chronos: {e}")
        print(f"      Попробуйте: pip install chronos-forecasting torch")
        HAS_CHRONOS = False
else:
    print("\n  [SKIP] Chronos пропущен (не установлен)")


print("\n" + "=" * 60)
print("БЛОК 4: МЕТРИКИ")
print("=" * 60)

def calc_metrics(preds_list, name):
    if not preds_list:
        return None
    df_p = pd.DataFrame(preds_list)
    yt = df_p["y_true"].values
    yp = df_p["pred"].values
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) == 0:
        return None
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    mape = np.mean(np.abs((yt - yp) / yt)) * 100 if np.all(yt != 0) else np.nan
    return {"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE_%": round(mape, 2), "n_forecasts": int(mask.sum())}

fm_metrics = []

if preds_timegpt:
    m = calc_metrics(preds_timegpt, "TimeGPT")
    if m:
        fm_metrics.append(m)
        print(f"  TimeGPT:      MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}")

if preds_chronos:
    m = calc_metrics(preds_chronos, "Chronos-Bolt")
    if m:
        fm_metrics.append(m)
        print(f"  Chronos-Bolt: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}")

if not fm_metrics:
    print("  [!] Ни одна foundation-модель не дала результатов")
    print("      Установите TimeGPT или Chronos и перезапустите")

fm_metrics_df = pd.DataFrame(fm_metrics)

prev_path = f"{FIG_DIR}/metrics_all_with_ml.csv"
if os.path.exists(prev_path):
    prev_metrics = pd.read_csv(prev_path)
    all_metrics = pd.concat([prev_metrics, fm_metrics_df], ignore_index=True)
else:
    all_metrics = fm_metrics_df

all_metrics = all_metrics.sort_values("MAE").reset_index(drop=True)

print(f"\n  ПОЛНАЯ СВОДНАЯ ТАБЛИЦА:")
print("  " + "=" * 65)
print(f"  {'#':>3s} {'Модель':17s} | {'MAE':>8s} | {'RMSE':>8s} | {'MAPE %':>8s} | {'N':>3s}")
print("  " + "-" * 65)

naive_mae = all_metrics[all_metrics['model'] == 'Naive']['MAE'].values
naive_mae = naive_mae[0] if len(naive_mae) > 0 else None

for i, row in all_metrics.iterrows():
    marker = ""
    if naive_mae and row['MAE'] < naive_mae and row['model'] != 'Naive':
        marker = " ★"
    is_fm = row['model'] in ['TimeGPT', 'Chronos-Bolt']
    prefix = "→ " if is_fm else "  "
    print(f"{prefix}{i+1:3d} {row['model']:17s} | {row['MAE']:8.4f} | {row['RMSE']:8.4f} | "
          f"{row['MAPE_%']:7.2f}% | {int(row['n_forecasts']):3d}{marker}")

all_metrics.to_csv(f"{FIG_DIR}/metrics_all_with_foundation.csv", index=False)
fm_metrics_df.to_csv(f"{FIG_DIR}/foundation_metrics.csv", index=False)
print(f"\n  [OK] metrics_all_with_foundation.csv")



print("\n" + "=" * 60)
print("БЛОК 5: ВИЗУАЛИЗАЦИЯ")
print("=" * 60)

colors_fm = {"TimeGPT": "#e74c3c", "Chronos-Bolt": "#3498db"}

if preds_timegpt or preds_chronos:
    fig, ax = plt.subplots(figsize=(16, 7))

    ref_preds = preds_timegpt if preds_timegpt else preds_chronos
    ref_df = pd.DataFrame(ref_preds)

    ax.plot(ref_df["ds"], ref_df["y_true"],
            color='black', linewidth=2, label='Факт', zorder=5)

    if preds_timegpt:
        df_tg = pd.DataFrame(preds_timegpt)
        m_tg = calc_metrics(preds_timegpt, "TimeGPT")
        ax.plot(df_tg["ds"], df_tg["pred"],
                color='#e74c3c', linewidth=1.3, alpha=0.8,
                label=f'TimeGPT (MAE={m_tg["MAE"]:.3f})')

    if preds_chronos:
        df_ch = pd.DataFrame(preds_chronos)
        m_ch = calc_metrics(preds_chronos, "Chronos-Bolt")
        ax.plot(df_ch["ds"], df_ch["pred"],
                color='#3498db', linewidth=1.3, alpha=0.8,
                label=f'Chronos-Bolt (MAE={m_ch["MAE"]:.3f})')

    ax.set_title(f'Foundation Models: zero-shot прогноз (h={HORIZON})')
    ax.set_ylabel('Ключевая ставка, % годовых')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fm_01_forecasts.png')
    plt.close()
    print("  [OK] fm_01_forecasts.png")

if len(all_metrics) > 1:
    fig, axes = plt.subplots(1, 3, figsize=(20, max(9, len(all_metrics) * 0.5)))
    metrics_to_plot = ["MAE", "RMSE", "MAPE_%"]
    metric_labels = ["MAE (п.п.)", "RMSE (п.п.)", "MAPE (%)"]

    all_colors = {
        "Naive": "#bdc3c7", "Drift": "#bdc3c7", "ARIMA": "#95a5a6",
        "Decomp+Trend": "#95a5a6", "AutoARIMA": "#2980b9", "AutoETS": "#3498db",
        "CES": "#1abc9c", "AutoTheta": "#f39c12", "Prophet-base": "#8e44ad",
        "Prophet-exog": "#e67e22", "LightGBM": "#2ecc71", "XGBoost": "#27ae60",
        "NBEATS": "#c0392b", "NHITS": "#16a085", "PatchTST": "#8e44ad",
        **colors_fm,
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

    plt.suptitle(f'Все модели включая Foundation (h={HORIZON})', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fm_02_comparison_all.png')
    plt.close()
    print("  [OK] fm_02_comparison_all.png")

all_fm_preds = pd.DataFrame()
if preds_timegpt:
    df_tg = pd.DataFrame(preds_timegpt)
    all_fm_preds["ds"] = df_tg["ds"]
    all_fm_preds["y_true"] = df_tg["y_true"]
    all_fm_preds["pred_TimeGPT"] = df_tg["pred"]
if preds_chronos:
    df_ch = pd.DataFrame(preds_chronos)
    if all_fm_preds.empty:
        all_fm_preds["ds"] = df_ch["ds"]
        all_fm_preds["y_true"] = df_ch["y_true"]
    all_fm_preds["pred_Chronos"] = df_ch["pred"]

if not all_fm_preds.empty:
    all_fm_preds.to_csv(f"{FIG_DIR}/fm_all_predictions.csv", index=False)
    print("  [OK] fm_all_predictions.csv")


print("\n" + "=" * 60)
print("ИТОГОВАЯ СВОДКА")
print("=" * 60)

n_models = len(all_metrics)
best = all_metrics.iloc[0]

print(f"""
ПОЛНАЯ ТАБЛИЦА ({n_models} моделей):

{all_metrics.to_string(index=False)}

ЛУЧШАЯ модель: {best['model']} (MAE = {best['MAE']:.4f})

FOUNDATION MODELS:""")

if preds_timegpt:
    m = calc_metrics(preds_timegpt, "TimeGPT")
    print(f"  TimeGPT:      MAE = {m['MAE']:.4f} (zero-shot)")


print("Сохранённые файлы:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith('fm_') or f == 'metrics_all_with_foundation.csv':
        path = os.path.join(FIG_DIR, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {path} ({size_kb:.0f} KB)")
