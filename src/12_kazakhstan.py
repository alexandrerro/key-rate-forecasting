
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, AutoCES
    HAS_SF = True
except ImportError:
    print("[!] statsforecast не установлен")
    HAS_SF = False

from paths import DATA_DIR, FIG_DIR
KZ_DATA = DATA_DIR / "kz_nixtla.csv"
RU_DATA = FIG_DIR / "nixtla_key_rate.csv"  # Данные РФ для сравнения
os.makedirs(FIG_DIR, exist_ok=True)

MIN_TRAIN_SIZE = 60
HORIZON = 1

plt.rcParams.update({
    'figure.figsize': (14, 6), 'figure.dpi': 120,
    'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

print("=" * 60)
print("ПРОВЕРКА РОБАСТНОСТИ: КАЗАХСТАН")
print("=" * 60)

kz_df = pd.read_csv(KZ_DATA, parse_dates=["ds"])
kz_df["unique_id"] = "kz_base_rate"
kz_df = kz_df.sort_values("ds").reset_index(drop=True)

print(f"\nКазахстан (НБРК):")
print(f"  Наблюдений: {len(kz_df)}")
print(f"  Период: {kz_df['ds'].min():%Y-%m} — {kz_df['ds'].max():%Y-%m}")
print(f"  Диапазон: {kz_df['y'].min()}% — {kz_df['y'].max()}%")
print(f"  Среднее: {kz_df['y'].mean():.2f}%, Медиана: {kz_df['y'].median():.1f}%")

n_kz = len(kz_df)
n_val_kz = n_kz - MIN_TRAIN_SIZE - HORIZON + 1
print(f"  Шагов валидации: {n_val_kz}")


print(f"\n{'='*60}")
print("БЛОК 2: NAIVE FORECAST")
print("=" * 60)

y_kz = kz_df["y"].values
dates_kz = kz_df["ds"].values

preds_naive_kz = []
for i in range(n_val_kz):
    t = MIN_TRAIN_SIZE + i
    actual = y_kz[t]
    pred = y_kz[t - 1]
    preds_naive_kz.append({"ds": dates_kz[t], "y_true": actual, "pred": pred})

df_naive_kz = pd.DataFrame(preds_naive_kz)
mae_naive_kz = np.mean(np.abs(df_naive_kz["y_true"] - df_naive_kz["pred"]))
rmse_naive_kz = np.sqrt(np.mean((df_naive_kz["y_true"] - df_naive_kz["pred"])**2))
print(f"  Naive КЗ: MAE = {mae_naive_kz:.4f}, RMSE = {rmse_naive_kz:.4f}")



sf_results_kz = []

if HAS_SF:
    print(f"\n{'='*60}")
    print("БЛОК 3: STATSFORECAST (AutoTheta, AutoETS, AutoARIMA)")
    print("=" * 60)

    # Те же модели и параметры, что для РФ — БЕЗ переобучения
    models = [
        AutoARIMA(season_length=1),
        AutoETS(season_length=1),
        AutoCES(season_length=1),
        AutoTheta(season_length=1),
    ]

    sf = StatsForecast(models=models, freq="MS", n_jobs=1)

    print(f"  Запуск кросс-валидации ({n_val_kz} окон)...")
    start_time = time.time()

    cv_kz = sf.cross_validation(
        df=kz_df,
        h=HORIZON,
        step_size=1,
        n_windows=n_val_kz,
    )

    elapsed = time.time() - start_time
    print(f"  Готово за {elapsed:.1f}s")

    # Убираем лишние столбцы
    if 'unique_id' in cv_kz.columns:
        cv_kz = cv_kz.drop(columns=['unique_id'])
    if 'cutoff' in cv_kz.columns:
        cv_kz = cv_kz.drop(columns=['cutoff'])

    print(f"  Столбцы: {list(cv_kz.columns)}")

    # Метрики
    y_true_kz = cv_kz["y"].values
    model_names_sf = [c for c in cv_kz.columns if c not in ["ds", "y"]]

    for model_name in model_names_sf:
        y_pred = cv_kz[model_name].values
        mask = np.isfinite(y_true_kz) & np.isfinite(y_pred)
        yt, yp = y_true_kz[mask], y_pred[mask]
        mae = np.mean(np.abs(yt - yp))
        rmse = np.sqrt(np.mean((yt - yp) ** 2))
        mape = np.mean(np.abs((yt - yp) / yt)) * 100
        sf_results_kz.append({
            "model": model_name, "MAE_KZ": round(mae, 4),
            "RMSE_KZ": round(rmse, 4), "MAPE_KZ": round(mape, 2),
            "n_kz": int(mask.sum()),
        })
        print(f"    {model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}")
else:
    print("\n  [SKIP] StatsForecast не установлен")


print(f"\n{'='*60}")
print("БЛОК 4: СРАВНЕНИЕ РФ vs КАЗАХСТАН")
print("=" * 60)

# Метрики РФ (из предыдущих запусков)
ru_metrics = {
    "Naive":     {"MAE_RU": 0.5687, "RMSE_RU": 1.5634},
    "AutoTheta": {"MAE_RU": 0.5824, "RMSE_RU": 1.5683},
    "AutoETS":   {"MAE_RU": 0.6610, "RMSE_RU": 1.6655},
    "AutoARIMA": {"MAE_RU": 0.7174, "RMSE_RU": 1.6376},
    "CES":       {"MAE_RU": 0.7284, "RMSE_RU": 1.6764},
}

# Собираем сводную таблицу
comparison = []

# Naive
comparison.append({
    "model": "Naive",
    "MAE_RU": 0.5687, "RMSE_RU": 1.5634,
    "MAE_KZ": round(mae_naive_kz, 4), "RMSE_KZ": round(rmse_naive_kz, 4),
})

# SF модели
for row in sf_results_kz:
    model = row["model"]
    if model in ru_metrics:
        comparison.append({
            "model": model,
            "MAE_RU": ru_metrics[model]["MAE_RU"],
            "RMSE_RU": ru_metrics[model]["RMSE_RU"],
            "MAE_KZ": row["MAE_KZ"],
            "RMSE_KZ": row["RMSE_KZ"],
        })

comp_df = pd.DataFrame(comparison)

# Добавляем столбец с разницей
comp_df["MAE_diff"] = comp_df["MAE_KZ"] - comp_df["MAE_RU"]
comp_df["MAE_diff_%"] = ((comp_df["MAE_KZ"] / comp_df["MAE_RU"]) - 1) * 100

print(f"\n  {'Модель':15s} | {'MAE РФ':>8s} | {'MAE КЗ':>8s} | {'Разница':>8s} | {'%':>7s} | Вывод")
print(f"  {'-'*70}")
for _, row in comp_df.iterrows():
    if row['MAE_diff'] > 0:
        verdict = "КЗ хуже"
    elif row['MAE_diff'] < 0:
        verdict = "КЗ лучше"
    else:
        verdict = "равны"
    print(f"  {row['model']:15s} | {row['MAE_RU']:8.4f} | {row['MAE_KZ']:8.4f} | "
          f"{row['MAE_diff']:+8.4f} | {row['MAE_diff_%']:+6.1f}% | {verdict}")

comp_df.to_csv(f"{FIG_DIR}/kz_comparison_ru_vs_kz.csv", index=False)
print(f"\n  [OK] kz_comparison_ru_vs_kz.csv")


print(f"\n{'='*60}")
print("БЛОК 5: ВИЗУАЛИЗАЦИЯ")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# РФ
if os.path.exists(RU_DATA):
    ru_df = pd.read_csv(RU_DATA, parse_dates=["ds"])
    axes[0].plot(ru_df["ds"], ru_df["y"], color='#2980b9', linewidth=2)
    axes[0].set_title(f'Ключевая ставка ЦБ РФ\n({len(ru_df)} мес, {ru_df["y"].min():.1f}%–{ru_df["y"].max():.1f}%)')
else:
    axes[0].set_title('Ключевая ставка ЦБ РФ (нет данных)')
axes[0].set_ylabel('% годовых')
axes[0].grid(True, alpha=0.3)

# КЗ
kz_monthly = pd.read_csv(DATA_DIR / "kz_base_rate_monthly.csv", parse_dates=["date"])
axes[1].plot(kz_monthly["date"], kz_monthly["base_rate"], color='#e74c3c', linewidth=2)
axes[1].set_title(f'Базовая ставка НБРК\n({len(kz_monthly)} мес, {kz_monthly["base_rate"].min():.1f}%–{kz_monthly["base_rate"].max():.1f}%)')
axes[1].set_ylabel('% годовых')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Сравнение процентных ставок: Россия vs Казахстан', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/kz_01_rates_comparison.png')
plt.close()
print("  [OK] kz_01_rates_comparison.png")

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(comp_df))
width = 0.35

bars1 = ax.bar(x - width/2, comp_df["MAE_RU"], width, label='Россия (ЦБ РФ)',
               color='#2980b9', alpha=0.85)
bars2 = ax.bar(x + width/2, comp_df["MAE_KZ"], width, label='Казахстан (НБРК)',
               color='#e74c3c', alpha=0.85)

ax.set_ylabel('MAE (п.п.)')
ax.set_title('Сравнение MAE: Россия vs Казахстан (те же модели, те же параметры)')
ax.set_xticks(x)
ax.set_xticklabels(comp_df["model"].values)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars1, comp_df["MAE_RU"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', fontsize=9)
for bar, val in zip(bars2, comp_df["MAE_KZ"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/kz_02_mae_comparison.png')
plt.close()
print("  [OK] kz_02_mae_comparison.png")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_naive_kz["ds"], df_naive_kz["y_true"],
        color='black', linewidth=2, label='Факт (НБРК)')
ax.plot(df_naive_kz["ds"], df_naive_kz["pred"],
        color='#e74c3c', linewidth=1.2, alpha=0.8, label=f'Naive (MAE={mae_naive_kz:.3f})')
ax.set_title('Казахстан: Naive прогноз vs факт')
ax.set_ylabel('% годовых')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/kz_03_naive_forecast.png')
plt.close()
print("  [OK] kz_03_naive_forecast.png")


print(f"\n{'='*60}")
print("ИТОГОВАЯ СВОДКА: РОБАСТНОСТЬ")
print("=" * 60)

avg_diff_pct = comp_df["MAE_diff_%"].mean()
naive_ratio = comp_df[comp_df['model']=='Naive']['MAE_KZ'].values[0] / comp_df[comp_df['model']=='Naive']['MAE_RU'].values[0]

print(f"""
РЕЗУЛЬТАТЫ ПРОВЕРКИ РОБАСТНОСТИ:

{comp_df.to_string(index=False)}

Средняя разница MAE (КЗ vs РФ): {avg_diff_pct:+.1f}%
Naive ratio (КЗ/РФ): {naive_ratio:.2f}x

ИНТЕРПРЕТАЦИЯ:""")

if abs(avg_diff_pct) < 30:
    print(f"  → Модели РОБАСТНЫ: разница менее 30%, результаты переносятся")
    print(f"    на другую генеральную совокупность (emerging market)")
elif abs(avg_diff_pct) < 60:
    print(f"  → Модели ЧАСТИЧНО робастны: разница {avg_diff_pct:.0f}%, ")
    print(f"    ranking моделей сохраняется, абсолютные метрики отличаются")
else:
    print(f"  → Модели НЕ робастны: разница {avg_diff_pct:.0f}%, ")
    print(f"    требуется переобучение под специфику КЗ")

# Проверяем, сохраняется ли ranking
ru_ranking = ["Naive", "AutoTheta", "AutoETS", "AutoARIMA", "CES"]
kz_ranking = comp_df.sort_values("MAE_KZ")["model"].tolist()

print(f"\n  Ranking РФ:  {' > '.join(ru_ranking[:len(kz_ranking)])}")
print(f"  Ranking КЗ:  {' > '.join(kz_ranking)}")
print(f"  Ranking {'СОХРАНИЛСЯ' if kz_ranking[:3] == ['Naive','AutoTheta','AutoETS'][:len(kz_ranking[:3])] else 'ИЗМЕНИЛСЯ'}")


print("Сохранённые файлы:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith('kz_'):
        path = os.path.join(FIG_DIR, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {path} ({size_kb:.0f} KB)")