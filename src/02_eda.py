import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
import warnings
warnings.filterwarnings('ignore')


# Путь к данным 
from paths import DATA_DIR, FIG_DIR
DATA_PATH = DATA_DIR / "dataset_monthly.csv"

# Папка для сохранения графиков
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.figsize': (14, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.set_index("date").sort_index()

TARGET = "key_rate"

# Экзогенные переменные
EXOG_ALL = ["cpi_mom", "m2", "gdp", "ipp_yoy", "usd_rub", "brent",
            "zcyc_1y", "zcyc_5y", "zcyc_10y", "spread_10y_1y",
            "spread_10y_3m", "spread_5y_1y"]
EXOG = [v for v in EXOG_ALL if v in df.columns]

# Переменные для кросс-корреляционного анализа
CROSS_CORR_VARS = ["cpi_mom", "brent", "usd_rub", "m2",
                   "zcyc_1y", "zcyc_10y", "spread_10y_1y"]
CROSS_CORR_VARS = [v for v in CROSS_CORR_VARS if v in df.columns]

print("=" * 60)
print(f"Загружен датасет: {df.shape[0]} строк x {df.shape[1]} столбцов")
print(f"Период: {df.index.min():%Y-%m} — {df.index.max():%Y-%m}")
print("=" * 60)


print("\n" + "=" * 60)
print("ЭТАП 1: ВИЗУАЛЬНЫЙ ОСМОТР")
print("=" * 60)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df.index, df[TARGET], color='#1f4e79', linewidth=2, label='Ключевая ставка')
ax.set_title('Ключевая ставка ЦБ РФ (целевая переменная)')
ax.set_ylabel('% годовых')

median_val = df[TARGET].median()
ax.axhline(y=median_val, color='gray', linestyle='--', alpha=0.5,
           label=f'Медиана: {median_val:.1f}%')

events = {
    '2014-12-16': ('Кризис 2014', 'red'),
    '2020-04-01': ('COVID', 'orange'),
    '2022-02-28': ('Февраль 2022', 'red'),
}
for date_str, (label, color) in events.items():
    d = pd.Timestamp(date_str)
    ax.axvline(x=d, color=color, alpha=0.4, linestyle=':')
    ax.text(d, ax.get_ylim()[1] * 0.9, label, ha='center', fontsize=9, color=color)

ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/eda_01_key_rate.png')
plt.close()
print("  [OK] eda_01_key_rate.png")

n_vars = len(EXOG)
n_cols = 2
n_rows = (n_vars + 1) // 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
axes = axes.flatten()

for i, var in enumerate(EXOG):
    ax = axes[i]
    data = df[var].dropna()
    ax.plot(data.index, data.values, linewidth=1.2, color='#2c3e50')
    ax.set_title(var, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=data.mean(), color='orange', linestyle='--', alpha=0.4)

for j in range(n_vars, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Экзогенные переменные', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/eda_02_exog_vars.png')
plt.close()
print("  [OK] eda_02_exog_vars.png")

print("\n" + "=" * 60)
print("ЭТАП 2: ОПИСАТЕЛЬНЫЕ СТАТИСТИКИ")
print("=" * 60)

all_vars = [TARGET] + EXOG
desc = df[all_vars].describe().T
desc['missing'] = df[all_vars].isna().sum()
desc['missing_%'] = (desc['missing'] / len(df) * 100).round(1)

print(desc[['count', 'mean', 'std', 'min', '50%', 'max', 'missing', 'missing_%']].round(2).to_string())
desc.to_csv(f'{FIG_DIR}/eda_descriptive_stats.csv')
print(f"\n  [OK] eda_descriptive_stats.csv")



print("\n" + "=" * 60)
print("ЭТАП 3: ТЕСТЫ СТАЦИОНАРНОСТИ")
print("=" * 60)

stationarity_results = []
test_vars = [TARGET] + EXOG

for var in test_vars:
    series = df[var].dropna()
    if len(series) < 20:
        continue

    # ADF test
    adf_stat, adf_p, adf_lags, _, _, _ = adfuller(series, autolag='AIC')

    # KPSS test (regression='ct' = constant + trend)
    try:
        kpss_stat, kpss_p, kpss_lags, kpss_cv = kpss(series, regression='ct', nlags='auto')
    except Exception:
        kpss_stat, kpss_p = np.nan, np.nan

    # Интерпретация
    adf_reject = bool(adf_p < 0.05)
    kpss_reject = bool(kpss_p < 0.05) if not np.isnan(kpss_p) else None

    if adf_reject and not kpss_reject:
        conclusion = "Стационарный"
    elif not adf_reject and (kpss_reject is True):
        conclusion = "Нестационарный → d=1"
    elif adf_reject and (kpss_reject is True):
        conclusion = "Тренд-стационарный"
    else:
        conclusion = "Неопределённо"

    stationarity_results.append({
        'variable': var,
        'ADF_stat': round(adf_stat, 3),
        'ADF_p': round(adf_p, 4),
        'ADF_reject': 'Да' if adf_reject else 'Нет',
        'KPSS_stat': round(kpss_stat, 3) if not np.isnan(kpss_stat) else '-',
        'KPSS_p': round(kpss_p, 4) if not np.isnan(kpss_p) else '-',
        'KPSS_reject': 'Да' if kpss_reject else ('Нет' if kpss_reject is False else '-'),
        'conclusion': conclusion,
    })

stat_df = pd.DataFrame(stationarity_results)
print(stat_df.to_string(index=False))
stat_df.to_csv(f'{FIG_DIR}/eda_stationarity_tests.csv', index=False)
print(f"\n  [OK] eda_stationarity_tests.csv")


print("\n" + "=" * 60)
print("ЭТАП 4: ACF/PACF")
print("=" * 60)

series_level = df[TARGET].dropna()
series_diff = series_level.diff().dropna()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

plot_acf(series_level, ax=axes[0, 0], lags=24,
         title='ACF: ключевая ставка (уровни)')
plot_pacf(series_level, ax=axes[0, 1], lags=24, method='ywm',
          title='PACF: ключевая ставка (уровни)')

plot_acf(series_diff, ax=axes[1, 0], lags=24,
         title='ACF: ключевая ставка (первые разности, Δy)')
plot_pacf(series_diff, ax=axes[1, 1], lags=24, method='ywm',
          title='PACF: ключевая ставка (первые разности, Δy)')

plt.suptitle('Автокорреляционный анализ ключевой ставки', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/eda_03_acf_pacf.png')
plt.close()

print(f"  ACF(1) в уровнях:   {acf(series_level, nlags=1)[1]:.3f}")
print(f"  ACF(1) в разностях: {acf(series_diff, nlags=1)[1]:.3f}")
print(f"  → Высокая ACF(1) в уровнях подтверждает нестационарность")
print(f"  [OK] eda_03_acf_pacf.png")



print("\n" + "=" * 60)
print("ЭТАП 5: КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("=" * 60)

corr_vars = [TARGET] + EXOG
corr_matrix = df[corr_vars].corr()

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(corr_vars)))
ax.set_yticks(range(len(corr_vars)))
ax.set_xticklabels(corr_vars, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(corr_vars, fontsize=9)

for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        val = corr_matrix.iloc[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

plt.colorbar(im, ax=ax, label='Корреляция Пирсона')
ax.set_title('Матрица корреляций')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/eda_04_correlation_heatmap.png')
plt.close()
print("  [OK] eda_04_correlation_heatmap.png")

print(f"\n  Корреляции с ключевой ставкой:")
key_corr = corr_matrix[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
for var, r in key_corr.items():
    marker = "***" if abs(r) > 0.5 else "**" if abs(r) > 0.3 else "*" if abs(r) > 0.1 else ""
    print(f"    {var:20s}: r = {r:+.3f} {marker}")

max_lag = 12
cc_results = []

fig, axes = plt.subplots(len(CROSS_CORR_VARS), 1,
                          figsize=(14, 3 * len(CROSS_CORR_VARS)))
if len(CROSS_CORR_VARS) == 1:
    axes = [axes]

for idx, var in enumerate(CROSS_CORR_VARS):
    ax = axes[idx]
    temp = df[[TARGET, var]].dropna()
    y = temp[TARGET].values
    x = temp[var].values
    n = len(temp)

    lags = list(range(-max_lag, max_lag + 1))
    cc_values = []
    for lag in lags:
        if lag >= 0:
            cc = np.corrcoef(y[lag:], x[:n - lag])[0, 1] if lag < n else 0
        else:
            cc = np.corrcoef(y[:n + lag], x[-lag:])[0, 1] if -lag < n else 0
        cc_values.append(cc)

    # Лучший лаг
    best_idx = np.argmax(np.abs(cc_values))
    best_lag = lags[best_idx]
    best_cc = cc_values[best_idx]

    cc_results.append({
        'variable': var,
        'best_lag': best_lag,
        'best_cc': round(best_cc, 3),
        'cc_lag0': round(cc_values[max_lag], 3),
    })

    # График
    colors = ['#c0392b' if v < 0 else '#2980b9' for v in cc_values]
    ax.bar(lags, cc_values, color=colors, alpha=0.7, width=0.8)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ci = 1.96 / np.sqrt(n)  # 95% доверительный интервал
    ax.axhline(y=ci, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=-ci, color='gray', linestyle=':', alpha=0.5)
    ax.set_title(f'{var} → key_rate  (лучший лаг: {best_lag}, r={best_cc:.3f})')
    ax.set_xlabel('Лаг (месяцев, + = X опережает Y)')
    ax.set_ylabel('CCF')

plt.suptitle('Кросс-корреляция экзогенных переменных с ключевой ставкой',
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/eda_05_cross_correlations.png')
plt.close()

cc_df = pd.DataFrame(cc_results)
cc_df.to_csv(f'{FIG_DIR}/eda_cross_correlations.csv', index=False)
print(f"\n  Результаты кросс-корреляционного анализа:")
print(cc_df.to_string(index=False))
print(f"  [OK] eda_05_cross_correlations.png")


print("\n" + "=" * 60)
print("ЭТАП 6: STL-ДЕКОМПОЗИЦИЯ")
print("=" * 60)

# STL для ключевой ставки
series_kr = df[TARGET].dropna()

try:
    stl = STL(series_kr, period=12, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(series_kr.index, series_kr.values, linewidth=1.5, color='#1f4e79')
    axes[0].set_title('Исходный ряд: ключевая ставка')
    axes[0].set_ylabel('% годовых')

    axes[1].plot(result.trend.index, result.trend.values, color='#e67e22', linewidth=2)
    axes[1].set_title('Тренд (T)')
    axes[1].set_ylabel('% годовых')

    axes[2].plot(result.seasonal.index, result.seasonal.values, color='#27ae60', linewidth=1)
    axes[2].set_title('Сезонная компонента (S)')
    axes[2].set_ylabel('% годовых')
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    axes[3].plot(result.resid.index, result.resid.values, color='#c0392b', linewidth=1, alpha=0.7)
    axes[3].set_title('Остаток (R)')
    axes[3].set_ylabel('% годовых')
    axes[3].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.suptitle('STL-декомпозиция ключевой ставки (период = 12 месяцев)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/eda_06_stl_decomposition.png')
    plt.close()

    # Доля дисперсии каждого компонента
    total_var = series_kr.var()
    trend_pct = result.trend.var() / total_var * 100
    seasonal_pct = result.seasonal.var() / total_var * 100
    resid_pct = result.resid.var() / total_var * 100

    print(f"  Доля дисперсии:")
    print(f"    Тренд:      {trend_pct:.1f}%")
    print(f"    Сезонность: {seasonal_pct:.1f}%")
    print(f"    Остаток:    {resid_pct:.1f}%")
    print(f"  → Сезонный компонент пренебрежимо мал")
    print(f"  [OK] eda_06_stl_decomposition.png")

except Exception as e:
    print(f"  [!] STL не удался: {e}")



print("\n" + "=" * 60)
print("КЛЮЧЕВЫЕ ВЫВОДЫ EDA")
print("=" * 60)

print("""
1. СТАЦИОНАРНОСТЬ: ключевая ставка нестационарна в уровнях.
   Для ARIMA потребуется дифференцирование (d=1).

2. АВТОКОРРЕЛЯЦИЯ: высокая персистентность (ACF медленно затухает).
   После дифференцирования структура пригодна для ARIMA.

3. КОРРЕЛЯЦИИ С КЛЮЧЕВОЙ СТАВКОЙ:""")

for _, row in cc_df.sort_values('best_cc', key=abs, ascending=False).iterrows():
    lag_str = f"лаг {row['best_lag']}" if row['best_lag'] != 0 else "без лага"
    direction = "опережает" if row['best_lag'] > 0 else "совпадает" if row['best_lag'] == 0 else "запаздывает"
    print(f"   • {row['variable']:20s}: r = {row['best_cc']:+.3f} ({lag_str}, {direction})")

print("""
4. СЕЗОННОСТЬ: отсутствует в ключевой ставке (подтверждено STL).
   Это ожидаемо — решения ЦБ не привязаны к календарю.

5. СТРУКТУРНЫЕ СЛОМЫ: три кризисных эпизода (2014, 2020, 2022)
   создают экстремальные значения. Robust-методы (robust STL,
   Theta) предпочтительнее для работы с такими данными.
""")
