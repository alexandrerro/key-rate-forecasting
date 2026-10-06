"""
==============================================================================
12_reverse_task.py — Обратная задача: прогнозирование реакции рынка на ставку ЦБ
==============================================================================

Прямая задача (Главы 3.1–3.8): key_rate = f(RUONIA, КБД, макро...)
  → Частная компания хочет угадать ставку ЦБ

Обратная задача (этот скрипт): RUONIA = g(key_rate, макро...)
  → ЦБ устанавливает ставку и хочет понять реакцию рынка

Аналогия с ценообразованием:
  Ценообразование: спрос = f(цена) → выбираем цену для макс. выручки
  ДКП:             RUONIA = g(ставка) → выбираем ставку для целевой инфляции

СТРУКТУРА СКРИПТА:
  Блок 1: Прогнозирование RUONIA как функции key_rate + макро
           Модели: Naive, AutoETS, AutoARIMA, LightGBM
           Expanding window CV, h=1

  Блок 2: Прогнозирование ROISFIX_3M как функции key_rate + макро
           Те же модели

  Блок 3: Сценарный анализ «что если» (LightGBM)
           Варьируем key_rate = 12%, 14%, 16%, 18%, 20%
           Фиксируем остальные → получаем кривую реакции

  Блок 4: Pass-through анализ
           На сколько п.п. изменится RUONIA при Δkey_rate = 1 п.п.?

  Блок 5: Доверительные интервалы (квантильная регрессия)

Как запустить:
    python 12_reverse_task.py

Входные данные: dataset_monthly.csv
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import lightgbm as lgb
import os
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta
    HAS_SF = True
except ImportError:
    HAS_SF = False
    print("[!] statsforecast не установлен — только Naive и LightGBM")

# ============================================================
# НАСТРОЙКИ
# ============================================================

from paths import DATA_DIR, FIG_DIR
DATA_PATH = DATA_DIR / "dataset_monthly.csv"
os.makedirs(FIG_DIR, exist_ok=True)

MIN_TRAIN_SIZE = 60
HORIZON = 6

# Целевые переменные для обратной задачи
TARGETS_REVERSE = {
    "ruonia": "RUONIA (ставка межбанковского рынка)",
    "roisfix_3m": "ROISFIX 3M (IRS на 3 месяца)",
}

# Предикторы для LightGBM (key_rate — управляющая, остальные — контекст)
EXOG_REVERSE = ["key_rate", "cpi_mom", "m2", "usd_rub", "brent"]

LGB_PARAMS = {
    "objective": "regression", "metric": "mae",
    "num_leaves": 15, "max_depth": 5, "learning_rate": 0.05,
    "n_estimators": 200, "min_child_samples": 5,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 0.1,
    "random_state": 42, "verbose": -1,
}

plt.rcParams.update({
    'figure.figsize': (14, 6), 'figure.dpi': 120,
    'savefig.dpi': 150, 'savefig.bbox': 'tight',
})


# ============================================================
# БЛОК 0: ЗАГРУЗКА И ПОДГОТОВКА
# ============================================================

print("=" * 60)
print("ОБРАТНАЯ ЗАДАЧА: РЕАКЦИЯ РЫНКА НА СТАВКУ ЦБ")
print("=" * 60)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Проверяем наличие целевых переменных
available_targets = {}
for target, desc in TARGETS_REVERSE.items():
    if target in df.columns and df[target].notna().sum() > MIN_TRAIN_SIZE:
        available_targets[target] = desc
        print(f"  [OK] {target}: {df[target].notna().sum()} наблюдений — {desc}")
    else:
        print(f"  [!] {target}: недостаточно данных, пропускаем")

if not available_targets:
    print("\n[!] Нет доступных целевых переменных. Проверьте датасет.")
    exit(1)

EXOG_REVERSE = [v for v in EXOG_REVERSE if v in df.columns]
print(f"\n  Предикторы: {EXOG_REVERSE}")
print(f"  Управляющая переменная: key_rate")


# ============================================================
# ФУНКЦИИ
# ============================================================

def calc_metrics(y_true, y_pred, name):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {"model": name, "MAE": np.nan, "RMSE": np.nan, "MAPE_%": np.nan, "n": 0}
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    mape = np.mean(np.abs((yt - yp) / yt)) * 100 if np.all(yt != 0) else np.nan
    return {"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE_%": round(mape, 2), "n": int(mask.sum())}


def build_features_reverse(df, target, exog_vars, horizon=1, lags=[1, 2, 3, 6]):
    """Feature engineering для обратной задачи."""
    df_fe = df[["date", target] + exog_vars].copy()

    # Лаги целевой (сдвинуты на horizon)
    for lag in lags:
        df_fe[f"{target}_lag{lag}"] = df_fe[target].shift(lag + horizon - 1)

    # Лаги экзогенных (сдвинуты на horizon)
    for var in exog_vars:
        df_fe[f"{var}_lag1"] = df_fe[var].shift(horizon)

    # Изменение ключевой ставки
    if "key_rate" in exog_vars:
        df_fe["key_rate_diff1"] = df_fe["key_rate"].shift(horizon) - df_fe["key_rate"].shift(horizon + 1)
        df_fe["key_rate_diff3"] = df_fe["key_rate"].shift(horizon) - df_fe["key_rate"].shift(horizon + 3)

    # Скользящие статистики
    for w in [3, 6]:
        df_fe[f"{target}_rmean{w}"] = df_fe[target].shift(horizon).rolling(w, min_periods=1).mean()

    feature_cols = [c for c in df_fe.columns if c not in ["date", target]]
    feature_cols = [c for c in feature_cols if c not in exog_vars]

    return df_fe, feature_cols


# ============================================================
# БЛОК 1–2: ПРОГНОЗИРОВАНИЕ ДЛЯ КАЖДОЙ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
# ============================================================

all_reverse_results = {}
all_lgb_models = {}

for target, desc in available_targets.items():
    print(f"\n{'='*60}")
    print(f"ПРОГНОЗИРОВАНИЕ: {target.upper()} ({desc})")
    print(f"{'='*60}")

    # --- Feature engineering ---
    df_fe, feature_cols = build_features_reverse(df, target, EXOG_REVERSE, horizon=HORIZON)
    df_clean = df_fe.dropna().reset_index(drop=True)

    n = len(df_clean)
    val_start = MIN_TRAIN_SIZE
    n_val = n - val_start

    print(f"  Признаков: {len(feature_cols)}")
    print(f"  Наблюдений: {n}, шагов валидации: {n_val}")

    y_all = df_clean[target].values
    X_all = df_clean[feature_cols].values
    dates_all = df_clean["date"].values

    # --- Naive ---
    preds_naive = []
    for i in range(n_val):
        t = val_start + i
        preds_naive.append({
            "ds": dates_all[t], "y_true": y_all[t], "pred": y_all[t-1]
        })

    # --- LightGBM ---
    print(f"  LightGBM: expanding window ({n_val} шагов)...")
    start_time = time.time()
    preds_lgb = []
    last_model = None

    for i in range(n_val):
        t = val_start + i
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_all[:t], y_all[:t])
        pred = model.predict(X_all[t:t+1])[0]
        preds_lgb.append({"ds": dates_all[t], "y_true": y_all[t], "pred": pred})
        last_model = model

    elapsed = time.time() - start_time
    print(f"  LightGBM завершён за {elapsed:.1f}s")

    all_lgb_models[target] = {
        "model": last_model,
        "feature_cols": feature_cols,
        "df_clean": df_clean,
    }

    # --- LightGBM: доверительные интервалы ---
    print(f"  LightGBM: квантильная регрессия (интервалы)...")
    lgb_lo = LGB_PARAMS.copy()
    lgb_lo["objective"] = "quantile"; lgb_lo["alpha"] = 0.025
    lgb_lo.pop("metric", None)

    lgb_hi = LGB_PARAMS.copy()
    lgb_hi["objective"] = "quantile"; lgb_hi["alpha"] = 0.975
    lgb_hi.pop("metric", None)

    preds_lgb_ci = []
    for i in range(n_val):
        t = val_start + i
        m_lo = lgb.LGBMRegressor(**lgb_lo)
        m_lo.fit(X_all[:t], y_all[:t])
        p_lo = m_lo.predict(X_all[t:t+1])[0]

        m_hi = lgb.LGBMRegressor(**lgb_hi)
        m_hi.fit(X_all[:t], y_all[:t])
        p_hi = m_hi.predict(X_all[t:t+1])[0]

        preds_lgb_ci.append({
            "ds": dates_all[t], "y_true": y_all[t],
            "pred": preds_lgb[i]["pred"],
            "lower_95": p_lo, "upper_95": p_hi,
        })

    # Coverage
    df_ci = pd.DataFrame(preds_lgb_ci)
    inside = (df_ci["y_true"] >= df_ci["lower_95"]) & (df_ci["y_true"] <= df_ci["upper_95"])
    coverage = inside.mean()
    avg_width = (df_ci["upper_95"] - df_ci["lower_95"]).mean()
    print(f"  Coverage (95% CI): {coverage:.1%}, ширина: {avg_width:.3f} п.п.")

    # --- StatsForecast (если доступен) ---
    preds_sf = {}
    if HAS_SF:
        print(f"  StatsForecast: expanding window...")
        nixtla_target = df[["date", target]].dropna().copy()
        nixtla_target = nixtla_target.rename(columns={"date": "ds", target: "y"})
        nixtla_target["unique_id"] = target
        nixtla_target = nixtla_target[["unique_id", "ds", "y"]].sort_values("ds")

        n_nixtla = len(nixtla_target)
        n_win_sf = n_nixtla - MIN_TRAIN_SIZE - HORIZON + 1

        sf_models = [
            AutoARIMA(season_length=1),
            AutoETS(season_length=1),
            AutoTheta(season_length=1),
        ]
        sf = StatsForecast(models=sf_models, freq="MS", n_jobs=1)

        try:
            cv_sf = sf.cross_validation(
                df=nixtla_target, h=HORIZON, step_size=1, n_windows=n_win_sf,
                level=[95],
            )
            if 'unique_id' in cv_sf.columns:
                cv_sf = cv_sf.drop(columns=['unique_id'])
            if 'cutoff' in cv_sf.columns:
                cv_sf = cv_sf.drop(columns=['cutoff'])

            sf_model_names = [c for c in cv_sf.columns
                              if c not in ["ds", "y"] and "-lo-" not in c and "-hi-" not in c]

            for mn in sf_model_names:
                m = calc_metrics(cv_sf["y"].values, cv_sf[mn].values, mn)
                preds_sf[mn] = m

                # Coverage если есть интервалы
                lo_col = f"{mn}-lo-95"
                hi_col = f"{mn}-hi-95"
                if lo_col in cv_sf.columns:
                    ins = (cv_sf["y"] >= cv_sf[lo_col]) & (cv_sf["y"] <= cv_sf[hi_col])
                    m["coverage_95"] = round(ins.mean(), 4)

                print(f"    {mn}: MAE={m['MAE']:.4f}" +
                      (f", coverage={m.get('coverage_95', 'N/A')}" if 'coverage_95' in m else ""))

        except Exception as e:
            print(f"    [!] StatsForecast ошибка: {e}")

    # --- Сводка метрик ---
    metrics_list = []
    m_naive = calc_metrics(
        pd.DataFrame(preds_naive)["y_true"].values,
        pd.DataFrame(preds_naive)["pred"].values, "Naive"
    )
    metrics_list.append(m_naive)

    m_lgb = calc_metrics(
        pd.DataFrame(preds_lgb)["y_true"].values,
        pd.DataFrame(preds_lgb)["pred"].values, "LightGBM"
    )
    m_lgb["coverage_95"] = round(coverage, 4)
    metrics_list.append(m_lgb)

    for mn, m in preds_sf.items():
        metrics_list.append(m)

    metrics_df = pd.DataFrame(metrics_list).sort_values("MAE")

    print(f"\n  МЕТРИКИ ({target}):")
    print(f"  {'Модель':15s} | {'MAE':>8s} | {'RMSE':>8s} | {'Coverage':>10s}")
    print(f"  {'-'*50}")
    for _, row in metrics_df.iterrows():
        cov = f"{row['coverage_95']:.1%}" if 'coverage_95' in row and pd.notna(row.get('coverage_95')) else "—"
        print(f"  {row['model']:15s} | {row['MAE']:8.4f} | {row['RMSE']:8.4f} | {cov:>10s}")

    all_reverse_results[target] = {
        "metrics": metrics_df,
        "preds_naive": preds_naive,
        "preds_lgb": preds_lgb,
        "preds_lgb_ci": preds_lgb_ci,
        "coverage": coverage,
        "avg_width": avg_width,
    }

    metrics_df.to_csv(f"{FIG_DIR}/reverse_{target}_metrics.csv", index=False)

    # --- График: прогноз vs факт с CI ---
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(df_ci["ds"], df_ci["y_true"], color='black', linewidth=2,
            label='Факт', zorder=5)
    ax.plot(df_ci["ds"], df_ci["pred"], color='#2980b9', linewidth=1.3,
            alpha=0.8, label=f'LightGBM (MAE={m_lgb["MAE"]:.3f})')
    ax.fill_between(df_ci["ds"], df_ci["lower_95"], df_ci["upper_95"],
                    color='#2980b9', alpha=0.15,
                    label=f'95% CI (cov={coverage:.1%})')

    outside = ~inside.values
    if outside.any():
        ax.scatter(df_ci["ds"][outside], df_ci["y_true"][outside],
                   color='red', s=30, zorder=6, label=f'Вне CI ({outside.sum()})')

    ax.set_title(f'Обратная задача: прогноз {target} (h=1)')
    ax.set_ylabel('% годовых')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/reverse_{target}_forecast.png')
    plt.close()
    print(f"  [OK] reverse_{target}_forecast.png")


# ============================================================
# БЛОК 3: СЦЕНАРНЫЙ АНАЛИЗ «ЧТО ЕСЛИ»
# ============================================================

print(f"\n{'='*60}")
print("БЛОК 3: СЦЕНАРНЫЙ АНАЛИЗ (what-if)")
print("=" * 60)

# Сценарии: разные значения ключевой ставки
SCENARIOS = [10, 12, 14, 15, 16, 18, 20, 21]

scenario_results = {}

for target, desc in available_targets.items():
    if target not in all_lgb_models:
        continue

    model_info = all_lgb_models[target]
    model = model_info["model"]
    feature_cols = model_info["feature_cols"]
    df_clean = model_info["df_clean"]

    # Берём последнюю строку как базовый сценарий
    last_row = df_clean[feature_cols].iloc[-1].copy()
    current_rate = df.loc[df["key_rate"].notna(), "key_rate"].iloc[-1]

    print(f"\n  {target}: текущая ставка = {current_rate}%")
    print(f"  {'Ставка':>8s} | {'Прогноз':>10s} | {'Δ от текущ.':>12s} | {'Pass-through':>13s}")
    print(f"  {'-'*50}")

    results = []
    for rate in SCENARIOS:
        scenario_row = last_row.copy()

        # Подставляем новую ставку во все признаки, содержащие key_rate
        for col in feature_cols:
            if "key_rate" in col and "lag1" in col:
                scenario_row[col] = rate
            elif "key_rate" in col and "diff" in col:
                scenario_row[col] = rate - current_rate

        X_scenario = scenario_row.values.reshape(1, -1)
        pred = model.predict(X_scenario)[0]

        # Прогноз при текущей ставке (baseline)
        pred_baseline = model.predict(last_row.values.reshape(1, -1))[0]

        delta = pred - pred_baseline
        rate_delta = rate - current_rate
        pass_through = delta / rate_delta if rate_delta != 0 else np.nan

        results.append({
            "key_rate": rate,
            f"pred_{target}": round(pred, 4),
            "delta_from_baseline": round(delta, 4),
            "pass_through": round(pass_through, 4) if not np.isnan(pass_through) else None,
        })

        pt_str = f"{pass_through:.3f}" if not np.isnan(pass_through) else "—"
        print(f"  {rate:7.1f}% | {pred:10.4f} | {delta:+11.4f} | {pt_str:>13s}")

    scenario_results[target] = pd.DataFrame(results)
    scenario_results[target].to_csv(f"{FIG_DIR}/reverse_{target}_scenarios.csv", index=False)

    # --- Квантильные интервалы для сценариев ---
    print(f"\n  Сценарные интервалы (95% CI):")

    lgb_lo_params = LGB_PARAMS.copy()
    lgb_lo_params["objective"] = "quantile"; lgb_lo_params["alpha"] = 0.025
    lgb_lo_params.pop("metric", None)
    lgb_hi_params = LGB_PARAMS.copy()
    lgb_hi_params["objective"] = "quantile"; lgb_hi_params["alpha"] = 0.975
    lgb_hi_params.pop("metric", None)

    X_train = df_clean[feature_cols].values[:-1]
    y_train = df_clean[target].values[:-1]

    m_lo = lgb.LGBMRegressor(**lgb_lo_params)
    m_lo.fit(X_train, y_train)
    m_hi = lgb.LGBMRegressor(**lgb_hi_params)
    m_hi.fit(X_train, y_train)

    scenario_ci = []
    print(f"  {'Ставка':>8s} | {'Прогноз':>10s} | {'Нижн. 95%':>10s} | {'Верхн. 95%':>10s} | {'Ширина':>8s}")
    print(f"  {'-'*55}")

    for rate in SCENARIOS:
        scenario_row = last_row.copy()
        for col in feature_cols:
            if "key_rate" in col and "lag1" in col:
                scenario_row[col] = rate
            elif "key_rate" in col and "diff" in col:
                scenario_row[col] = rate - current_rate

        X_sc = scenario_row.values.reshape(1, -1)
        pred = model.predict(X_sc)[0]
        lo = m_lo.predict(X_sc)[0]
        hi = m_hi.predict(X_sc)[0]
        width = hi - lo

        scenario_ci.append({
            "key_rate": rate, "pred": round(pred, 4),
            "lower_95": round(lo, 4), "upper_95": round(hi, 4),
            "width": round(width, 4),
        })

        print(f"  {rate:7.1f}% | {pred:10.4f} | {lo:10.4f} | {hi:10.4f} | {width:8.4f}")

    scenario_ci_df = pd.DataFrame(scenario_ci)
    scenario_ci_df.to_csv(f"{FIG_DIR}/reverse_{target}_scenarios_ci.csv", index=False)


# ============================================================
# БЛОК 4: ВИЗУАЛИЗАЦИЯ СЦЕНАРНОГО АНАЛИЗА
# ============================================================

print(f"\n{'='*60}")
print("БЛОК 4: ВИЗУАЛИЗАЦИЯ")
print("=" * 60)

for target, desc in available_targets.items():
    if target not in scenario_results:
        continue

    sc_df = scenario_results[target]
    sc_ci = pd.read_csv(f"{FIG_DIR}/reverse_{target}_scenarios_ci.csv")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Левый: кривая реакции с CI ---
    ax = axes[0]
    ax.plot(sc_ci["key_rate"], sc_ci["pred"], 'o-', color='#2980b9',
            linewidth=2, markersize=8, label=f'Прогноз {target}', zorder=5)
    ax.fill_between(sc_ci["key_rate"], sc_ci["lower_95"], sc_ci["upper_95"],
                    color='#2980b9', alpha=0.15, label='95% CI')

    # Текущая ставка
    current_rate = df.loc[df["key_rate"].notna(), "key_rate"].iloc[-1]
    ax.axvline(x=current_rate, color='red', linestyle='--', alpha=0.7,
               label=f'Текущая ставка ({current_rate}%)')

    # Линия 45° (pass-through = 1)
    min_r, max_r = min(SCENARIOS), max(SCENARIOS)
    baseline_pred = sc_df[f"pred_{target}"].iloc[
        sc_df["key_rate"].tolist().index(
            min(SCENARIOS, key=lambda x: abs(x - current_rate))
        )
    ]
    ax.plot([min_r, max_r],
            [baseline_pred + (min_r - current_rate), baseline_pred + (max_r - current_rate)],
            '--', color='gray', alpha=0.5, label='Pass-through = 1.0')

    ax.set_xlabel('Ключевая ставка ЦБ, %')
    ax.set_ylabel(f'{target}, %')
    ax.set_title(f'Кривая реакции: {desc}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Правый: pass-through ---
    ax = axes[1]
    pt_values = sc_df["pass_through"].dropna()
    pt_rates = sc_df.loc[sc_df["pass_through"].notna(), "key_rate"]

    if len(pt_values) > 0:
        colors = ['#2ecc71' if v >= 0.8 else '#f39c12' if v >= 0.5 else '#e74c3c'
                  for v in pt_values]
        ax.bar(pt_rates, pt_values, color=colors, alpha=0.8, width=0.8)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Полная передача')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Ключевая ставка ЦБ, %')
        ax.set_ylabel('Pass-through (Δtarget / Δkey_rate)')
        ax.set_title(f'Коэффициент передачи ставки в {target}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Обратная задача: {desc}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/reverse_{target}_scenarios_plot.png')
    plt.close()
    print(f"  [OK] reverse_{target}_scenarios_plot.png")


# --- Сводный график: все целевые на одном ---
if len(available_targets) > 1:
    fig, ax = plt.subplots(figsize=(12, 7))

    colors_targets = {"ruonia": "#2980b9", "roisfix_3m": "#e74c3c", "cpi_mom": "#27ae60"}

    for target in available_targets:
        if target in scenario_results:
            sc_df = scenario_results[target]
            sc_ci = pd.read_csv(f"{FIG_DIR}/reverse_{target}_scenarios_ci.csv")
            color = colors_targets.get(target, '#888888')

            ax.plot(sc_ci["key_rate"], sc_ci["pred"], 'o-', color=color,
                    linewidth=2, markersize=6, label=target)
            ax.fill_between(sc_ci["key_rate"], sc_ci["lower_95"], sc_ci["upper_95"],
                            color=color, alpha=0.1)

    ax.axvline(x=current_rate, color='gray', linestyle='--', alpha=0.7,
               label=f'Текущая ставка ({current_rate}%)')
    ax.set_xlabel('Ключевая ставка ЦБ, %')
    ax.set_ylabel('Прогнозное значение, %')
    ax.set_title('Сценарный анализ: реакция рыночных индикаторов на ставку ЦБ')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/reverse_all_scenarios.png')
    plt.close()
    print(f"  [OK] reverse_all_scenarios.png")


# ============================================================
# БЛОК 5: PASS-THROUGH АНАЛИЗ
# ============================================================

print(f"\n{'='*60}")
print("БЛОК 5: PASS-THROUGH АНАЛИЗ")
print("=" * 60)

for target, desc in available_targets.items():
    if target not in scenario_results:
        continue

    sc_df = scenario_results[target]
    pt = sc_df["pass_through"].dropna()

    if len(pt) == 0:
        continue

    avg_pt = pt.mean()
    print(f"\n  {target}:")
    print(f"    Средний pass-through: {avg_pt:.3f}")

    if avg_pt > 0.9:
        print(f"    → Полная передача: {target} практически повторяет ставку ЦБ")
    elif avg_pt > 0.7:
        print(f"    → Высокая передача: {target} сильно зависит от ставки")
    elif avg_pt > 0.3:
        print(f"    → Частичная передача: другие факторы тоже влияют")
    else:
        print(f"    → Низкая передача: {target} слабо реагирует на ставку")


# ============================================================
# БЛОК 6: ИТОГОВАЯ СВОДКА
# ============================================================

print(f"\n{'='*60}")
print("ИТОГОВАЯ СВОДКА: ОБРАТНАЯ ЗАДАЧА")
print("=" * 60)

print(f"""
РЕЗУЛЬТАТЫ ОБРАТНОЙ ЗАДАЧИ:

Для каждой целевой переменной ({', '.join(available_targets.keys())})
обучены модели Naive, LightGBM{' + AutoARIMA, AutoETS, AutoTheta' if HAS_SF else ''}
с expanding window CV (h=1).

СЦЕНАРНЫЙ АНАЛИЗ:
  При изменении ключевой ставки с {current_rate}% на разные уровни,
  модель предсказывает реакцию рыночных индикаторов.""")

for target, desc in available_targets.items():
    if target in scenario_results:
        sc_df = scenario_results[target]
        pt = sc_df["pass_through"].dropna()
        if len(pt) > 0:
            print(f"\n  {target} ({desc}):")
            print(f"    Средний pass-through: {pt.mean():.3f}")
            res = all_reverse_results.get(target, {})
            if res:
                print(f"    Coverage (95% CI): {res.get('coverage', 0):.1%}")
                print(f"    Средняя ширина CI: {res.get('avg_width', 0):.3f} п.п.")

print(f"""
ДЛЯ ДИПЛОМА (п. 3.9):
  «Обратная задача — прогнозирование реакции рыночных индикаторов
   (RUONIA, ROISFIX) на изменение ключевой ставки ЦБ — решена
   с использованием тех же моделей, что и прямая задача.
   Сценарный анализ показал, что при изменении ключевой ставки
   на 1 п.п. RUONIA изменяется на ~{scenario_results.get('ruonia', pd.DataFrame({'pass_through': [0.9]}))['pass_through'].dropna().mean():.2f} п.п.
   (pass-through ≈ {scenario_results.get('ruonia', pd.DataFrame({'pass_through': [0.9]}))['pass_through'].dropna().mean():.0%}),
   что указывает на {'полную' if scenario_results.get('ruonia', pd.DataFrame({'pass_through': [0.9]}))['pass_through'].dropna().mean() > 0.9 else 'высокую' if scenario_results.get('ruonia', pd.DataFrame({'pass_through': [0.9]}))['pass_through'].dropna().mean() > 0.7 else 'частичную'} передачу
   ставки в рыночные индикаторы.
   Данный подход аналогичен задаче ценообразования: ЦБ выбирает
   ставку, оценивая ожидаемую реакцию рынка, подобно тому как
   компания выбирает цену, оценивая ожидаемый спрос.»
""")

# Файлы
print("Сохранённые файлы:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith('reverse_'):
        path = os.path.join(FIG_DIR, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {path} ({size_kb:.0f} KB)")