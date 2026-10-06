
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import lightgbm as lgb
import time
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS
    HAS_SF = True
except ImportError:
    HAS_SF = False

from paths import DATA_DIR, FIG_DIR
DATA_PATH = DATA_DIR / "dataset_monthly.csv"
os.makedirs(FIG_DIR, exist_ok=True)

HORIZON = 3
MIN_TRAIN_SIZE = 60
EXOG_TO_FORECAST = ["zcyc_1y","zcyc_5y","zcyc_10y","cpi_mom","usd_rub","brent","m2","ruonia"]
ROISFIX_COLS = ["roisfix_1w","roisfix_1m","roisfix_3m","roisfix_6m"]

LGB_PARAMS = {
    "objective":"regression","metric":"mae","num_leaves":15,"max_depth":5,
    "learning_rate":0.05,"n_estimators":200,"min_child_samples":5,
    "subsample":0.8,"colsample_bytree":0.8,"reg_alpha":0.1,"reg_lambda":0.1,
    "random_state":42,"verbose":-1,
}

plt.rcParams.update({'figure.figsize':(14,6),'figure.dpi':120,'savefig.dpi':150,'savefig.bbox':'tight'})

print("="*60)
print(f"ДВУХСТАДИЙНЫЙ ПРОГНОЗ + ROISFIX (h={HORIZON})")
print("="*60)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
TARGET = "key_rate"
EXOG_TO_FORECAST = [v for v in EXOG_TO_FORECAST if v in df.columns]
ROISFIX_COLS = ["roisfix_1w", "roisfix_1m", "roisfix_3m", "roisfix_6m", "roisfix_1y"]
print(f"Экзогенные: {EXOG_TO_FORECAST}")
print(f"ROISFIX: {ROISFIX_COLS}")

LAGS = [1,2,3,6,12]
for lag in LAGS:
    df[f"lag_{lag}"] = df[TARGET].shift(lag + HORIZON - 1)

for w in [3,6,12]:
    df[f"rolling_mean_{w}"] = df[TARGET].shift(HORIZON).rolling(window=w,min_periods=1).mean()
    df[f"rolling_std_{w}"] = df[TARGET].shift(HORIZON).rolling(window=w,min_periods=1).std()

df["diff_1"] = df[TARGET].shift(HORIZON) - df[TARGET].shift(HORIZON+1)
df["diff_3"] = df[TARGET].shift(HORIZON) - df[TARGET].shift(HORIZON+3)
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter

for var in EXOG_TO_FORECAST:
    df[f"{var}_lagH"] = df[var].shift(HORIZON)
    df[f"{var}_oracle"] = df[var]

for col in ROISFIX_COLS:
    df[f"{col}_lag1"] = df[col].shift(1)

base_features = []
for col in df.columns:
    if col in ["date",TARGET]+EXOG_TO_FORECAST+ROISFIX_COLS: continue
    if any(col.startswith(p) for p in ["lag_","rolling_","diff_"]): base_features.append(col)
    elif col in ["month","quarter"]: base_features.append(col)

oracle_features = [f"{v}_oracle" for v in EXOG_TO_FORECAST]
roisfix_features = [f"{c}_lag1" for c in ROISFIX_COLS]

features_B = base_features
features_C = base_features + oracle_features
features_E = base_features + roisfix_features

all_cols = list(set(["date",TARGET]+features_C+[f"{v}_lagH" for v in EXOG_TO_FORECAST]+roisfix_features))
all_cols = [c for c in all_cols if c in df.columns]
df_clean = df[all_cols].dropna().reset_index(drop=True)

n = len(df_clean)
val_start = MIN_TRAIN_SIZE
n_val_steps = n - val_start
print(f"Признаков B:{len(features_B)}, C:{len(features_C)}, E:{len(features_E)}")
print(f"Наблюдений: {n}, шагов: {n_val_steps}")

def forecast_exog_autoets(sv, sd, nt, h):
    if not HAS_SF: return np.full(h, sv[nt-1])
    tdf = pd.DataFrame({"unique_id":"x","ds":sd[:nt],"y":sv[:nt]})
    try:
        sf = StatsForecast(models=[AutoETS(season_length=1)],freq="MS",n_jobs=1)
        sf.fit(tdf)
        return sf.predict(h=h)["AutoETS"].values
    except: return np.full(h, sv[nt-1])

y_all = df_clean[TARGET].values
dates_all = df_clean["date"].values
X_B = df_clean[[f for f in features_B if f in df_clean.columns]].values
X_C = df_clean[[f for f in features_C if f in df_clean.columns]].values
X_E = df_clean[[f for f in features_E if f in df_clean.columns]].values

exog_series = {v: df[v].values for v in EXOG_TO_FORECAST if v in df.columns}
full_dates = pd.to_datetime(df["date"])
date_to_idx = {d:i for i,d in enumerate(full_dates)}

pA,pB,pC,pD,pE = [],[],[],[],[]

print(f"\nЗапуск {n_val_steps} шагов...")
t0 = time.time()

for i in range(n_val_steps):
    t = val_start + i
    actual = y_all[t]
    pd_ = dates_all[t]
    yt = y_all[:t]
    fidx = date_to_idx.get(pd.Timestamp(pd_))

    pA.append({"ds":pd_,"y_true":actual,"pred":y_all[t-1] if t>0 else y_all[0]})

    mB = lgb.LGBMRegressor(**LGB_PARAMS); mB.fit(X_B[:t],yt)
    pB.append({"ds":pd_,"y_true":actual,"pred":mB.predict(X_B[t:t+1])[0]})

    mC = lgb.LGBMRegressor(**LGB_PARAMS); mC.fit(X_C[:t],yt)
    pC.append({"ds":pd_,"y_true":actual,"pred":mC.predict(X_C[t:t+1])[0]})

    if fidx is not None:
        net = fidx - HORIZON + 1
        fev = []
        for v in EXOG_TO_FORECAST:
            if v in exog_series and net > 10:
                sv = exog_series[v][:net]; sd = full_dates[:net]; vm = ~pd.isna(sv)
                if vm.sum()>10:
                    fev.append(forecast_exog_autoets(sv[vm],sd[vm],vm.sum(),HORIZON)[-1])
                else: fev.append(sv[vm][-1] if vm.any() else 0)
            else: fev.append(np.nan)
        xtd = np.concatenate([X_B[t:t+1].flatten(),np.array(fev)]).reshape(1,-1)
        mD = lgb.LGBMRegressor(**LGB_PARAMS); mD.fit(X_C[:t],yt)
        pD.append({"ds":pd_,"y_true":actual,"pred":mD.predict(xtd)[0]})
    else:
        pD.append({"ds":pd_,"y_true":actual,"pred":pB[-1]["pred"]})

    mE = lgb.LGBMRegressor(**LGB_PARAMS); mE.fit(X_E[:t],yt)
    pE.append({"ds":pd_,"y_true":actual,"pred":mE.predict(X_E[t:t+1])[0]})

    if (i+1)%10==0 or (i+1)==n_val_steps:
        print(f"  [{i+1}/{n_val_steps}] {time.time()-t0:.0f}s")

print(f"\nГотово за {time.time()-t0:.0f}s")

def cm(pl,n):
    d=pd.DataFrame(pl); yt=d["y_true"].values; yp=d["pred"].values
    m=np.isfinite(yt)&np.isfinite(yp); yt,yp=yt[m],yp[m]
    return {"model":n,"MAE":round(np.mean(np.abs(yt-yp)),4),
            "RMSE":round(np.sqrt(np.mean((yt-yp)**2)),4),
            "MAPE_%":round(np.mean(np.abs((yt-yp)/yt))*100,2),"n_forecasts":int(m.sum())}

res = [cm(pA,f"Naive h={HORIZON}"),cm(pB,"LightGBM лаги"),cm(pC,"LightGBM oracle"),
       cm(pD,"LightGBM 2-stage"),cm(pE,"LightGBM ROISFIX")]
rdf = pd.DataFrame(res).sort_values("MAE")

print(f"\n{'Сценарий':25s} | {'MAE':>8s} | {'RMSE':>8s} | {'MAPE%':>8s}")
print("-"*55)
nm = [r for r in res if 'Naive' in r['model']][0]['MAE']
for _,r in rdf.iterrows():
    mk = " ★" if r['MAE']<nm and 'Naive' not in r['model'] else ""
    print(f"{r['model']:25s} | {r['MAE']:8.4f} | {r['RMSE']:8.4f} | {r['MAPE_%']:7.2f}%{mk}")

rdf.to_csv(f"{FIG_DIR}/h3_metrics_with_roisfix.csv",index=False)

nm_=next(r for r in res if 'Naive' in r['model'])
om=next(r for r in res if 'oracle' in r['model'])
dm=next(r for r in res if '2-stage' in r['model'])
lm=next(r for r in res if 'лаги' in r['model'])
rm=next(r for r in res if 'ROISFIX' in r['model'])

print(f"\nАНАЛИЗ:")
print(f"  Цена ошибки прогноза фичей: {dm['MAE']-om['MAE']:.4f}")
print(f"  ROISFIX vs 2-stage: {dm['MAE']-rm['MAE']:+.4f}")
print(f"  ROISFIX vs Naive: {rm['MAE']-nm_['MAE']:+.4f}")
print(f"  Oracle RMSE vs Naive RMSE: {om['RMSE']-nm_['RMSE']:+.4f}")

ch3 = {f"Naive h={HORIZON}":"#bdc3c7","LightGBM лаги":"#e74c3c",
       "LightGBM oracle":"#3498db","LightGBM 2-stage":"#2ecc71","LightGBM ROISFIX":"#9b59b6"}

fig,ax=plt.subplots(figsize=(16,7))
dA=pd.DataFrame(pA); dE=pd.DataFrame(pE); dC=pd.DataFrame(pC); dD=pd.DataFrame(pD)
ax.plot(dA["ds"],dA["y_true"],color='black',lw=2,label='Факт',zorder=5)
ax.plot(dA["ds"],dA["pred"],color='#bdc3c7',lw=1,alpha=.7,label=f'Naive (MAE={nm_["MAE"]:.3f})')
ax.plot(dE["ds"],dE["pred"],color='#9b59b6',lw=1.8,alpha=.9,label=f'ROISFIX (MAE={rm["MAE"]:.3f})')
ax.plot(dD["ds"],dD["pred"],color='#2ecc71',lw=1.2,alpha=.7,label=f'2-stage (MAE={dm["MAE"]:.3f})')
ax.plot(dC["ds"],dC["pred"],color='#3498db',lw=1,alpha=.5,ls='--',label=f'Oracle (MAE={om["MAE"]:.3f})')
ax.set_title(f'h={HORIZON}: пять сценариев'); ax.set_ylabel('% годовых')
ax.legend(loc='upper left',fontsize=9); ax.grid(True,alpha=.3)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/h3_01_forecasts_roisfix.png'); plt.close()
print("[OK] h3_01_forecasts_roisfix.png")

fig,axes=plt.subplots(1,2,figsize=(16,6))
for ax,metric,label in zip(axes,["MAE","RMSE"],["MAE","RMSE"]):
    s=rdf.sort_values(metric)
    bars=ax.barh(s["model"].values,s[metric].values,
                 color=[ch3.get(m,'#888') for m in s["model"].values],alpha=.85)
    for b,v in zip(bars,s[metric].values):
        ax.text(b.get_width()+.02,b.get_y()+b.get_height()/2,f'{v:.3f}',va='center',fontsize=11,fontweight='bold')
    ax.set_xlabel(label); ax.set_title(label); ax.invert_yaxis(); ax.grid(True,alpha=.3,axis='x')
plt.suptitle(f'h={HORIZON}: MAE и RMSE',fontsize=14,y=1.02)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/h3_02_comparison_roisfix.png'); plt.close()
print("[OK] h3_02_comparison_roisfix.png")



print(f"\n{'='*60}")
print(f"ИНТЕРВАЛЬНЫЕ ОЦЕНКИ (h={HORIZON})")
print("=" * 60)

lgb_lo = LGB_PARAMS.copy()
lgb_lo["objective"] = "quantile"
lgb_lo["alpha"] = 0.025
lgb_lo.pop("metric", None)

lgb_hi = LGB_PARAMS.copy()
lgb_hi["objective"] = "quantile"
lgb_hi["alpha"] = 0.975
lgb_hi.pop("metric", None)

ci_scenarios = {
    "LightGBM ROISFIX": (X_E, pE),
    "LightGBM oracle": (X_C, pC),
}

ci_results = {}

for scen_name, (X_data, preds_list) in ci_scenarios.items():
    print(f"\n  {scen_name}: квантильная регрессия...")
    ci_preds = []

    for i in range(n_val_steps):
        t = val_start + i
        yt = y_all[:t]

        m_lo = lgb.LGBMRegressor(**lgb_lo)
        m_lo.fit(X_data[:t], yt)
        p_lo = m_lo.predict(X_data[t:t+1])[0]

        m_hi = lgb.LGBMRegressor(**lgb_hi)
        m_hi.fit(X_data[:t], yt)
        p_hi = m_hi.predict(X_data[t:t+1])[0]

        ci_preds.append({
            "ds": dates_all[t],
            "y_true": y_all[t],
            "pred": preds_list[i]["pred"],
            "lower_95": p_lo,
            "upper_95": p_hi,
        })

    df_ci = pd.DataFrame(ci_preds)
    yt_ci = df_ci["y_true"].values
    lo_ci = df_ci["lower_95"].values
    hi_ci = df_ci["upper_95"].values

    inside = (yt_ci >= lo_ci) & (yt_ci <= hi_ci)
    coverage = inside.mean()
    avg_width = (hi_ci - lo_ci).mean()

    ci_results[scen_name] = {
        "df": df_ci,
        "coverage": coverage,
        "width": avg_width,
        "inside": inside,
    }

    print(f"    Coverage (95% CI): {coverage:.1%}")
    print(f"    Средняя ширина CI: {avg_width:.2f} п.п.")


fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, (scen_name, ci_data) in zip(axes, ci_results.items()):
    df_ci = ci_data["df"]
    cov = ci_data["coverage"]
    wid = ci_data["width"]
    outside = ~ci_data["inside"]

    color = "#9b59b6" if "ROISFIX" in scen_name else "#3498db"

    ax.plot(df_ci["ds"], df_ci["y_true"],
            color='black', linewidth=2, label='Факт', zorder=5)
    ax.plot(df_ci["ds"], df_ci["pred"],
            color=color, linewidth=1.3, alpha=0.8, label=scen_name)
    ax.fill_between(df_ci["ds"], df_ci["lower_95"], df_ci["upper_95"],
                    color=color, alpha=0.15,
                    label=f'95% CI (cov={cov:.1%}, ш={wid:.1f})')

    if outside.any():
        ax.scatter(df_ci["ds"][outside], df_ci["y_true"][outside],
                   color='red', s=40, zorder=6,
                   label=f'Вне CI ({outside.sum()})')

    ax.set_title(scen_name)
    ax.set_ylabel('% годовых')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle(f'Интервальные оценки h={HORIZON} (квантильная регрессия)',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/h3_04_intervals.png')
plt.close()
print(f"\n  [OK] h3_04_intervals.png")


print(f"\n  СВОДКА ИНТЕРВАЛЬНЫХ ОЦЕНОК:")
print(f"  {'Сценарий':25s} | {'Coverage':>10s} | {'Ширина CI':>10s}")
print(f"  {'-'*50}")
for scen_name, ci_data in ci_results.items():
    print(f"  {scen_name:25s} | {ci_data['coverage']:>9.1%} | {ci_data['width']:>8.2f} п.п.")

for scen_name, ci_data in ci_results.items():
    safe_name = scen_name.replace(" ", "_").lower()
    ci_data["df"].to_csv(f"{FIG_DIR}/h{HORIZON}_ci_{safe_name}.csv", index=False)


print(f"  ROISFIX MAE={rm['MAE']:.4f}, 2-stage MAE={dm['MAE']:.4f}, Naive MAE={nm_['MAE']:.4f}")
print(f"  Oracle MAE={om['MAE']:.4f}, RMSE={om['RMSE']:.4f} (лучше Naive RMSE={nm_['RMSE']:.4f} на {(1-om['RMSE']/nm_['RMSE'])*100:.1f}%)")
