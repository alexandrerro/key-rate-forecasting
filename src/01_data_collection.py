import pandas as pd
import numpy as np
import requests
import os
import io
import time
import warnings
warnings.filterwarnings('ignore')

from paths import DATA_DIR, FIG_DIR
OUTPUT_FILE = DATA_DIR / "dataset_monthly.csv"
DATE_START = "2013-09-01"
DATE_END = "2026-03-31"



def month_range(start="2013-09-01", end="2026-03-01"):
    """Полный месячный индекс."""
    return pd.date_range(start=start, end=end, freq="MS")


def merge_monthly(base, new_df, on="date"):
    """Левый merge по дате."""
    return base.merge(new_df, on=on, how="left")



print("=" * 60)
print("СОЗДАНИЕ БАЗОВОЙ СЕТКИ ДАТ")
print("=" * 60)

dates = month_range(DATE_START, DATE_END)
df = pd.DataFrame({"date": dates})
print(f"  Сетка: {df['date'].min():%Y-%m} — {df['date'].max():%Y-%m} ({len(df)} месяцев)")



print(f"\n{'='*60}")
print("БЛОК 1: КЛЮЧЕВАЯ СТАВКА ЦБ РФ")
print("=" * 60)

key_rate_changes = [
    ("2013-09-13", 5.50),
    ("2013-10-14", 5.50),
    ("2014-03-03", 7.00),
    ("2014-04-28", 7.50),
    ("2014-07-28", 8.00),
    ("2014-11-05", 9.50),
    ("2014-12-12", 10.50),
    ("2014-12-16", 17.00),
    ("2015-02-02", 15.00),
    ("2015-03-16", 14.00),
    ("2015-05-05", 12.50),
    ("2015-06-16", 11.50),
    ("2015-07-31", 11.00),
    ("2015-10-30", 11.00),
    ("2015-12-11", 11.00),
    ("2016-02-01", 11.00),
    ("2016-03-18", 11.00),
    ("2016-06-10", 10.50),
    ("2016-07-29", 10.50),
    ("2016-09-16", 10.00),
    ("2016-10-28", 10.00),
    ("2016-12-16", 10.00),
    ("2017-02-03", 10.00),
    ("2017-03-27", 9.75),
    ("2017-04-28", 9.25),
    ("2017-06-19", 9.00),
    ("2017-07-28", 9.00),
    ("2017-09-15", 8.50),
    ("2017-10-27", 8.25),
    ("2017-12-15", 7.75),
    ("2018-02-09", 7.50),
    ("2018-03-23", 7.25),
    ("2018-04-27", 7.25),
    ("2018-06-15", 7.25),
    ("2018-07-27", 7.25),
    ("2018-09-14", 7.50),
    ("2018-10-26", 7.50),
    ("2018-12-14", 7.75),
    ("2019-02-08", 7.75),
    ("2019-03-22", 7.75),
    ("2019-04-26", 7.75),
    ("2019-06-14", 7.50),
    ("2019-07-26", 7.25),
    ("2019-09-06", 7.00),
    ("2019-10-25", 6.50),
    ("2019-12-13", 6.25),
    ("2020-02-07", 6.00),
    ("2020-04-24", 5.50),
    ("2020-06-19", 4.50),
    ("2020-07-24", 4.25),
    ("2020-09-18", 4.25),
    ("2020-10-23", 4.25),
    ("2020-12-18", 4.25),
    ("2021-02-12", 4.25),
    ("2021-03-19", 4.50),
    ("2021-04-23", 5.00),
    ("2021-06-11", 5.50),
    ("2021-07-23", 6.50),
    ("2021-09-10", 6.75),
    ("2021-10-22", 7.50),
    ("2021-12-17", 8.50),
    ("2022-02-14", 9.50),
    ("2022-02-28", 20.00),
    ("2022-04-11", 17.00),
    ("2022-05-04", 14.00),
    ("2022-06-10", 11.00),
    ("2022-07-25", 8.00),
    ("2022-09-16", 8.00),
    ("2022-10-28", 7.50),
    ("2022-12-16", 7.50),
    ("2023-02-10", 7.50),
    ("2023-03-17", 7.50),
    ("2023-04-28", 7.50),
    ("2023-06-09", 7.50),
    ("2023-07-21", 8.50),
    ("2023-08-15", 12.00),
    ("2023-09-15", 13.00),
    ("2023-10-27", 15.00),
    ("2023-12-15", 16.00),
    ("2024-02-16", 16.00),
    ("2024-03-22", 16.00),
    ("2024-04-26", 16.00),
    ("2024-06-07", 16.00),
    ("2024-07-26", 18.00),
    ("2024-09-13", 19.00),
    ("2024-10-25", 21.00),
    ("2024-12-20", 21.00),
    ("2025-02-14", 21.00),
    ("2025-03-21", 21.00),
    ("2025-04-25", 21.00),
    ("2025-06-06", 21.00),
    ("2025-07-25", 21.00),
    ("2025-09-12", 21.00),
    ("2025-10-24", 21.00),
    ("2025-12-19", 21.00),
    ("2026-02-14", 21.00),
    ("2026-03-21", 15.00),
]

kr_changes = pd.DataFrame(key_rate_changes, columns=["date", "key_rate"])
kr_changes["date"] = pd.to_datetime(kr_changes["date"])
kr_changes = kr_changes.sort_values("date")

full_daily = pd.date_range("2013-09-01", "2026-03-31", freq="D")
kr_daily = pd.DataFrame({"date": full_daily})
kr_daily = kr_daily.merge(kr_changes, on="date", how="left")
kr_daily["key_rate"] = kr_daily["key_rate"].ffill()

kr_monthly = kr_daily.set_index("date").resample("ME").last()
kr_monthly.index = kr_monthly.index.to_period("M").to_timestamp()
kr_monthly.index.name = "date"
kr_monthly = kr_monthly.reset_index()

df = merge_monthly(df, kr_monthly)
print(f"  Ключевая ставка: {df['key_rate'].notna().sum()}/{len(df)} месяцев")
print(f"  Диапазон: {df['key_rate'].min()}% — {df['key_rate'].max()}%")


print(f"\n{'='*60}")
print("БЛОК 2: BRENT (FRED API)")
print("=" * 60)


FRED_API_KEY = os.environ["FRED_API_KEY"]

try:
    if FRED_API_KEY:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        brent_daily = fred.get_series("DCOILBRENTEU",
                                       observation_start="2013-09-01",
                                       observation_end="2026-03-31")
        brent_monthly = brent_daily.resample("ME").mean()
        brent_monthly.index = brent_monthly.index.to_period("M").to_timestamp()
        brent_df = brent_monthly.reset_index()
        brent_df.columns = ["date", "brent"]
        print("  Brent загружен через FRED API")
    else:
        # Fallback: загрузка CSV с FRED
        url = ("https://fred.stlouisfed.org/graph/fredgraph.csv"
               "?id=DCOILBRENTEU&vintage_date=2026-04-01")
        resp = requests.get(url, timeout=30)
        brent_raw = pd.read_csv(io.StringIO(resp.text), parse_dates=["DATE"])
        brent_raw.columns = ["date", "brent"]
        brent_raw["brent"] = pd.to_numeric(brent_raw["brent"], errors="coerce")
        brent_raw = brent_raw.dropna()
        brent_monthly = brent_raw.set_index("date").resample("ME").mean()
        brent_monthly.index = brent_monthly.index.to_period("M").to_timestamp()
        brent_df = brent_monthly.reset_index()
        brent_df.columns = ["date", "brent"]
        print("  Brent загружен через FRED CSV URL")

    df = merge_monthly(df, brent_df)
    print(f"  Brent: {df['brent'].notna().sum()}/{len(df)} месяцев")

except Exception as e:
    print(f"  [!] Ошибка загрузки Brent: {e}")
    df["brent"] = np.nan


print(f"\n{'='*60}")
print("БЛОК 3: USD/RUB (ЦБ РФ)")
print("=" * 60)

try:
    url = (
        "https://www.cbr.ru/Queries/UniDbQuery/DownloadExcel/98956?"
        "Posted=True&From=01.09.2013&To=31.03.2026&mode=1&VAL_NM_RQ=R01235"
    )
    resp = requests.get(url, timeout=30)
    usd_raw = pd.read_excel(io.BytesIO(resp.content), skiprows=1)
    usd_raw.columns = ["date", "cobs", "usd_rub"]
    usd_raw["date"] = pd.to_datetime(usd_raw["date"], dayfirst=True, errors="coerce")
    usd_raw["usd_rub"] = pd.to_numeric(usd_raw["usd_rub"], errors="coerce")
    usd_raw = usd_raw[["date", "usd_rub"]].dropna()
    usd_monthly = usd_raw.set_index("date").resample("ME").last()
    usd_monthly.index = usd_monthly.index.to_period("M").to_timestamp()
    usd_df = usd_monthly.reset_index()
    usd_df.columns = ["date", "usd_rub"]
    df = merge_monthly(df, usd_df)
    print(f"  USD/RUB: {df['usd_rub'].notna().sum()}/{len(df)} месяцев")
except Exception as e:
    print(f"  [!] Ошибка загрузки USD/RUB: {e}")
    df["usd_rub"] = np.nan


print(f"\n{'='*60}")
print("БЛОК 4: M2 — Денежная масса (ЦБ РФ)")
print("=" * 60)


try:
    if os.path.exists("m2_raw.xlsx"):
        m2_raw = pd.read_excel("m2_raw.xlsx")
        m2_raw.columns = ["date", "m2"]
        m2_raw["date"] = pd.to_datetime(m2_raw["date"], errors="coerce")
        m2_raw["m2"] = pd.to_numeric(m2_raw["m2"], errors="coerce")
        m2_raw = m2_raw.dropna(subset=["date", "m2"])
        m2_raw["date"] = m2_raw["date"].dt.to_period("M").dt.to_timestamp()
        df = merge_monthly(df, m2_raw[["date", "m2"]])
        print(f"  M2: {df['m2'].notna().sum()}/{len(df)} месяцев")
    else:
        print("  [!] m2_raw.xlsx не найден — M2 = NaN")
        df["m2"] = np.nan
except Exception as e:
    print(f"  [!] Ошибка M2: {e}")
    df["m2"] = np.nan


print(f"\n{'='*60}")
print("БЛОК 5: CPI — ИПЦ (Росстат)")
print("=" * 60)


try:
    cpi_url = ("https://rosstat.gov.ru/storage/mediabank/ipc_mes_sub.xlsx")
    resp = requests.get(cpi_url, timeout=30)
    cpi_raw = pd.read_excel(io.BytesIO(resp.content), skiprows=3)
    cpi_col = [c for c in cpi_raw.columns if "федерация" in str(c).lower() or c == "Unnamed: 1"]
    if cpi_col:
        cpi_df = cpi_raw[[cpi_raw.columns[0], cpi_col[0]]].copy()
        cpi_df.columns = ["period", "cpi_mom"]
        cpi_df = cpi_df.dropna()
        cpi_df["cpi_mom"] = pd.to_numeric(cpi_df["cpi_mom"], errors="coerce")
        print(f"  CPI загружен: {cpi_df['cpi_mom'].notna().sum()} значений")
    else:
        print("  [!] Столбец CPI не найден — нужна ручная загрузка")
        df["cpi_mom"] = np.nan

except Exception as e:
    print(f"  [!] Ошибка загрузки CPI: {e}")
    df["cpi_mom"] = np.nan

    if os.path.exists("cpi_raw.xlsx"):
        cpi_raw = pd.read_excel("cpi_raw.xlsx")
        cpi_raw.columns = ["date", "cpi_mom"]
        cpi_raw["date"] = pd.to_datetime(cpi_raw["date"], errors="coerce")
        cpi_raw["cpi_mom"] = pd.to_numeric(cpi_raw["cpi_mom"], errors="coerce")
        cpi_raw["date"] = cpi_raw["date"].dt.to_period("M").dt.to_timestamp()
        df = merge_monthly(df, cpi_raw[["date", "cpi_mom"]])
        print(f"  CPI из файла: {df['cpi_mom'].notna().sum()}/{len(df)} месяцев")


print(f"\n{'='*60}")
print("БЛОК 6: ВВП (Росстат, квартальный → линейная интерполяция)")
print("=" * 60)

try:
    if os.path.exists("gdp_raw.xlsx"):
        gdp_raw = pd.read_excel("gdp_raw.xlsx")
        gdp_raw.columns = ["date", "gdp"]
        gdp_raw["date"] = pd.to_datetime(gdp_raw["date"], errors="coerce")
        gdp_raw["gdp"] = pd.to_numeric(gdp_raw["gdp"], errors="coerce")
        gdp_raw = gdp_raw.dropna()

        # Интерполяция: квартальный → месячный
        gdp_raw = gdp_raw.set_index("date").resample("MS").interpolate("linear")
        gdp_raw = gdp_raw.reset_index()
        gdp_raw.columns = ["date", "gdp"]

        df = merge_monthly(df, gdp_raw[["date", "gdp"]])
        print(f"  ВВП: {df['gdp'].notna().sum()}/{len(df)} месяцев (интерполирован)")
    else:
        print("  [!] gdp_raw.xlsx не найден — ВВП = NaN")
        df["gdp"] = np.nan
except Exception as e:
    print(f"  [!] Ошибка ВВП: {e}")
    df["gdp"] = np.nan


print(f"\n{'='*60}")
print("БЛОК 7: ИПП — Индекс промышленного производства (Росстат)")
print("=" * 60)

try:
    if os.path.exists("ipp_raw.xlsx"):
        ipp_raw = pd.read_excel("ipp_raw.xlsx")
        ipp_raw.columns = ["date", "ipp_yoy"]
        ipp_raw["date"] = pd.to_datetime(ipp_raw["date"], errors="coerce")
        ipp_raw["ipp_yoy"] = pd.to_numeric(ipp_raw["ipp_yoy"], errors="coerce")
        ipp_raw["date"] = ipp_raw["date"].dt.to_period("M").dt.to_timestamp()
        df = merge_monthly(df, ipp_raw[["date", "ipp_yoy"]])
        print(f"  ИПП: {df['ipp_yoy'].notna().sum()}/{len(df)} месяцев")
    else:
        print("  [!] ipp_raw.xlsx не найден")
        df["ipp_yoy"] = np.nan
except Exception as e:
    print(f"  [!] Ошибка ИПП: {e}")
    df["ipp_yoy"] = np.nan

print(f"\n{'='*60}")
print("БЛОК 8: КБД ОФЗ — Кривая бескупонной доходности (ЦБ РФ)")
print("=" * 60)

ZCYC_TENORS = [0.25, 0.5, 0.75, 1, 2, 3, 5, 7, 10, 15, 20, 30]

try:
    from bs4 import BeautifulSoup
    import re

    print("  Загрузка параметров КБД с cbr.ru...")
    zcyc_records = []

    start_dt = pd.Timestamp("2013-09-01")
    end_dt = pd.Timestamp("2026-03-31")
    current = start_dt

    def nelson_siegel(t, b0, b1, b2, tau):
        """Модель Нельсона-Сигеля для расчёта доходности."""
        x = t / tau
        e = np.exp(-x)
        return b0 + (b1 + b2) * (1 - e) / x - b2 * e

    session = requests.Session()

    while current <= end_dt:
        date_str = current.strftime("%d.%m.%Y")
        url = f"https://www.cbr.ru/hd_base/zcyc_params/?DateTo={date_str}"

        try:
            resp = session.get(url, timeout=20)
            soup = BeautifulSoup(resp.text, "lxml")
            table = soup.find("table")

            if table:
                rows = table.find_all("tr")
                for row in rows[2:]:  # Пропускаем заголовки
                    cells = row.find_all("td")
                    if len(cells) >= 6:
                        try:
                            date_cell = cells[0].get_text(strip=True)
                            b0 = float(cells[1].get_text(strip=True).replace(",", "."))
                            b1 = float(cells[2].get_text(strip=True).replace(",", "."))
                            b2 = float(cells[3].get_text(strip=True).replace(",", "."))
                            tau = float(cells[4].get_text(strip=True).replace(",", "."))
                            row_date = pd.to_datetime(date_cell, dayfirst=True)

                            rec = {"date": row_date}
                            for t in ZCYC_TENORS:
                                col = f"zcyc_{t}y".replace(".25", "0.25").replace(".5", "0.5").replace(".75", "0.75")
                                rec[col] = round(nelson_siegel(t, b0, b1, b2, tau), 4)
                            zcyc_records.append(rec)
                        except (ValueError, IndexError):
                            pass
        except Exception:
            pass

        current += pd.DateOffset(months=1)
        time.sleep(0.2)

    if zcyc_records:
        zcyc_df = pd.DataFrame(zcyc_records)
        zcyc_df["month"] = zcyc_df["date"].dt.to_period("M").dt.to_timestamp()
        zcyc_monthly = zcyc_df.groupby("month").last().reset_index()
        zcyc_monthly = zcyc_monthly.rename(columns={"month": "date"})
        zcyc_monthly = zcyc_monthly.drop(columns=["date_x", "date_y"], errors="ignore")

        rename_map = {}
        for t in ZCYC_TENORS:
            old = f"zcyc_{t}y"
            new = f"zcyc_{t}y"
            rename_map[old] = new
        zcyc_monthly = zcyc_monthly.rename(columns=rename_map)

        zcyc_cols = [f"zcyc_{t}y" for t in ZCYC_TENORS
                     if f"zcyc_{t}y" in zcyc_monthly.columns]
        df = merge_monthly(df, zcyc_monthly[["date"] + zcyc_cols])
        print(f"  КБД: {df['zcyc_1y'].notna().sum()}/{len(df)} месяцев")
    else:
        for t in ZCYC_TENORS:
            df[f"zcyc_{t}y"] = np.nan

except ImportError:
    for t in ZCYC_TENORS:
        df[f"zcyc_{t}y"] = np.nan
except Exception as e:
    print(f"  [!] Ошибка КБД: {e}")
    for t in ZCYC_TENORS:
        df[f"zcyc_{t}y"] = np.nan


print(f"\n{'='*60}")
print("БЛОК 9: СПРЕДЫ КБД")
print("=" * 60)

df["spread_10y_1y"] = df["zcyc_10y"] - df["zcyc_1y"]
df["spread_10y_3m"] = df["zcyc_10y"] - df.get("zcyc_0.25y", np.nan)
df["spread_5y_1y"] = df["zcyc_5y"] - df["zcyc_1y"]

print(f"  spread_10y_1y: {df['spread_10y_1y'].notna().sum()}/{len(df)} месяцев")
print(f"  spread_10y_3m: {df['spread_10y_3m'].notna().sum()}/{len(df)} месяцев")
print(f"  spread_5y_1y:  {df['spread_5y_1y'].notna().sum()}/{len(df)} месяцев")



print(f"\n{'='*60}")
print("БЛОК 10: RUONIA (ЦБ РФ)")
print("=" * 60)


try:
    if os.path.exists("ruonia.xlsx"):
        ruonia_raw = pd.read_excel("ruonia.xlsx", skiprows=1, header=None)
        ruonia_raw.columns = (["date", "ruonia"] +
                              [f"col_{i}" for i in range(ruonia_raw.shape[1] - 2)])
        ruonia_raw["date"] = pd.to_datetime(ruonia_raw["date"],
                                            format="%d-%m-%Y", errors="coerce")
        ruonia_raw["ruonia"] = (ruonia_raw["ruonia"].astype(str)
                                .str.replace(",", ".").str.strip())
        ruonia_raw["ruonia"] = pd.to_numeric(ruonia_raw["ruonia"], errors="coerce")
        ruonia_raw = ruonia_raw[["date", "ruonia"]].dropna().sort_values("date")

        ruonia_monthly = (ruonia_raw.set_index("date").resample("ME").last())
        ruonia_monthly.index = ruonia_monthly.index.to_period("M").to_timestamp()
        ruonia_df = ruonia_monthly.reset_index()
        ruonia_df.columns = ["date", "ruonia"]

        df = merge_monthly(df, ruonia_df)
        df["spread_ruonia_keyrate"] = df["ruonia"] - df["key_rate"]
        print(f"  RUONIA: {df['ruonia'].notna().sum()}/{len(df)} месяцев")
    else:
        print("  [!] ruonia.xlsx не найден")
        df["ruonia"] = np.nan
        df["spread_ruonia_keyrate"] = np.nan
except Exception as e:
    print(f"  [!] Ошибка RUONIA: {e}")
    df["ruonia"] = np.nan
    df["spread_ruonia_keyrate"] = np.nan


print(f"\n{'='*60}")
print("БЛОК 11: ROISFIX (roisfix.ru)")
print("=" * 60)

ROISFIX_TENORS = ["1w", "2w", "1m", "2m", "3m", "6m", "1y", "2y"]

try:
    roisfix_path = None
    for name in ["roisfix_raw.csv",
                  "2010-01-01_2026-04-01.csv",
                  "roisfix.csv"]:
        if os.path.exists(name):
            roisfix_path = name
            break

    if roisfix_path:
        roisfix_raw = pd.read_csv(roisfix_path, encoding="cp1251",
                                   sep=None, engine="python", skiprows=1)

        rename_rf = {"Дата ставки": "date",
                     "1W": "roisfix_1w", "2W": "roisfix_2w",
                     "1M": "roisfix_1m", "2M": "roisfix_2m",
                     "3M": "roisfix_3m", "6M": "roisfix_6m",
                     "1Y": "roisfix_1y", "2Y": "roisfix_2y"}
        roisfix_raw = roisfix_raw.rename(columns=rename_rf)

        if "Unnamed: 9" in roisfix_raw.columns:
            roisfix_raw = roisfix_raw.drop(columns=["Unnamed: 9"])

        roisfix_raw["date"] = pd.to_datetime(roisfix_raw["date"],
                                              format="%d-%m-%Y", errors="coerce")

        roisfix_cols = [f"roisfix_{t}" for t in ROISFIX_TENORS
                        if f"roisfix_{t}" in roisfix_raw.columns]
        for col in roisfix_cols:
            roisfix_raw[col] = (roisfix_raw[col].astype(str)
                                .str.replace(",", ".").str.strip())
            roisfix_raw[col] = pd.to_numeric(roisfix_raw[col], errors="coerce")
            roisfix_raw.loc[roisfix_raw[col] == 0, col] = np.nan

        roisfix_raw = roisfix_raw.dropna(subset=["date"]).sort_values("date")

        roisfix_monthly = (roisfix_raw.set_index("date")[roisfix_cols]
                           .resample("ME").last())
        roisfix_monthly.index = roisfix_monthly.index.to_period("M").to_timestamp()
        roisfix_df = roisfix_monthly.reset_index()
        roisfix_df = roisfix_df.rename(columns={"index": "date"})
        if roisfix_df.columns[0] != "date":
            roisfix_df.columns = ["date"] + list(roisfix_df.columns[1:])

        df = merge_monthly(df, roisfix_df)
        for t in ROISFIX_TENORS:
            col = f"roisfix_{t}"
            if col in df.columns:
                n = df[col].notna().sum()
                print(f"  {col}: {n}/{len(df)} ({n/len(df)*100:.0f}%)")
    else:
        print("  [!] Файл ROISFIX не найден")
        for t in ROISFIX_TENORS:
            df[f"roisfix_{t}"] = np.nan

except Exception as e:
    print(f"  [!] Ошибка ROISFIX: {e}")
    for t in ROISFIX_TENORS:
        df[f"roisfix_{t}"] = np.nan


print(f"\n{'='*60}")
print("БЛОК 12: ИТОГОВЫЙ ДАТАСЕТ")
print("=" * 60)

COLUMN_ORDER = [
    "date", "key_rate",
    "cpi_mom", "m2", "gdp", "ipp_yoy", "usd_rub", "brent",
    "zcyc_0.25y", "zcyc_0.5y", "zcyc_0.75y", "zcyc_1y",
    "zcyc_2y", "zcyc_3y", "zcyc_5y", "zcyc_7y",
    "zcyc_10y", "zcyc_15y", "zcyc_20y", "zcyc_30y",
    "spread_10y_1y", "spread_10y_3m", "spread_5y_1y",
    "ruonia", "spread_ruonia_keyrate",
    "roisfix_1w", "roisfix_2w", "roisfix_1m", "roisfix_2m",
    "roisfix_3m", "roisfix_6m", "roisfix_1y", "roisfix_2y",
]

for col in COLUMN_ORDER:
    if col not in df.columns:
        df[col] = np.nan

df = df[COLUMN_ORDER].sort_values("date").reset_index(drop=True)

df.to_csv(OUTPUT_FILE, index=False)

print(f"\n  Итоговый датасет: {df.shape[0]} строк × {df.shape[1]} столбцов")
print(f"  Период: {df['date'].min():%Y-%m} — {df['date'].max():%Y-%m}")
print(f"  Файл сохранён: {OUTPUT_FILE}")

print(f"\n  Покрытие:")
for col in COLUMN_ORDER:
    if col == "date":
        continue
    n = df[col].notna().sum()
    status = "OK" if n > 140 else ("ЧАСТИЧНО" if n > 60 else "ПУСТО")
    print(f"    [{status:8s}] {col:25s}: {n}/{len(df)} ({n/len(df)*100:.0f}%)")

