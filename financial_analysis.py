import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import difflib
import re
import calendar

# ---------- Helpers ----------

def clean_series(s):
    """
    Normalize a pandas Series or scalar like '₹ 1,234', '—', '1,234' to numeric floats.
    """
    if isinstance(s, pd.Series):
        cleaned = s.astype(str)
        cleaned = cleaned.str.replace("₹", "", regex=False)
        cleaned = cleaned.str.replace(",", "", regex=False)
        cleaned = cleaned.str.replace("—", "0", regex=False).str.replace("–", "0", regex=False)
        cleaned = cleaned.str.strip()
        return pd.to_numeric(cleaned, errors="coerce")
    else:
        try:
            return float(str(s).replace("₹", "").replace(",", "").replace("—", "0").replace("–", "0").strip())
        except Exception:
            return np.nan


def normalize_label(lbl):
    if lbl is None:
        return ""
    lbl = lbl.lower()
    lbl = re.sub(r"[^\w\s]", "", lbl)
    lbl = re.sub(r"\s+", " ", lbl).strip()
    return lbl


def find_best_label(df, keywords):
    """
    Return best-matching label from df.index based on substring then fuzzy match.
    """
    if df is None or df.empty:
        return None
    idxs = list(df.index)
    for kw in keywords:
        for idx in idxs:
            if kw.lower() in str(idx).lower():
                return idx
    # fuzzy fallback
    norm_map = {normalize_label(idx): idx for idx in idxs}
    for kw in keywords:
        norm_kw = normalize_label(kw)
        close = difflib.get_close_matches(norm_kw, list(norm_map.keys()), n=1, cutoff=0.6)
        if close:
            return norm_map[close[0]]
    return None


# ---------- Date/TTM helpers ----------

QUARTER_MAP = {"Mar": 3, "Jun": 6, "Sep": 9, "Dec": 12}

def parse_period_label_to_date(label):
    """
    Convert labels like "Jun 2025" into a Timestamp of the quarter end date (e.g., 2025-06-30).
    """
    parts = label.strip().split()
    if len(parts) != 2:
        return None
    mon_str, year_str = parts
    mon = mon_str[:3].title()
    if mon not in QUARTER_MAP:
        return None
    try:
        year = int(year_str)
    except:
        return None
    month = QUARTER_MAP[mon]
    day = calendar.monthrange(year, month)[1]
    return pd.Timestamp(year=year, month=month, day=day)


def to_quarterly_series(raw_series):
    """
    Takes a pandas Series with index like ['Jun 2022', ...] and returns a datetime-indexed numeric series.
    """
    data = {}
    for label, val in raw_series.items():
        dt = parse_period_label_to_date(label)
        if dt is None:
            continue
        num = clean_series(val)
        data[dt] = num
    if not data:
        return pd.Series(dtype=float)
    s = pd.Series(data).sort_index()
    return s


def compute_ttm(series):
    """
    Trailing twelve months: rolling sum over last 4 quarters.
    """
    return series.rolling(window=4, min_periods=4).sum()


# ---------- Web scraping (Screener.in) ----------

def _get_screener_page(ticker):
    url = f"https://www.screener.in/company/{ticker}/consolidated/"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


def _parse_table(soup, section_name):
    for header in soup.find_all(["h2", "h3"]):
        if section_name.lower() in header.get_text(strip=True).lower():
            table = header.find_next("table")
            if table:
                df = pd.read_html(str(table))[0]
                df = df.set_index(df.columns[0])
                df.columns = df.columns.astype(str)
                return df
    # fallback
    tables = soup.find_all("table")
    if tables:
        df = pd.read_html(str(tables[0]))[0]
        df = df.set_index(df.columns[0])
        df.columns = df.columns.astype(str)
        return df
    return pd.DataFrame()


def fetch_financials_screener(ticker):
    soup = _get_screener_page(ticker)
    pl = _parse_table(soup, "profit and loss")
    bs = _parse_table(soup, "balance sheet")
    cf = _parse_table(soup, "cash flow")
    return {"pl": pl, "bs": bs, "cf": cf}


# ---------- Metrics ----------

def compute_growth_rates(pl_df):
    sales_label = find_best_label(pl_df, ["total income", "sales", "revenue"])
    profit_label = find_best_label(pl_df, ["profit after tax", "net profit", "pat"])
    growth = {}
    if sales_label:
        sales = clean_series(pl_df.loc[sales_label])
        growth["Sales YoY (%)"] = sales.pct_change() * 100
    if profit_label:
        profit = clean_series(pl_df.loc[profit_label])
        growth["Net Profit YoY (%)"] = profit.pct_change() * 100
    return pd.DataFrame(growth)


def compute_annual_profitability_and_dupont(
    pl_df,
    bs_df,
    net_profit_label_override=None,
    ebit_label_override=None,
    revenue_label_override=None,
    equity_label_override=None,
    total_assets_label_override=None,
):
    """
    Computes annual ROE, ROCE, and DuPont components at year-end (March) using TTM figures.
    """
    # Label resolution with overrides
    net_profit_label = net_profit_label_override or find_best_label(pl_df, ["profit after tax", "net profit", "pat"])
    ebit_label = ebit_label_override or find_best_label(pl_df, ["ebit", "operating profit", "operating income"])
    revenue_label = revenue_label_override or find_best_label(pl_df, ["total income", "sales", "revenue"])
    equity_label = equity_label_override or find_best_label(bs_df, ["shareholders' funds", "total equity", "net worth", "equity"])
    total_assets_label = total_assets_label_override or find_best_label(bs_df, ["total assets"])

    if not all([net_profit_label, ebit_label, revenue_label, equity_label, total_assets_label]):
        return pd.DataFrame()

    # Quarterly TTM series
    net_profit_q = to_quarterly_series(pl_df.loc[net_profit_label])
    ebit_q = to_quarterly_series(pl_df.loc[ebit_label])
    revenue_q = to_quarterly_series(pl_df.loc[revenue_label])

    net_profit_ttm = compute_ttm(net_profit_q)
    ebit_ttm = compute_ttm(ebit_q)
    revenue_ttm = compute_ttm(revenue_q)

    # Annual balance-sheet items (March)
    equity_ann = to_quarterly_series(bs_df.loc[equity_label])
    assets_ann = to_quarterly_series(bs_df.loc[total_assets_label])

    # Annual debt aggregation
    debt_labels = [idx for idx in bs_df.index if any(k in idx.lower() for k in ["borrowings", "debt"])]
    debt_ann = {}
    for d in debt_labels:
        series = to_quarterly_series(bs_df.loc[d])
        for dt, val in series.items():
            if dt not in debt_ann:
                debt_ann[dt] = 0
            debt_ann[dt] += val
    debt_ann = pd.Series(debt_ann).sort_index()

    rows = []
    for date in sorted(equity_ann.index):
        if date.month != 3:  # only year-end March
            continue
        prev_year = date - pd.DateOffset(years=1)
        if prev_year not in equity_ann.index or prev_year not in assets_ann.index:
            continue
        if date not in net_profit_ttm.index or date not in revenue_ttm.index or date not in ebit_ttm.index:
            continue

        avg_equity = (equity_ann.loc[date] + equity_ann.loc[prev_year]) / 2
        avg_assets = (assets_ann.loc[date] + assets_ann.loc[prev_year]) / 2

        roe = (net_profit_ttm.loc[date] / avg_equity) * 100 if avg_equity and avg_equity != 0 else np.nan
        debt_at_date = debt_ann.get(date, 0)
        capital_employed = avg_equity + debt_at_date
        roce = (ebit_ttm.loc[date] / capital_employed) * 100 if capital_employed and capital_employed != 0 else np.nan

        profit_margin = (net_profit_ttm.loc[date] / revenue_ttm.loc[date]) * 100 if revenue_ttm.loc[date] and revenue_ttm.loc[date] != 0 else np.nan
        asset_turnover = revenue_ttm.loc[date] / avg_assets if avg_assets and avg_assets != 0 else np.nan
        equity_multiplier = avg_assets / avg_equity if avg_equity and avg_equity != 0 else np.nan
        implied_roe = (
            (profit_margin / 100) * asset_turnover * equity_multiplier * 100
            if all(v is not None for v in [profit_margin, asset_turnover, equity_multiplier])
            else np.nan
        )

        rows.append({
            "Date": date,
            "ROE (%)": roe,
            "ROCE (%)": roce,
            "Profit Margin (%)": profit_margin,
            "Asset Turnover": asset_turnover,
            "Equity Multiplier": equity_multiplier,
            "Implied ROE (%)": implied_roe,
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return df


def compute_annual_leverage(
    bs_df,
    pl_df,
    cf_df,
    equity_label_override=None,
    ebitda_label_override=None,
    op_profit_label_override=None,
    dep_label_override=None,
    cfo_label_override=None,
):
    """
    Computes annual leverage-related ratios (Debt/Equity, Debt/EBITDA, CashFlow/EBITDA) at March year-ends.
    Supports overrides for key labels.
    """
    equity_label = equity_label_override or find_best_label(bs_df, ["shareholders' funds", "total equity", "net worth", "equity"])
    debt_labels = [idx for idx in bs_df.index if any(k in idx.lower() for k in ["borrowings", "debt"])]

    equity_ann = to_quarterly_series(bs_df.loc[equity_label]) if equity_label else pd.Series(dtype=float)

    # Aggregate annual debt
    debt_ann = {}
    for d in debt_labels:
        series = to_quarterly_series(bs_df.loc[d])
        for dt, val in series.items():
            if dt not in debt_ann:
                debt_ann[dt] = 0
            debt_ann[dt] += val
    debt_ann = pd.Series(debt_ann).sort_index()

    # EBITDA TTM
    if ebitda_label_override:
        ebitda_q = to_quarterly_series(pl_df.loc[ebitda_label_override])
    else:
        explicit_ebitda_label = find_best_label(pl_df, ["ebitda"])
        op_profit_label = op_profit_label_override or find_best_label(pl_df, ["ebit", "operating profit", "operating income"])
        dep_label = dep_label_override or find_best_label(pl_df, ["depreciation", "depreciation and amortisation", "depreciation & amortisation"])
        if explicit_ebitda_label:
            ebitda_q = to_quarterly_series(pl_df.loc[explicit_ebitda_label])
        else:
            ebit_q = to_quarterly_series(pl_df.loc[op_profit_label]) if op_profit_label else pd.Series(dtype=float)
            dep_q = to_quarterly_series(pl_df.loc[dep_label]) if dep_label else pd.Series(dtype=float)
            ebitda_q = ebit_q + dep_q
    ebitda_ttm = compute_ttm(ebitda_q)

    # Cash Flow from Operations TTM
    cfo_label = cfo_label_override or find_best_label(cf_df, ["cash from operating activities", "net cash from operating activities", "cash from operations"])
    cfo_q = to_quarterly_series(cf_df.loc[cfo_label]) if cfo_label else pd.Series(dtype=float)
    cfo_ttm = compute_ttm(cfo_q)

    rows = []
    for date in sorted(equity_ann.index):
        if date.month != 3:
            continue
        if date not in debt_ann.index or date not in ebitda_ttm.index or date not in cfo_ttm.index:
            continue
        if date not in equity_ann.index:
            continue

        equity = equity_ann.loc[date]
        debt = debt_ann.loc[date]
        debt_equity = debt / equity if equity and equity != 0 else np.nan
        debt_ebitda = debt / ebitda_ttm.loc[date] if ebitda_ttm.loc[date] and ebitda_ttm.loc[date] != 0 else np.nan
        cashflow_ebitda = cfo_ttm.loc[date] / ebitda_ttm.loc[date] if ebitda_ttm.loc[date] and ebitda_ttm.loc[date] != 0 else np.nan

        rows.append({
            "Date": date,
            "Debt/Equity": debt_equity,
            "Debt/EBITDA": debt_ebitda,
            "CashFlow/EBITDA": cashflow_ebitda,
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return df


def pitrovski_f_score(pl_df, bs_df, cf_df):
    years = list(pl_df.columns) if not pl_df.empty else []
    score = {year: 0 for year in years}

    net_profit_label = find_best_label(pl_df, ["profit after tax", "net profit", "pat"])
    operating_cf_label = find_best_label(cf_df, ["cash from operating activities", "net cash from operating activities", "cash from operations"])
    total_assets_label = find_best_label(bs_df, ["total assets"])
    current_assets_label = find_best_label(bs_df, ["current assets"])
    current_liabilities_label = find_best_label(bs_df, ["current liabilities"])
    revenue_label = find_best_label(pl_df, ["total income", "sales", "revenue"])
    cogs_label = find_best_label(pl_df, ["cost of goods sold", "raw materials", "direct cost"])

    net_profit = clean_series(pl_df.loc[net_profit_label]) if net_profit_label else None
    operating_cf = clean_series(cf_df.loc[operating_cf_label]) if operating_cf_label else None
    total_assets = clean_series(bs_df.loc[total_assets_label]) if total_assets_label else None
    current_assets = clean_series(bs_df.loc[current_assets_label]) if current_assets_label else None
    current_liabilities = clean_series(bs_df.loc[current_liabilities_label]) if current_liabilities_label else None
    revenue = clean_series(pl_df.loc[revenue_label]) if revenue_label else None
    cogs = clean_series(pl_df.loc[cogs_label]) if cogs_label else None

    debt_labels = [idx for idx in bs_df.index if any(k in idx.lower() for k in ["borrowings", "debt"])]
    total_debt = pd.Series(0, index=total_assets.index) if total_assets is not None else None
    for d in debt_labels:
        if d in bs_df.index:
            total_debt = total_debt + clean_series(bs_df.loc[d]).fillna(0)

    roa = None
    if net_profit is not None and total_assets is not None:
        roa = net_profit / total_assets.replace({0: np.nan})

    leverage = None
    if total_debt is not None and total_assets is not None:
        leverage = total_debt / total_assets.replace({0: np.nan})

    gross_margin = None
    if revenue is not None and cogs is not None:
        gross_margin = (revenue - cogs) / revenue.replace({0: np.nan})

    asset_turnover = None
    if revenue is not None and total_assets is not None:
        asset_turnover = revenue / total_assets.replace({0: np.nan})

    for i, year in enumerate(years):
        pts = 0
        if net_profit is not None and not pd.isna(net_profit.get(year, np.nan)) and net_profit.get(year, 0) > 0:
            pts += 1
        if operating_cf is not None and not pd.isna(operating_cf.get(year, np.nan)) and operating_cf.get(year, 0) > 0:
            pts += 1
        if roa is not None and i > 0:
            prev = years[i - 1]
            if not pd.isna(roa.get(year, np.nan)) and not pd.isna(roa.get(prev, np.nan)):
                if roa.get(year, 0) > roa.get(prev, 0):
                    pts += 1
        if operating_cf is not None and net_profit is not None:
            if net_profit.get(year, 0) != 0 and operating_cf.get(year, 0) / net_profit.get(year, 1) > 1:
                pts += 1
        if leverage is not None and i > 0:
            prev = years[i - 1]
            if not pd.isna(leverage.get(year, np.nan)) and not pd.isna(leverage.get(prev, np.nan)):
                if leverage.get(year, 0) < leverage.get(prev, 0):
                    pts += 1
        if current_assets is not None and current_liabilities is not None and i > 0:
            def cr(col):
                ca = current_assets.get(col, np.nan)
                cl = current_liabilities.get(col, np.nan)
                if cl == 0 or pd.isna(cl):
                    return np.nan
                return ca / cl
            curr_cr = cr(year)
            prev_cr = cr(years[i - 1])
            if not pd.isna(curr_cr) and not pd.isna(prev_cr) and curr_cr > prev_cr:
                pts += 1
        if gross_margin is not None and i > 0:
            prev = years[i - 1]
            if not pd.isna(gross_margin.get(year, np.nan)) and not pd.isna(gross_margin.get(prev, np.nan)):
                if gross_margin.get(year, 0) > gross_margin.get(prev, 0):
                    pts += 1
        if asset_turnover is not None and i > 0:
            prev = years[i - 1]
            if not pd.isna(asset_turnover.get(year, np.nan)) and not pd.isna(asset_turnover.get(prev, np.nan)):
                if asset_turnover.get(year, 0) > asset_turnover.get(prev, 0):
                    pts += 1

        score[year] = pts

    score_df = pd.DataFrame.from_dict(score, orient="index", columns=["Piotroski F-Score"])
    score_df.index.name = "Year"
    return score_df


def interpret_balance_sheet_strength(bs_df):
    equity_label = find_best_label(bs_df, ["shareholders' funds", "total equity", "net worth", "equity"])
    debt_labels = [idx for idx in bs_df.index if any(k in idx.lower() for k in ["borrowings", "debt"])]

    equity_latest = None
    if equity_label:
        equity_series = clean_series(bs_df.loc[equity_label])
        if not equity_series.dropna().empty:
            val = equity_series.dropna().iloc[-1]
            if hasattr(val, "item"):
                val = val.item()
            equity_latest = val

    total_debt_series = None
    if debt_labels:
        summed = None
        for d in debt_labels:
            if d in bs_df.index:
                series = clean_series(bs_df.loc[d]).fillna(0)
                summed = series if summed is None else summed + series
        total_debt_series = summed

    total_debt_latest = None
    if total_debt_series is not None and not total_debt_series.dropna().empty:
        val = total_debt_series.dropna().iloc[-1]
        if hasattr(val, "item"):
            val = val.item()
        total_debt_latest = val

    return {
        "Equity (latest)": equity_latest,
        "Total Debt (latest)": total_debt_latest,
    }
