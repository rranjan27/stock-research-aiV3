import streamlit as st
import pandas as pd
import numpy as np

from financial_analysis import (
    fetch_financials_screener,
    compute_growth_rates,
    compute_annual_profitability_and_dupont,
    compute_annual_leverage,
    pitrovski_f_score,
    interpret_balance_sheet_strength,
    find_best_label,
)

st.set_page_config(page_title="Indian Stock Financial Analysis", layout="wide")
st.title("Financial Analysis Agent — Indian Stock (India-focused)")

ticker = st.text_input("Enter Screener.in ticker (e.g., TCS, RELIANCE)", value="TCS").strip().upper()

# Manual overrides
with st.expander("Manual label overrides (for profitability / DuPont)", expanded=False):
    override_net_profit = st.text_input("Net Profit / PAT label override", value="")
    override_ebit = st.text_input("EBIT / Operating Profit label override", value="")
    override_revenue = st.text_input("Revenue / Sales label override", value="")
    override_equity = st.text_input("Equity / Net Worth label override", value="")
    override_total_assets = st.text_input("Total Assets label override", value="")

with st.expander("Manual label overrides (for leverage)", expanded=False):
    override_equity_lev = st.text_input("Equity label override (leverage)", value="")
    override_ebitda = st.text_input("EBITDA label override", value="")
    override_op_profit = st.text_input("Operating Profit label override (if no EBITDA)", value="")
    override_depreciation = st.text_input("Depreciation label override", value="")
    override_cfo = st.text_input("Cash from Operations label override", value="")

if st.button("Run Financial Analysis"):
    try:
        st.info(f"Fetching financials for {ticker}...")
        data = fetch_financials_screener(ticker)
        pl = data.get("pl", pd.DataFrame())
        bs = data.get("bs", pd.DataFrame())
        cf = data.get("cf", pd.DataFrame())

        # Debug scraped row names
        st.subheader("Scraped Row Names")
        st.write("Profit & Loss rows:", list(pl.index))
        st.write("Balance Sheet rows:", list(bs.index))
        st.write("Cash Flow rows:", list(cf.index))

        # Display growth
        st.subheader("Growth Rates (quarterly)")
        growth = compute_growth_rates(pl)
        if not growth.empty:
            st.table(growth.round(2))
        else:
            st.warning("Could not compute growth rates (missing revenue or profit labels).")

        # Annual Profitability + DuPont
        st.subheader("Annual ROE / ROCE and DuPont (TTM-based, year-end)")
        annual_pd = compute_annual_profitability_and_dupont(
            pl,
            bs,
            net_profit_label_override=override_net_profit or None,
            ebit_label_override=override_ebit or None,
            revenue_label_override=override_revenue or None,
            equity_label_override=override_equity or None,
            total_assets_label_override=override_total_assets or None,
        )
        if not annual_pd.empty:
            st.table(annual_pd.round(2))
        else:
            st.warning("Insufficient data to compute annual ROE/ROCE/DuPont. Try overriding labels or ensure enough quarters exist.")

        # Annual Leverage
        st.subheader("Annual Leverage Ratios (Debt/Equity, Debt/EBITDA, CashFlow/EBITDA)")
        annual_lev = compute_annual_leverage(
            bs,
            pl,
            cf,
            equity_label_override=override_equity_lev or None,
            ebitda_label_override=override_ebitda or None,
            op_profit_label_override=override_op_profit or None,
            dep_label_override=override_depreciation or None,
            cfo_label_override=override_cfo or None,
        )
        if not annual_lev.empty:
            # clean infinities if any
            lev_display = annual_lev.replace([np.inf, -np.inf], pd.NA)
            st.table(lev_display.round(2))
        else:
            st.warning("Insufficient data for annual leverage metrics. Try overriding labels.")

        # Piotroski F-score
        st.subheader("Piotroski F-Score")
        fscore = pitrovski_f_score(pl, bs, cf)
        if not fscore.empty:
            st.table(fscore)
        else:
            st.warning("Piotroski score could not be computed (missing inputs).")

        # Balance sheet strength
        st.subheader("Balance Sheet Strength Snapshot")
        bs_summary = interpret_balance_sheet_strength(bs)
        st.json(bs_summary)

        # Summary
        st.subheader("Quick Summary")
        conclusions = []
        if not annual_pd.empty:
            if "ROE (%)" in annual_pd and not annual_pd["ROE (%)"].isnull().all():
                conclusions.append(f"Latest ROE (TTM): {annual_pd['ROE (%)'].dropna().iloc[-1]:.2f}%. ")
            if "ROCE (%)" in annual_pd and not annual_pd["ROCE (%)"].isnull().all():
                conclusions.append(f"Latest ROCE (TTM): {annual_pd['ROCE (%)'].dropna().iloc[-1]:.2f}%. ")
        if "Sales YoY (%)" in growth and not growth["Sales YoY (%)"].isnull().all():
            conclusions.append(f"Recent Sales growth (QoQ): {growth['Sales YoY (%)'].dropna().iloc[-1]:.2f}%. ")
        if "Net Profit YoY (%)" in growth and not growth["Net Profit YoY (%)"].isnull().all():
            conclusions.append(f"Recent Profit growth (QoQ): {growth['Net Profit YoY (%)'].dropna().iloc[-1]:.2f}%. ")

        if conclusions:
            st.success("".join(conclusions))
        else:
            st.info("Not enough clean data to form a summary. Try overriding labels or check scraped row names above.")

    except Exception as e:
        st.error(f"Error during analysis: {e}")
