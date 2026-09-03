import streamlit as st

categorized_tickers = {
    "Benchmark": [
        "^GSPC",
        "^IXIC",
        "^DJI",
        "^RUT",
        "^VIX",
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "ACWI",
    ],
    "Big Tech": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "PLTR",
        "NFLX",
        "AMD",
    ],
    "Finance": [
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "BLK",
        "AXP",
        "V",
        "MA",
        "BRK-B",
    ],
    "Healthcare": [
        "JNJ",
        "UNH",
        "PFE",
        "ABBV",
        "MRK",
        "LLY",
        "TMO",
        "ABT",
        "AMGN",
        "GILD",
    ],
    "Consumer Goods": [
        "WMT",
        "PG",
        "KO",
        "PEP",
        "COST",
        "MCD",
        "DIS",
        "NKE",
        "SBUX",
        "HD",
    ],
    "Energy & Industry": [
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",
        "BA",
        "CAT",
        "HON",
        "UPS",
        "GE",
    ],
}

# --- CALLBACKs for widget synchronization ---
def sync_from_multiselect():
    """Updates the state of pills based on what is selected in the multiselect widget"""
    current_selected = set(st.session_state["selected_tickers"])
    for cat, tickers in categorized_tickers.items():
        st.session_state[f"pills_{cat}"] = [t for t in tickers if t in current_selected]

def sync_from_pills(cat, cat_tickers):
    """Updates the multiselect widget based on the selections made in the category pills"""
    pill_selected = set(st.session_state[f"pills_{cat}"])
    current_multi = set(st.session_state["selected_tickers"])
    
    # Remove all tickers belonging to this category from the overall list
    current_multi -= set(cat_tickers)
    # Add back the ones that are currently selected in the clicked category
    current_multi |= pill_selected
    
    st.session_state["selected_tickers"] = list(current_multi)