import pandas as pd


weight_dict = {
    "AAPL": "Information Technology",
    "NVDA": "Information Technology",
    "AMD":  "Information Technology",
    "ZS":   "Information Technology",
    "GOOGL": "Communication Services",
    "TSLA": "Consumer Discretionary",
    "COST": "Consumer Staples",
    "WMT":  "Consumer Staples",
    "JPM":  "Financials",
    "HOOD": "Financials",
    "V":    "Financials",
    "JNJ":  "Health Care",
    "AAL":  "Industrials",
    "CVX":  "Energy",
    "NEE":  "Utilities",
    "PLD":  "Real Estate",
    "NEM":  "Materials",
    "TLT":  "Fixed Income",
}

# Equal-weight portfolio: 1/n for each holding, summing to 1.
weights = pd.Series(1 / len(weight_dict), index=list(weight_dict))
