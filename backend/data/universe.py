#change tickers to match portfolio of interest
tickers = [
    # Information Technology
    "AAPL", "NVDA", "AMD", "ZS",
    # Communication Services
    "GOOGL",
    # Consumer Discretionary
    "TSLA",
    # Consumer Staples
    "COST", "WMT",
    # Financials
    "JPM", "HOOD", "V",
    # Health Care
    "JNJ",
    # Industrials
    "AAL",
    # Energy
    "CVX",
    # Utilities
    "NEE",
    # --- added for diversification ---
    "PLD",   # Real Estate (REIT)
    "NEM",   # Materials (gold — low correlation to equities)
    "TLT",   # US Treasuries ETF (bond ballast; strongest diversifier)
]


benchmark = "SPY"

sector_map = {
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
