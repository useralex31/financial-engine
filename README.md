# 📊 Financial Market Theory Exam Engine

A comprehensive, interactive exam preparation tool for Financial Market Theory (FMT) courses. Built with Streamlit for easy deployment and use.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 🎯 Features

### Module 1: Portfolio Optimizer
- Mean-Variance Optimization with optional ESG constraints
- Tangency Portfolio & Capital Market Line
- Optimal Risky Portfolio calculations
- Multi-asset portfolio construction with constraints
- Support for short-selling restrictions, leverage caps, and position limits

### Module 2: Bond Math
- Duration & Convexity calculations
- Bond pricing and yield-to-maturity
- Price sensitivity analysis (Δy shocks)
- Immunization strategies
- Horizon analysis

### Module 3: Stock Valuation
- Dividend Discount Models (Gordon Growth, Multi-stage)
- P/E ratio analysis
- Free Cash Flow valuation
- Terminal value calculations

### Module 4: Human Capital
- Human Capital valuation as bond/stock-like asset
- Labor income hedging strategies
- Total wealth portfolio optimization
- Lifecycle investment considerations

### Module 5: Factor Models
- CAPM: Beta, Alpha, and Security Market Line
- Multi-factor models (Fama-French)
- APT (Arbitrage Pricing Theory)
- Factor risk decomposition
- Reverse APT problems

### Module 6: Probability
- Normal distribution calculations
- Portfolio VaR and CVaR
- Shortfall probability
- Joint probability scenarios

### Module 7: Universal Solver
- Automated problem detection
- Step-by-step solutions
- Formula display with LaTeX

### Module 8: Exam Triage
- Goal-based problem routing
- Quick identification of problem types
- Guided solution paths

## 🚀 Quick Start

### Option 1: Use Online (Streamlit Cloud)
Visit the deployed app: [Your Streamlit App URL]

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/fmt-exam-engine.git
cd fmt-exam-engine

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📋 Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list

### Core Dependencies
- `streamlit` - Web application framework
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scipy` - Scientific computing & optimization
- `matplotlib` - Visualization
- `cvxpy` (optional) - Convex optimization for advanced portfolio problems

## 📖 Usage Tips

1. **Input Formats**: The app accepts flexible input formats:
   - Decimals: `0.08, 0.15`
   - Percentages: `8%, 15%`
   - Space-separated: `0.08 0.15`
   - Mixed: `8% 0.15 12`

2. **Copy from PDFs**: Paste data directly from exam PDFs - the parser handles most formats automatically.

3. **Excel Formulas**: Click the 📝 expanders to see equivalent Excel formulas for manual verification.

4. **Module Selection**: Use the sidebar to navigate between modules. Each module has multiple tabs for different problem types.

5. **Exam Triage**: Don't know where to start? Use Module 8 to identify the problem type and get routed to the right solver.

## 📚 Covered Exam Topics (2023-2025)

| Topic | Module | Coverage |
|-------|--------|----------|
| Mean-Variance Optimization | 1 | ✅ Full |
| CAPM & Factor Models | 5 | ✅ Full |
| Bond Duration & Convexity | 2 | ✅ Full |
| Human Capital | 4 | ✅ Full |
| ESG Portfolio Constraints | 1 | ✅ Full |
| Dividend Discount Models | 3 | ✅ Full |
| APT & Arbitrage | 5 | ✅ Full |
| Probability Calculations | 6 | ✅ Full |

## 🔧 Configuration

The app can be customized via `.streamlit/config.toml`:
- Theme colors
- Server settings
- Upload size limits

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Disclaimer**: This tool is for educational purposes only. Always verify calculations independently for actual exam preparation.
