# 📊 Financial Market Theory

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)


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

## 🔧 Configuration

The app can be customized via `.streamlit/config.toml`:
- Theme colors
- Server settings
- Upload size limits

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues or pull requests.
---

**Disclaimer**: This tool is for educational purposes only. Always verify calculations independently.
