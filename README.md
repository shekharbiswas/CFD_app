# CFD Hedging & VIX Momentum Strategy Simulation App

## Overview

This Streamlit application simulates and compares two investment portfolio strategies:

*   **Model A (Classic):** A traditional portfolio with a fixed **80% S&P 500 equities and 20% cash** allocation.
*   **Model B (VIX Momentum Hedge):** A portfolio with the same fixed 80/20 base equity/cash allocation, but which dynamically activates **short S&P 500 CFD positions** as a hedge. The decision to hedge or cover these CFD positions is driven by a **VIX momentum strategy**, which considers recent VIX percentage changes and absolute VIX levels.

Users can adjust various configuration parameters related to Model B's hedging strategy and CFD costs via an interactive sidebar to observe their impact on overall portfolio performance and risk characteristics. The simulation uses historical market data.

[_Project Repository for Detailed Analysis Scripts_](https://github.com/shekharbiswas/CFD_Simulation)
[_This Streamlit App Repository_](https://github.com/shekharbiswas/CFD_app)

## Features

*   Interactive sidebar to configure simulation parameters for:
    *   Initial Capital.
    *   **Model B's VIX Momentum Strategy:**
        *   CFD Hedge Ratio (proportion of equity to hedge).
        *   VIX Absolute Cover Threshold.
        *   *(Other VIX momentum parameters like lookback period, percentage change thresholds for short/cover, and consecutive days are currently fixed in `scripts/config.py` but could be added to the UI for more advanced control).*
    *   **CFD Costs (for Model B):**
        *   Broker Annual Financing Fee (markup/markdown relative to SOFR).
        *   Spread Cost (in S&P 500 points per contract transaction).
        *   CFD Initial Margin Percent.
*   Fixed 80% Equity / 20% Cash base allocation for both Model A and Model B.
*   Visual comparison of Model A vs. Model B portfolio values over time using an interactive Plotly chart.
*   Summary metrics (Final Value, Total Return) for both portfolios displayed directly in the app.
*   Highlighting of predefined analysis periods (e.g., COVID crisis, VIX spikes) on the performance plot.

## 📁 Project Structure

The project is organized as follows:

```bash
CFD_app/
├── .devcontainer/ # Development container configuration (if used)
├── data/ # Potentially for other data, currently primary CSV is at root
├── scripts/ # Backend Python modules
│ ├── config.py
│ ├── signal_generation.py
│ ├── simulation_engine.py
│ └── (risk_metrics.py, data_loader.py - optional for future use)
├── .gitignore
├── README.md # This file
├── app.py # Main Streamlit application script
├── requirements.txt # Python dependencies
└── vix_sp500_data.csv # Primary historical data file
```



## 📜 Script Breakdown

*   **`app.py`**: The main Streamlit application script that provides the user interface and orchestrates the simulation.
*   **`vix_sp500_data.csv`**: The historical data file for S&P 500 prices, VIX levels, SOFR rates, and pre-calculated S&P 500 returns. Located in the project root.
*   **`scripts/config.py`**: Stores static configuration parameters not controlled by the UI (e.g., VIX momentum rule details, fixed portfolio labels, plot colors).
*   **`scripts/signal_generation.py`**: Implements the VIX momentum strategy to generate daily trading signals for Model B.
*   **`scripts/simulation_engine.py`**: Contains the core functions (`simulate_portfolio_A` and `simulate_portfolio_B_momentum`) for running the day-by-day portfolio simulations.
*   **`scripts/risk_metrics.py`**: (Currently not fully integrated into Streamlit for all metrics display) Provides functions for calculating a comprehensive set of financial risk and performance metrics.
*   **`scripts/data_loader.py`**: (Currently, data loading for the main CSV is handled in `app.py`) This module would be used if fetching from an API or more complex local data processing were re-enabled.
*   **`requirements.txt`**: Lists the Python libraries required to run the app.

## ⚙️ Setup and Configuration

1.  **API Key (Only if FMP fetching is re-enabled in `scripts/data_loader.py`):**
    *   This project primarily uses the local `vix_sp500_data.csv`. However, if you modify `scripts/data_loader.py` to fetch live data from Financial Modeling Prep (FMP), you will need an FMP API key.
    *   If using FMP, open `scripts/config.py` (or a dedicated secrets management file).
    *   Replace the placeholder value for `FMP_API_KEY` with your actual FMP API key.

2.  **Data File (`vix_sp500_data.csv`):**
    *   Ensure the `vix_sp500_data.csv` file is present in the **root directory** of the project (alongside `app.py`).
    *   This CSV **must** contain the following columns: `date`, `S&P500`, `VIX`, `SOFR_Rate`, `SP500_Return`, `Prev_S&P500`.
    *   The `app.py` script is configured to load data from this location.

3.  **Review `scripts/config.py` (For Advanced/Static Parameters):**
    *   While many parameters are adjustable via the Streamlit UI, some core VIX momentum rules and plotting defaults are set in `scripts/config.py`. Review this file for any deeper configuration needs.

## 🚀 How to Run Locally

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/shekharbiswas/CFD_app.git # Or your specific Streamlit app repo URL
    cd CFD_app
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Ensure you have Python 3.8+ installed. Then, from the project root directory:
    ```bash
    pip install -r requirements.txt
    ```
    (If `requirements.txt` is missing, manually install: `pip install streamlit pandas numpy plotly`)

4.  **Ensure Data and Scripts are in Place:**
    *   `vix_sp500_data.csv` should be in the `CFD_app` root directory.
    *   The `scripts` folder (containing `config.py`, `signal_generation.py`, `simulation_engine.py`) should be in the `CFD_app` root directory.

5.  **Run the Streamlit App:**
    From the root directory of the cloned repository (`CFD_app/`), execute:
    ```bash
    streamlit run app.py
    ```

6.  **View in Browser:**
    Streamlit will typically open the app automatically in your default web browser (e.g., `http://localhost:8501`).

## 🛠 Usage

*   Use the **sidebar** on the left to adjust configurable parameters for the simulation (Initial Capital, Model B Hedge Ratio, VIX Absolute Cover Threshold, CFD Costs).
*   The base **Equity/Cash allocation is fixed at 80%/20%** for both models.
*   Click "**🚀 Run Simulation & Plot**".
*   View the interactive Plotly chart comparing portfolio values and summary metrics.

## 📊 Expected Output

*   Console messages (if any, mostly for debugging if run locally outside Streamlit's direct management).
*   An interactive Streamlit web page displaying:
    *   Portfolio performance comparison chart.
    *   Summary metrics (Final Value, Total Return).
    *   Highlighted crisis periods on the chart.

## ⚡ Troubleshooting

*   **`FileNotFoundError: [Errno 2] No such file or directory: 'vix_sp500_data.csv'`**: Ensure `vix_sp500_data.csv` is in the project root directory (where `app.py` is located).
*   **Module Not Found (e.g., `No module named 'scripts.config'`):** Ensure the `scripts` folder is in the same directory as `app.py` and that `scripts` contains an `__init__.py` file (even if empty) to be treated as a package (though often not strictly necessary for direct script imports if `app.py` and `scripts/` are in the same root and Python's path is set up as expected).
*   **Plotly plots not showing**: Check browser pop-up blockers if running locally. Streamlit usually handles rendering well.
*   **`KeyError` or `AttributeError`**: Often due to misnamed columns in the CSV, incorrect keys in `default_config_ui` within `app.py`, or missing parameters in the `cfg_object` passed to backend modules. Check error messages for specifics.
*   **Data Issues**: If the `vix_sp500_data.csv` is missing required columns or has formatting issues, `load_and_prepare_data` in `app.py` might raise errors or return an empty DataFrame.

## 🌐 Deployment

This app can be deployed on Streamlit Community Cloud:

1.  Ensure your GitHub repository is public and contains `app.py`, the `scripts/` folder, `vix_sp500_data.csv`, and `requirements.txt`.
2.  Go to [share.streamlit.io](https://share.streamlit.io/).
3.  Connect your GitHub account, select this repository, set the main file path to `app.py`, and deploy.

## 👤 Who Should Use This Project?

*   **Finance Students & Educators:** For practical application of derivatives and quantitative simulation.
*   **Portfolio and Fund Managers:** To explore VIX-based dynamic hedging concepts.
*   **Retail Investors:** For educational insights into CFD hedging mechanics and trade-offs (**not investment advice**).

## License
SB
