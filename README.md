# CFD Hedging strategy simulation app

This Streamlit application simulates and compares a classic S&P 500 portfolio (Model A) against a portfolio that uses VIX-triggered short S&P 500 CFDs for hedging (Model B). Users can adjust various configuration parameters to see their impact on portfolio performance.

[_details_](https://github.com/shekharbiswas/CFD_Simulation)
## Features

*   Interactive sidebar to configure simulation parameters:
    *   Initial Capital
    *   Equity Allocation
    *   VIX Threshold for hedging
    *   Hedge Ratio
    *   CFD Cost parameters (financing fee, borrowing cost, spread)
*   Visual comparison of Model A (Classic) vs. Model B (CFD-Hedged) portfolio values over time using an interactive Plotly chart.
*   Summary metrics for both portfolios.
*   Highlighting of specific analysis periods (e.g., COVID crisis) on the plot.

## Files

*   `app.py`: The main Streamlit application script.
*   `data/vix_sp500_data.csv`: The historical data file for S&P 500, VIX, and SOFR.
*   `requirements.txt`: Python libraries required to run the app.
*   `.streamlit/config.toml` (Optional): For custom Streamlit theming (not strictly required to run if defaults are okay).

## How to Run Locally

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/shekharbiswas/CFD_app.git
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
    Ensure you have Python 3.8+ installed. Then, install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit App:**
    From the root directory of the cloned repository (`CFD_app/`), execute:
    ```bash
    streamlit run app.py
    ```

5.  **View in Browser:**
    Streamlit will typically open the app automatically in your default web browser. If not, it will provide a local URL (usually `http://localhost:8501`) that you can open.

## Usage

*   Once the app is running, use the sidebar on the left to adjust the configuration parameters.
*   Click the "🚀 Run Simulation & Plot" button to generate and display the portfolio comparison chart and summary metrics based on your selected parameters.
*   Interact with the Plotly chart (zoom, pan, hover for details).

## Data

The application uses historical data from `data/vix_sp500_data.csv`. This file should contain columns for 'date', 'S&P500', 'VIX', and 'SOFR'.

## Deployment

This app is structured to be easily deployable on Streamlit Community Cloud:

1.  Ensure your GitHub repository is public.
2.  Go to [share.streamlit.io](https://share.streamlit.io/).
3.  Connect your GitHub account.
4.  Click "New app" and select this repository.
5.  The main file path should be `app.py`.
6.  Deploy.

## Contributing

Feel free to fork the repository, make improvements, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
SB
