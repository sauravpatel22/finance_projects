import yfinance as yf
import csv
import numpy as np
import matplotlib.pyplot as plot

# Using YF API, import data into a .csv file
def get_data():
    ticker = 'USDINR=X'
    interval = '1d'
    filename ='usd_inr_data.csv'
    data = yf.download(ticker, period='1y', interval=interval)
    data.fillna(method='ffill', inplace = True) # Handle missing data by forward filling
    # Write data to a CSV file
    data.to_csv(filename)

# Read Data
def read_csv():
    filename='usd_inr_data.csv'
    data = []
    with open(filename, mode = 'r') as file:
        dailydata = csv.DictReader(file)
        for row in dailydata:
            # Convert relevant fields to float for calculation purposes
            data.append({
                'Date': row['Date'],
                'Open': float(row['Open']),
                'High': float(row['High']),
                'Low': float(row['Low']),
                'Close': float(row['Close']),
                'Adj Close': float(row['Adj Close']),
                'Volume': int(row['Volume'])
            })
    return data

# Method to calculate SMP(simple moving cross)
def smp(data, window):
    closes = [row['Close'] for row in data]
    return [np.mean(closes[i:i + window]) if i >= window - 1 else 0 for i in range(len(closes))]

# Method that implements a buy/sell signal strategy at crossovers
def generate_signals(data):
    short_ma = smp(data, 20)
    long_ma = smp(data, 50)
    signals = []

    for i in range(len(data)):
        signal = 0 # Initial value
        if i >= 20 and short_ma[i] > long_ma[i]:
            signal = 1  # Buy signal
        elif i >= 20 and short_ma[i] < long_ma[i]:
            signal = -1  # Sell signal
        signals.append(signal)
    return signals

# Backtesting Engine and Risk Management
class BacktestEngine:
    def __init__(self, capital = 100000, risk_limit = 0.1): # Risk is 10% of capacity
        self.capital = capital
        self.initial_capital = capital
        self.position_size_limit = 0.05 * capital # 5% of total capital
        self.risk_limit = risk_limit # Stop if total loss exceeds 10%
        self.pnl = 0 # Track profit and loss
        self.equity_curve = [] # Track account equity over time
        self.daily_returns = [] # Track daily returns
        self.trades = [] # Track the outcomes of trades

    def execute_trade(self, position, price):
        trade_value = self.position_size_limit * price
        if position == 1: # Long position
            self.pnl += trade_value
            self.trades.append(1)  # Mark a win
        elif position == -1: # Short position
            self.pnl -= trade_value
            self.trades.append(-1)  # Mark a loss

        # Risk management: stop if losses exceed 10% of initial capital
        if self.pnl < -self.risk_limit * self.initial_capital:
            print("Max loss reached. Stopping trades.")
            return False
        return True
    
    def run_backtest(self, data, signals):
        prev_equity = self.capital
        for i in range(len(data)):
            position = signals[i]
            price = data[i]['Close']
            if position == 1:  # Buy signal
                if not self.execute_trade(1, price):
                    break
            elif position == -1:  # Sell signal
                if not self.execute_trade(-1, price):
                    break
            self.equity_curve.append(self.capital + self.pnl)
            daily_return = (self.equity_curve[-1] - prev_equity) / prev_equity if prev_equity != 0 else 0
            self.daily_returns.append(daily_return)
            prev_equity = self.equity_curve[-1]

# 5. Performance Evaluation and Plotting
def evaluate_performance(engine):
    total_trades = len(engine.trades)
    final_equity = engine.equity_curve[-1] if engine.equity_curve else engine.capital
    total_profit = final_equity - engine.initial_capital

    # Sharpe Ratio Calculation
    risk_free_rate = 0.03 / 252 # Assuming annual risk-free rate of 3% and 252 trading days
    excess_returns = np.array(engine.daily_returns) - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) != 0 else 0

    # Winning Percentage Calculation
    winning_trades = len([trade for trade in engine.trades if trade == 1])
    winning_percentage = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    print(f"Final Profit/Loss: {total_profit}")
    print(f"Number of Trades: {total_trades}")
    print(f"Final Capital: {final_equity}")
    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Winning Percentage: {winning_percentage}%")

def plot_equity_curve(engine):
    plot.figure(figsize = (10, 5))
    plot.plot(engine.equity_curve, label = 'Equity Curve')
    plot.title('Equity Curve for USD/INR Trades')
    plot.xlabel('Trade Number')
    plot.ylabel('Account Equity')
    plot.legend()
    plot.show()

# Main function to run the entire backtest
def main():
    # Fetch data and save it to a .csv file
    get_data()

    # Read the data 
    data = read_csv()

    # Generate buy/sell signals
    signals = generate_signals(data)

    # Initialize test engine
    engine = BacktestEngine()

    # Using the data, run the engine
    engine.run_backtest(data, signals)

    # Evalueate performance
    evaluate_performance(engine)

    # Show results on a chart
    plot_equity_curve(engine)

if __name__ == "__main__":
    main()
