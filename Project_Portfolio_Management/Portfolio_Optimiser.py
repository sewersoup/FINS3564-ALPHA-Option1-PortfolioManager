"""
FINS3645 Option #1 Asset Management Project
Code by: Alexander Devyn Maseimilian (z5376366)
Project Name: ALPHA (Automated Learning Portfolio Handling Advisor)

ALPHA aims to design an automated, data-driven portfolio management system for an asset management firm,
utilizing a wide range of inputs including market and sentiment indicators, news, and other diverse data streams.
This involves setting up a data ingestion pipeline to structure incoming data, conducting exploratory data analysis
and feature extraction, developing predictive machine learning models for market trends and asset returns,
and optimizing portfolio allocation based on these predictions. The performance of the models and the resulting
portfolio allocations will be back-tested using historical data for further fine-tuning and improvement.
"""

# Import external libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplcyberpunk
import seaborn as sns
import scipy.optimize as sco
import scipy.interpolate as sci

# Styling for matplotlib
plt.style.use("cyberpunk")
plt.style.use("dark_background")

# Number of Assets
noa = 10

# Station 1: ETL
def station_1():
    # Import stocks, client details, and economic indicators data from csv files
    stocks = pd.read_excel("ASX200top10.xlsx", sheet_name="Bloomberg raw", header=[0, 1], index_col=0)
    clients = pd.read_excel("Client_Details.xlsx", sheet_name="Data", index_col=0)
    indicators = pd.read_excel("Economic_Indicators.xlsx")
    news = pd.read_json("news_dump.json")

    # Removes all other columns except the PX_LAST column
    stocks = stocks.iloc[:, stocks.columns.get_level_values(1) == 'PX_LAST']
    stocks.columns = stocks.columns.droplevel(1)

    # Plots prices of each stock overtime
    for stock in stocks.columns:
        plt.figure(figsize=(18, 10))
        stocks[stock].plot(color="#DAFF00")
        plt.title(stock + " Price")
        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.grid(False)
        mplcyberpunk.add_glow_effects()
        # save the figure before showing it
        # plt.savefig(f"{stock}_price.png", format='png', dpi=300)
        plt.show()

    # Removes AS51 Index from the rest of the data
    asx = stocks['AS51 Index']
    stocks = stocks.drop('AS51 Index', axis=1)

    # Print the adjusted dataframe
    print(stocks)

    return stocks


# Station 2: Feature Engineering
def station_2(stocks):
    # Calculate returns and covariance matrix
    rets = np.log(stocks / stocks.shift(1))
    rets.hist(bins=50, figsize=(18, 10), color="#DAFF00", grid=False)
    plt.tight_layout()
    mplcyberpunk.add_glow_effects()
    # plt.savefig("returns_hist.png")
    plt.show()

    # Plot returns
    for col in rets.columns:
        plt.figure(figsize=(18, 10))
        plt.plot(rets[col], color="#DAFF00")
        plt.title(col)
        plt.grid(False)
        plt.tight_layout()
        mplcyberpunk.make_lines_glow()
        # If you want to save the plot for each series
        # plt.savefig(f"returns_plot_{col}.png")
        plt.show()

    # Collect basic statistics relevant for Station #3 Model Design
    print(rets)
    print(rets.describe())
    print(rets.mean() * 252)
    print(rets.cov() * 252)

    # Plots the covariance heatmap
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.heatmap(rets.cov() * 252, cmap="YlGnBu", annot=True)
    fig.tight_layout()
    # plt.savefig("heatmap.png")
    plt.show()

    return rets


# Station 3 + 4 (Implementation of Station 3): Model Design and Implementation
def station_3_4(rets):
    clients = pd.read_excel("Client_Details.xlsx", sheet_name="Data", index_col=0)

    # Clarity and parsimony of the resulting data is critical in this stage
    def statistics(weights):
        weights = np.array(weights)
        pret = np.sum(rets.mean() * weights) * 252
        pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
        return np.array([pret, pvol, pret / pvol])

    def min_func_sharpe(weights):
        return -statistics(weights)[2]

    # Function to plot returns based on optimal portfolio weightings
    def plot_portfolio_returns(sequence, initial_investment, type):
        # Get weights from opts
        weights = np.array(sequence['x'].round(3))

        # Calculate weighted returns
        weighted_rets = (rets * weights).sum(axis=1)

        # Calculate cumulative returns
        cumulative_rets = np.exp(weighted_rets.cumsum()) - 1

        # Calculate portfolio value
        portfolio_value = (1 + cumulative_rets) * initial_investment

        # Plot portfolio value
        plt.figure(figsize=(18, 10))
        plt.plot(portfolio_value, color="#DAFF00")  # Set the line color here
        mplcyberpunk.add_glow_effects()
        plt.title("Portfolio Returns")
        plt.ylabel("Portfolio Value")
        plt.xlabel("Date")
        plt.grid(False)  # Set grid to False
        plt.tight_layout()
        # Save the figure to a PNG file before displaying it
        # plt.savefig(f"{type}_returns.png", format='png', dpi=300)
        plt.show()

    # Plot pie chart function
    def plot_pie_chart(weights, colors, title):
        # Exclude stocks with 0% weight
        non_zero_weights = [weight for weight in weights if weight != 0]
        non_zero_labels = [label for weight, label in zip(weights, stock_names) if weight != 0]

        # Ensure the number of colors matches the number of non-zero labels
        colors = colors[:len(non_zero_labels)]

        # Plot the pie chart
        plt.figure(figsize=(18, 10))
        wedges, texts, autotexts = plt.pie(non_zero_weights, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Change the font color for the percentages to black
        for autotext in autotexts:
            autotext.set_color('black')

        plt.legend(wedges, non_zero_labels, title="Stocks", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.title(title)
        plt.tight_layout()
        # plt.savefig('Pie_Sharpe.png')
        plt.show()

    # Plot consumer sentiment index
    def plot_consumer_sentiment_index():
        indicators = pd.read_excel("Economic_Indicators.xlsx")

        # Include the rows with 'Monthly Indicators' and 'Consumer Sentiment Index'
        selected_rows = indicators[
            indicators['Key Indicators - Australia'].isin(['Monthly Indicators', 'Consumer Sentiment Index'])]

        # Creating a new DataFrame without headers
        new_df = pd.DataFrame(selected_rows.values)

        print(new_df)

        # Getting dates from the first row, excluding any NaN or missing values
        dates = [str(date) for date in new_df.iloc[0, 1:].values if pd.notnull(date)]

        # Getting Consumer Sentiment Index values from the second row, excluding corresponding NaN or missing values
        consumer_sentiment_index_values = [value for value in new_df.iloc[1, 1:].values if pd.notnull(value)]

        # Make sure lengths match, in case there are more dates than values
        dates = dates[:len(consumer_sentiment_index_values)]

        # Convert the consumer sentiment values to numeric, handling any non-numeric strings
        consumer_sentiment_index_values = pd.to_numeric(consumer_sentiment_index_values, errors='coerce')

        # Getting dates from the first row, excluding any NaN or missing values
        dates = [str(date) for date in new_df.iloc[0, 1:].values if pd.notnull(date)]

        # Getting Consumer Sentiment Index values from the second row, excluding corresponding NaN or missing values
        consumer_sentiment_index_values = [value for value in new_df.iloc[1, 1:].values if pd.notnull(value)]

        # Make sure lengths match, in case there are more dates than values
        dates = dates[:len(consumer_sentiment_index_values)]

        # Convert the consumer sentiment values to numeric, handling any non-numeric strings
        consumer_sentiment_index_values = pd.to_numeric(consumer_sentiment_index_values, errors='coerce')

        # Reverse the order of dates and values
        dates = dates[::-1]
        consumer_sentiment_index_values = consumer_sentiment_index_values[::-1]

        # Plotting
        plt.figure(figsize=(18, 10))
        plt.plot(dates, consumer_sentiment_index_values, color="#DAFF00")
        mplcyberpunk.add_glow_effects()
        plt.xlabel('Date')
        plt.ylabel('Consumer Sentiment Index')
        plt.title('Consumer Sentiment Index Over Time')
        plt.xticks(rotation=45)
        plt.grid(False)  # Set grid to False
        plt.tight_layout()
        # plt.savefig('CSI_chart.png')
        plt.show()

    # Define your custom HEX colors
    colors = ['#DAFF00', '#00DAFF', '#FF00DA', '#FF005B']

    #  Stock names derived from rets
    stock_names = rets.columns.tolist()

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bnds = tuple((0, 1) for x in range(noa))
    noa * [1. / noa, ]

    opts = sco.minimize(min_func_sharpe, noa * [1. / noa, ], method='SLSQP',
                        bounds=bnds, constraints=cons)

    # Save the output in a pandas DataFrame
    df_max_sharpe = pd.DataFrame([opts['x'].round(3), statistics(opts['x']).round(3)],
                                 columns=stock_names,
                                 index=['Weights', 'Statistics'])

    print("***Maximization of Sharpe Ratio***")
    # Print DataFrame
    print(df_max_sharpe)
    # Save the dataframe to a csv file
    df_max_sharpe.to_csv('Max_Sharpe_Ratio.csv')

    # Get the weights from the DataFrame
    sharpe_weights = df_max_sharpe.loc['Weights']

    # Plot optimal tangent weights pie chart
    plot_pie_chart(sharpe_weights, colors, 'Optimal Portfolio Weights (Max Sharpe Ratio)')

    def min_func_variance(weights):
        return statistics(weights)[1] ** 2

    optv = sco.minimize(min_func_variance, noa * [1. / noa, ], method='SLSQP',
                        bounds=bnds, constraints=cons)

    # Save the output in a pandas DataFrame
    df_min_variance = pd.DataFrame([optv['x'].round(3), statistics(optv['x']).round(3)],
                                   columns=stock_names,
                                   index=['Weights', 'Statistics'])

    print("****Minimizing Variance***")
    # print(optv)
    print(df_min_variance)
    # Save the dataframe to a csv file
    df_min_variance.to_csv('Min_Variance.csv')

    # plot sharpe and variance portfolio returns with initial investment of $1000
    plot_portfolio_returns(opts, 1000, "max_sharpe")
    plot_portfolio_returns(optv, 1000, "min_variance")

    # Returns the portfolio standard deviation
    def min_func_port(weights):
        return statistics(weights)[1]

    # Setup random portfolio weights
    weights = np.random.random(noa)
    weights /= np.sum(weights)

    # Derive Portfolio Returns & simulate various 2500x combinations
    print(np.sum(rets.mean() * weights) * 252)
    print(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

    prets = []
    pvols = []
    for p in range(2500):
        weights = np.random.random(noa)
        weights /= np.sum(weights)
        prets.append(np.sum(rets.mean() * weights) * 252)
        pvols.append(np.sqrt(np.dot(weights.T,
                                    np.dot(rets.cov() * 252, weights))))
    prets = np.array(prets)
    pvols = np.array(pvols)

    def min_func_port(weights):
        return statistics(weights)[1]

    bnds = tuple((0, 1) for x in weights)
    trets = np.linspace(0.0, 0.25, 50)
    tvols = []
    for tret in trets:
        cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - tret},
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        res = sco.minimize(min_func_port, noa * [1. / noa, ], method='SLSQP',
                           bounds=bnds, constraints=cons)
        tvols.append(res['fun'])
    tvols = np.array(tvols)

    plt.figure(figsize=(18, 10))
    plt.scatter(pvols, prets,
                c=prets / pvols, marker='o')
    # random portfolio composition
    plt.scatter(tvols, trets,
                c=trets / tvols, marker='x')
    # efficient frontier
    plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0],
             '^', markersize=15.0)
    # portfolio with highest Sharpe ratio
    plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0],
             '^', markersize=15.0)
    # minimum variance portfolio
    plt.grid(False)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharpe ratio')
    mplcyberpunk.make_scatter_glow()
    plt.tight_layout()
    # plt.savefig("efficient_frontier1.png")
    plt.show()

    ind = np.argmin(tvols)
    evols = tvols[ind:]
    erets = trets[ind:]
    tck = sci.splrep(evols, erets)

    def f(x):
        ''' Efficient frontier function (splines approximation). '''
        return sci.splev(x, tck, der=0)

    def df(x):
        ''' First derivative of efficient frontier function. '''
        return sci.splev(x, tck, der=1)

    def equations(p, rf=0.01):
        eq1 = rf - p[0]
        eq2 = rf + p[1] * p[2] - f(p[2])
        eq3 = p[1] - df(p[2])
        return eq1, eq2, eq3

    opt = sco.fsolve(equations, [0.01, 0.5, 0.15])
    # print(opt)
    # print(np.round(equations(opt), 6))

    plt.figure(figsize=(18, 10))
    plt.scatter(pvols, prets,
                c=(prets - 0.01) / pvols, marker='o')
    # random portfolio composition
    plt.plot(evols, erets, 'g', lw=4.0)
    # efficient frontier
    cx = np.linspace(0.0, 0.3)
    plt.plot(cx, opt[0] + opt[1] * cx, lw=1.5)
    # capital market line
    plt.plot(opt[2], f(opt[2]), '^', markersize=15.0)
    plt.grid(False)
    plt.axhline(0, color='k', ls='--', lw=2.0)
    plt.axvline(0, color='k', ls='--', lw=2.0)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharpe ratio')
    plt.ylim(-0.05, 0.27)
    plt.xlim(0.135, 0.2)
    mplcyberpunk.make_scatter_glow()
    plt.tight_layout()
    # plt.savefig("efficient_frontier2.png")
    plt.show()

    cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - f(opt[2])},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    res = sco.minimize(min_func_port, noa * [1. / noa, ], method='SLSQP',
                       bounds=bnds, constraints=cons)

    # Save the output in a pandas DataFrame, using the stock names as columns
    df_optimal_tangent = pd.DataFrame([res['x'].round(3), statistics(res['x']).round(3)], columns=stock_names,
                                      index=['Weights', 'Statistics'])

    print("***Optimal Tangent Portfolio***")
    # Print DataFrame
    print(df_optimal_tangent)
    # Save the dataframe to a csv file
    df_optimal_tangent.to_csv('Optimal_Tangent.csv')

    # Get the weights from the DataFrame
    tangent_weights = df_optimal_tangent.loc['Weights']

    # Plot optimal tangent weights pie chart
    plot_pie_chart(tangent_weights, colors, 'Optimal Portfolio Weights (Optimal Tangent)')

    # Plot optimal tangent portfolio returns
    plot_portfolio_returns(res, 1000, "optimal_tangent")

    # Mean and standard deviation of the optimal tangent portfolio
    ret_rp = opt[2]
    vol_rp = float(f(opt[2]))

    # Calculates the risky share given a risk appetite
    def y_star(ret_rp, vol_rp, A, rf=0.01):
        return (ret_rp - rf) / (A * (vol_rp ** 2))

    # Create an empty list to hold the rows
    rows = []

    # Loop through the risk profiles from 10 down to 1
    for i in range(10, 0, -1):
        # Convert risk profile to risk aversion parameter (you can adjust the mapping)
        A = 11 - i

        y = y_star(ret_rp, vol_rp, A)
        risk_free_weight = max(1 - y, 0)

        # Proportional risky weights
        risky_weights = res['x'] * max(y, 0)

        # Scale the risky weights so they sum to 1 - risk_free_weight
        risky_weights_scaled = risky_weights * (1 - risk_free_weight) / sum(risky_weights)

        # Create a dictionary to hold the weightings for this risk profile
        weightings_dict = {'Risk Profile': i}
        weightings_dict.update(dict(zip(stock_names, risky_weights_scaled.round(3))))

        # Include the risk-free asset
        weightings_dict['Risk-free asset'] = risk_free_weight

        # Append the dictionary to the rows list
        rows.append(weightings_dict)

    # Create a DataFrame from the rows list
    risk_profile_weightings = pd.DataFrame(rows, columns=['Risk Profile'] + stock_names + ['Risk-free asset'])

    # Write the DataFrame to a CSV file
    risk_profile_weightings.to_csv('Risk_Profile_Weightings.csv', index=False)

    # Print the DataFrame
    print(risk_profile_weightings)

    # Print CSI plot
    plot_consumer_sentiment_index()

def main():
    # Station 1
    df1 = station_1()

    # Station 2
    df2 = station_2(df1)

    # Station 3 + 4
    station_3_4(df2)


if __name__ == '__main__':
    main()