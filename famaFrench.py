import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels import regression
from statsmodels.formula.api import ols

file_path2 = r'./data_format.xlsx'
file_path3 = r'./Pricedata_format.xlsx'
file_path4 = r'./MarketValuedata.xlsx'
file_path5 = r'./BMdata.xlsx'


# Select stocks to make up the portfolio
def CalculateData(MarktValue_Df, BM_Df, date, stocks):
    # Get stock amount
    amount = len(stocks)

    # Sort Market Cap
    MarktValue_Df = MarktValue_Df.sort_values(by=date, ascending=True, axis=1)
    MarktValue_Df = MarktValue_Df.T

    # Sort BM ratio
    BM_Df = BM_Df.sort_values(by=date, ascending=True, axis=1)
    BM_Df = BM_Df.T

    # Select BIG MarketValue Company
    Big = MarktValue_Df[int(amount - amount / 3):].index
    Big = Big.values

    # Select MEDIUM MarketValue Company
    Mid_MV = MarktValue_Df[int(amount / 3):int(amount - amount / 3)].index
    Mid_MV = Mid_MV.values

    # Select SMALL MarketValue Company
    Small = MarktValue_Df[:int(amount / 3)].index
    Small = Small.values

    # Select high BM portfolio
    High = BM_Df[int(amount - amount / 3):].index
    High = High.values

    # Select mid BM portfolio
    Mid = BM_Df[int(amount / 3):int(amount - amount / 3)].index
    Mid = Mid.values

    # Select low BM portfolio
    Low = BM_Df[:int(amount / 3)].index
    Low = Low.values

    # Select top 5 High BM Big size companies
    HB_portfolio = list(set(High).intersection(set(Big)))
    HB_portfolio.sort()
    HB_portfolio = HB_portfolio[0:5]

    #  Select top 5 High BM Medium size companies
    HM_portfolio = list(set(High).intersection(set(Mid_MV)))
    HM_portfolio.sort()
    HM_portfolio = HM_portfolio[0:5]

    #  Select top 5 High BM Small size companies
    HS_portfolio = list(set(High).intersection(set(Small)))
    HS_portfolio.sort()
    HS_portfolio = HS_portfolio[0:5]

    # Select top 5 Medium BM Big size companies
    MB_portfolio = list(set(Mid).intersection(set(Big)))
    MB_portfolio.sort()
    MB_portfolio = MB_portfolio[0:5]

    # Select top 5 Medium BM Medium size companies
    MM_portfolio = list(set(Mid).intersection(set(Mid_MV)))
    MM_portfolio.sort()
    MM_portfolio = MM_portfolio[0:5]

    # Select top 5 Medium BM Small size companies
    MS_portfolio = list(set(Mid).intersection(set(Small)))
    MS_portfolio.sort()
    MS_portfolio = MS_portfolio[0:5]

    # Select top 5 Low BM Big size companies
    LB_portfolio = list(set(Low).intersection(set(Big)))
    LB_portfolio.sort()
    LB_portfolio = LB_portfolio[0:5]

    # Select top 5 Low BM Medium size companies
    LM_portfolio = list(set(Low).intersection(set(Mid_MV)))
    LM_portfolio.sort()
    LM_portfolio = LM_portfolio[0:5]

    # Select top 5 Low BM Small size companies
    LS_portfolio = list(set(Low).intersection(set(Small)))
    LS_portfolio.sort()
    LS_portfolio = LS_portfolio[0:5]

    # Merge all portfolios into a list
    portfolio = []
    portfolio.insert(0, HB_portfolio)
    portfolio.insert(1, HM_portfolio)
    portfolio.insert(2, HS_portfolio)
    portfolio.insert(3, MB_portfolio)
    portfolio.insert(4, MM_portfolio)
    portfolio.insert(5, MS_portfolio)
    portfolio.insert(6, LB_portfolio)
    portfolio.insert(7, LM_portfolio)
    portfolio.insert(8, LS_portfolio)

    return portfolio

    # If you want to calculate the SMB and HML:
    # ----------------------------------------- #
    # Calculate SMB
    # SMB = Ret_Rate_Df[Small][1:].sum(axis=1) / len(Small) - Ret_Rate_Df[Big][1:].sum(axis=1) / len(Big)
    # # Calculate HML
    # HML = Ret_Rate_Df[High][1:].sum(axis=1)/len(High) - Ret_Rate_Df[Low][1:].sum(axis=1)/len(Low)
    # # print(HML)
    # # Index Return Rate --- use FTSE100 index
    # RM = np.diff(np.log(FTSE100['Close']))-0.0068/12

    # # print(len(SMB))
    # # print(len(HML))
    # # print(date)
    # # Create a data frame to store data
    # X = pd.DataFrame({"RM": RM, "SMB": SMB, "HML": HML})

    # # Set factors
    # factors = ["RM", "SMB", "HML"]
    # X = X[factors]

    # Do linear regression, calculate ai
    # t_scores = [0.0] * amount

    # # Iterate to calculate the alpha for each stock
    # for i in range(0, amount):
    #     t_stock = stocks[i]
    #     t_r = Linreg(X, Ret_Rate_Df[t_stock][1:] - 0.0068/12, len(factors))
    #     t_scores[i] = t_r[0]

    # scores = pd.DataFrame({'code': stocks, 'score': t_scores})

    # scores = scores.sort_values(by='score')

    # return scores


# Calculate the value of the portolios
def Port_Cap(portfolio, Ret_Rate_Df, Price_Df, money):
    # Get the stock tickets
    stocks_name = portfolio[0:5]

    # Get the top 5 stocks' monthly return
    stocks_rate = Ret_Rate_Df[stocks_name]
    stocks_rate_list = stocks_rate[0:1].values
    stocks_rate_list = stocks_rate_list[0]

    # Get the top 5 stocks' stock price, convert it to list for further calculation
    stocks_price = Price_Df[stocks_name]
    stocks_price_list = stocks_price[0:1].values
    stocks_price_list = stocks_price_list[0]

    # Assign the same amount to each stock
    data = [0.2, 0.2, 0.2, 0.2, 0.2]
    allocation_money = [0, 0, 0, 0, 0]
    for i in range(0, len(data)):
        allocation_money[i] = money * data[i]

    stocks_money = []
    for i in range(0, len(stocks_price_list)):
        stocks_money.insert(
            i, allocation_money[i] - (allocation_money[i] % stocks_price_list[i]))

    overage_money = np.sum(allocation_money) - np.sum(stocks_money)
    overage_money = round(overage_money, 2)

    # Calculate investment return
    stocks_final_money = []
    for i in range(0, len(stocks_price_list)):
        stocks_final_money.insert(i, (1 - 0.001 + stocks_rate_list[i]) * stocks_money[i])

    final_money = overage_money + round(np.sum(stocks_final_money), 2)

    return final_money


# linear regression model
def Linreg(X, Y, columns=3):
    X = sm.add_constant(np.array(X))
    Y = np.array(Y)
    if len(Y) > 1:
        results = regression.linear_model.OLS(Y, X).fit()
        return results.params
    else:
        return [float("NaN")] * (columns + 1)


# Main function
def main():
    # Read data from Excel
    Price_df = pd.read_excel(file_path1, sheet_name='price_format')
    MktCap_df = pd.read_excel(file_path2, sheet_name='market_value_format')
    BM_df = pd.read_excel(file_path3, sheet_name='mb_format')
    FTSE100_df = pd.read_excel(file_path4, sheet_name='ftse100')

    # Set index
    Price_df.set_index(["Dates"], inplace=True)
    MktCap_df.set_index(["Dates"], inplace=True)
    BM_df.set_index(["Dates"], inplace=True)
    FTSE100_df.set_index(["Dates"], inplace=True)

    # Filter data 2014.6-2019.6
    Price_df = Price_df[152:-3]
    MktCap_df = MktCap_df[152:-3]
    BM_df = BM_df[152:-3]

    # Date list
    date_list = BM_df.index

    # Stock names
    stock_names = BM_df.columns.values.tolist()

    # Calculate monthly Return Rate
    Ret_Rate_Df = pd.DataFrame(index=date_list, columns=stock_names)

    for row in range(0, len(Price_df)):
        if (row == 0):
            for stock_name in stock_names:
                Ret_Rate_Df.iloc[row][stock_name] = 0
        else:
            for stock_name in stock_names:
                Ret_Rate_Df.iloc[row][stock_name] = (Price_df.iloc[row][stock_name] / Price_df.iloc[row - 1][
                    stock_name]) - 1

    # Holding time period
    AllDate = len(date_list)

    # Create different data frames in order to store different portfolios' investment return
    HB_frames_port = []
    HM_frames_port = []
    HS_frames_port = []
    MB_frames_port = []
    MM_frames_port = []
    MS_frames_port = []
    LB_frames_port = []
    LM_frames_port = []
    LS_frames_port = []

    for i in range(0, AllDate - 1):
        if (i == 0):
            HB_port_val = 100000
            HM_port_val = 100000
            HS_port_val = 100000
            MB_port_val = 100000
            MM_port_val = 100000
            MS_port_val = 100000
            LB_port_val = 100000
            LM_port_val = 100000
            LS_port_val = 100000
            HB_frames_port.append(HB_port_val)
            HM_frames_port.append(HM_port_val)
            HS_frames_port.append(HS_port_val)
            MB_frames_port.append(MB_port_val)
            MM_frames_port.append(MM_port_val)
            MS_frames_port.append(MS_port_val)
            LB_frames_port.append(LB_port_val)
            LM_frames_port.append(LM_port_val)
            LS_frames_port.append(LS_port_val)
        else:
            HB_money = HB_port_val
            HM_money = HM_port_val
            HS_money = HS_port_val
            MB_money = MB_port_val
            MM_money = MM_port_val
            MS_money = MS_port_val
            LB_money = LB_port_val
            LM_money = LM_port_val
            LS_money = LS_port_val

            all_portfolio = CalculateData(MktCap_df, BM_df, date_list[i], stock_names)

            HB_portfolio = all_portfolio[0]
            HM_portfolio = all_portfolio[1]
            HS_portfolio = all_portfolio[2]
            MB_portfolio = all_portfolio[3]
            MM_portfolio = all_portfolio[4]
            MS_portfolio = all_portfolio[5]
            LB_portfolio = all_portfolio[6]
            LM_portfolio = all_portfolio[7]
            LS_portfolio = all_portfolio[8]

            # Calculate monthly return respectively
            # High Big
            HB_port_val = Port_Cap(
                HB_portfolio, Ret_Rate_Df[i:i + 1], Price_df[i:i + 1], HB_money)
            HB_frames_port.append(HB_port_val)

            # High Mid
            HM_port_val = Port_Cap(
                HM_portfolio, Ret_Rate_Df[i:i + 1], Price_df[i:i + 1], HM_money)
            HM_frames_port.append(HM_port_val)
            # High Small
            HS_port_val = Port_Cap(
                HS_portfolio, Ret_Rate_Df[i:i + 1], Price_df[i:i + 1], HS_money)
            HS_frames_port.append(HS_port_val)

            # Mid Big
            MB_port_val = Port_Cap(
                MB_portfolio, Ret_Rate_Df[i:i + 1], Price_df[i:i + 1], MB_money)
            MB_frames_port.append(MB_port_val)
            # Mid Mid
            MM_port_val = Port_Cap(
                MM_portfolio, Ret_Rate_Df[i:i + 1], Price_df[i:i + 1], MM_money)
            MM_frames_port.append(MM_port_val)
            # Mid Small
            MS_port_val = Port_Cap(
                MS_portfolio, Ret_Rate_Df[i:i + 1], Price_df[i:i + 1], MS_money)
            MS_frames_port.append(MS_port_val)

            # Low Big
            LB_port_val = Port_Cap(
                LB_portfolio, Ret_Rate_Df[i:i + 1], Price_df[i:i + 1], LB_money)
            LB_frames_port.append(LB_port_val)
            # Low Mid
            LM_port_val = Port_Cap(
                LM_portfolio, Ret_Rate_Df[i:i + 1], Price_df[i:i + 1], LM_money)
            LM_frames_port.append(LM_port_val)
            # Low Small
            LS_port_val = Port_Cap(
                LS_portfolio, Ret_Rate_Df[i:i + 1], Price_df[i:i + 1], LS_money)
            LS_frames_port.append(LS_port_val)

    # FTSE100 index monthly return
    FTSE100 = FTSE100_df[2:]['Close'].values / FTSE100_df[0:1]['Close'].values
    FTSE100 = np.insert(FTSE100, 0, 1)
    FTSE100 = FTSE100 * 100000
    print(len(FTSE100))
    print(len(HB_frames_port))

    # Merge the portfolios' return and FTSE100 index return into one data frame
    result_port = {'FTSE100': FTSE100, 'HB_Portfolio': HB_frames_port, 'HM_Portfolio': HM_frames_port,
                   'HS_Portfolio': HS_frames_port,
                   'MB_Portfolio': MB_frames_port, 'MM_Portfolio': MM_frames_port, 'MS_Portfolio': MS_frames_port,
                   'LB_Portfolio': LB_frames_port, 'LM_Portfolio': LM_frames_port, 'LS_Portfolio': LS_frames_port}

    result_port_df = pd.DataFrame(data=result_port, index=date_list[:-1])
    print(result_port_df)

    # Calculate monthly return for each portfolio
    result_monthly_return = np.diff(np.log(result_port_df), axis=0) + 0 * result_port_df[1:]

    # print(result_monthly_return)
    # draw scattor plot
    result_monthly_return.plot(kind='scatter', x='FTSE100', y='LS_Portfolio')
    # Linear regression, fit by scatter plot
    beta_PVAL, alpha_PVAL = np.polyfit(
        result_monthly_return['FTSE100'], result_monthly_return['LS_Portfolio'], 1)

    # Calculate our strategy return
    # Our strategy is to invest low BM and small size companies

    P_end = result_port_df['LS_Portfolio'][-1:].values
    P_start = result_port_df['LS_Portfolio'][0:1].values
    TR = (P_end - P_start) / P_start * 100
    print('Strategy Return ', TR[0])

    # Calculate FTSE100 index reurn
    M_end = result_port_df['FTSE100'][-1:].values
    M_start = result_port_df['FTSE100'][0:1].values
    BR = (M_end - M_start) / M_start * 100
    print('Index Return ', BR[0])

    # Calculating Strategy Annualized Returns
    P = sum(result_montly_return['LS_Portfolio'].values)
    TAR = ((1 + P) ** (12 / 60) - 1) * 100
    print('Strategy Annualized Returns', TAR)

    #  Calculating Index Annualized Returns
    M = sum(result_montly_return['FTSE100'].values)
    BAR = ((1 + M) ** (12 / 60) - 1) * 100
    print('Index Annualized Returns', BAR)


    # Beta
    print('a', alpha_PVAL)
    print('beta_PVAL:', beta_PVAL)
    # Alpha
    alpha = (TAR - (4 + beta_PVAL * (BAR - 4)))
    print('alpha_PVAL:', alpha)
    # Sharpe ratio
    mean = np.mean(result_monthly_return['HB_Portfolio'].values)
    std = np.std(result_montly_return['HB_Portfolio'].values)
    SR = np.sqrt(12) * ((mean - 0.0068 / 12) / std)
    print('shape_PVAL:', SR)

    # Draw a fitted straight line
    plt.plot(result_monthly_return_return['FTSE100'], beta_PVAL *
             result_monthly_return_return['FTSE100'] + alpha_PVAL, '-', color='r')

    # Print correlation
    print(result_monthly_return_return.corr(method='pearson'))
    print(result_port_df[-1:])
    print((result_port_df[-1:].values - result_port_df[0:1].values) / result_port_df[0:1].values)

    # Plot returns
    result_port_df.plot()
    plt.show()


if __name__ == '__main__':
    main()
