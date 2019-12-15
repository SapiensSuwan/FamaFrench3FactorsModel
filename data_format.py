import xlrd
import pandas as pd

# Read data from excel
# Create new excel sheets to store different data respectively

file_path = r'./tickets_data.xlsx'
file_path2 = r'./data_format.xlsx'
file_path3 = r'./Pricedata_format.xlsx'
file_path4 = r'./MarketValuedata.xlsx'
file_path5 = r'./BMdata.xlsx'


# Read data from Excel
data = xlrd.open_workbook(file_path)
data2 = xlrd.open_workbook(file_path2)

# Read data from each sheet in Excel
table = data.sheet_by_name('Current Mkt Cap')
table2 = data2.sheet_by_name('market_value_format')

# Get the tickets code from 'Current Mkt Cap'
data1_rowvalue = table.row_values(0)

data2_rowvalue = table2.row_values(0)


# Intersect the ftse100 stock tickets with all tickets, it will filter out the ftse100 stocks
x = list(set(data1_rowvalue).intersection(set(data2_rowvalue)))

# Absolute complement
y = list(set(data1_rowvalue)-set(x))

# Convert Excel data to pandas
# Price_df = pd.read_excel(file_path, sheet_name='Price')
# MktCap_df = pd.read_excel(file_path, sheet_name='Current Mkt Cap')
MB_df = pd.read_excel(file_path, sheet_name='Mkt to Book')

# Drop the unnecessary data
# MktCap_df = MktCap_df.drop(columns=y)
# Price_df = Price_df.drop(columns=y)
MB_df = MB_df.drop(columns=y)


# Set "Dates" as index
# Price_df.set_index(["Dates"], inplace=True)

MB_df.set_index(["Dates"], inplace=True)

# MktCap_df.set_index(["Dates"], inplace=True)

stock_names = MB_df.columns.values.tolist()

# Convert data from M/B ratio to B/M ratio

for index, row in MB_df.iterrows():
    for stock_name in stock_names:
        row[stock_name] = 1/row[stock_name]

# Write data into excel
Price_df.to_excel(file_path3, sheet_name="price_format")
MB_df.to_excel(file_path5, sheet_name="mb_format")
MktCap_df.to_excel(file_path4, sheet_name="market_value_format")
format = lambda q: '%.2f' % q

# Price_df.applymap(format)

# print(Price_df)
