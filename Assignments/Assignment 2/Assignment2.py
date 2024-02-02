#%%
import wrds
import pandas as pd
# %%
user_name = 'aghajanzadeh93'
password = '4061@Morteza'
wrds_db = wrds.Connection(wrds_username=user_name, wrds_password=password)
#%%  List all libraries
wrds_db.list_libraries()
# %% 
monthly_price_data = wrds_db.raw_sql(
    """select cusip, permno, date, prc, ret, shrout from crsp.msf""", 
                     date_cols=['date'])

# %%
# Closing Price or Bid/Ask Average
# Variable Name = PRC
monthly_price_data.describe()
# %%
# Prc is the closing price or the negative bid/ask average for a trading day. If the closing price is not available on any given trading day, the number in the price field has a negative sign to indicate that it is a bid/ask average and not an actual closing price. Please note that in this field the negative sign is a symbol and that the value of the bid/ask average is not negative.

# If neither closing price nor bid/ask average is available on a date, prc is set to zero. In a monthly database, prc is the price on the last trading date of the month. The price series begins the first month-end after the security begins trading and ends the last complete month of trading.

# If the security of a company is included in the Composite Pricing network, the closing price while listed on NYSE or AMEX on a trading date is the last trading price for that day on the exchange that the security last traded.

# Similarly, highs, lows, and volumes include trades on all exchanges on which that security traded. For example, if a stock trades on both the NYSE and the PACX (Pacific Stock Exchange), and the last trade occurs on the PACX, the closing price on that day represents the closing price on the PACX, not the NYSE. Price data for Nasdaq securities comes directly from the NASD with the close of the day at 4:00 p.m. Eastern Time. Automated trades after hours on Nasdaq are counted on the next trading date, although the volumes are applied to the current date. Daily trading prices for The Nasdaq National Market securities were first reported November 1, 1982. Daily trading prices for The Nasdaq Small Cap Market were first reported June 15, 1992. prc for Nasdaq securities is always a negative bid/ask average before this time. All prices are raw prices as they were reported at the time of trading.
#%%
wrds_db.raw_sql(
    """
    select cusip, permno, date, prc, ret, shrout
    from crsp.msf
    where ticker = 'AAPL' 
      """, 
                     date_cols=['date'])
#%%
wrds_db.raw_sql(
    """select CUSIP
    from comp.funda
    where tic = 'AAPL' 
      """)