#%%
import wrds
import pandas as pd
import numpy as np
# %%
user_name = 'aghajanzadeh93'
password = '4061@Morteza'
wrds_db = wrds.Connection(wrds_username=user_name, wrds_password=password)
#%%  List all libraries
# wrds_db.list_libraries()
# %% Get CRSP Data
monthly_price_data = wrds_db.raw_sql(
    """
    select permno, date, prc, ret, shrout 
    from crsp.msf
    """, 
                     date_cols=['date'])
columns = ['cusip',
 'permno',
 'permco',
 'issuno',
 'hexcd',
 'hsiccd',
 'date',
 'bidlo',
 'askhi',
 'prc',
 'vol',
 'ret',
 'bid',
 'ask',
 'shrout',
 'cfacpr',
 'cfacshr',
 'altprc',
 'spread',
 'altprcdt',
 'retx']

# %%
# Closing Price or Bid/Ask Average
# Variable Name = PRC
monthly_price_data.describe()
# %%

#%% Clean Monthly Price Data
monthly_price_data['year'] = monthly_price_data['date'].dt.year
monthly_price_data['month'] = monthly_price_data['date'].dt.month
monthly_price_data.loc[monthly_price_data.month < 6, 'year'] = monthly_price_data.loc[monthly_price_data.month < 6]['year'] - 1
monthly_price_data['year'] = monthly_price_data['year'].astype(int)
monthly_price_data.drop(columns=['month'], inplace=True)
monthly_price_data.sort_values(by=['permno', 'year'], inplace=True)
monthly_price_data['prc_flag'] = monthly_price_data['prc'].apply(lambda x: 1 if x <= 0 else 0)
monthly_price_data['prc'] = monthly_price_data['prc'].abs()
monthly_price_data['mcap'] = monthly_price_data['prc'] * monthly_price_data['shrout'] / 1e6
monthly_price_data.reset_index(drop=True, inplace=True)
monthly_price_data.head()
#%%
monthly_price_data.describe()
#%%
book_wrds = wrds_db.raw_sql(
    """select GVKEY, BKVLPS, FYEAR
    from comp.funda
      """).sort_values(by='fyear')
book_wrds.dropna(inplace=True)
book_wrds.reset_index(drop=True, inplace=True)
book_wrds['year'] = book_wrds['fyear'].astype(int)
book_wrds.drop(columns=['fyear'], inplace=True)
linking_table = wrds_db.raw_sql(
    """select gvkey, lpermno as permno
    from crsp_a_ccm.ccmxpf_lnkhist
      """)
linking_table.dropna(inplace=True)
linking_table.reset_index(drop=True, inplace=True)
book_wrds = book_wrds.merge(linking_table, on='gvkey', how='left')
book_wrds.isnull().sum()
#%% Moody's Data from French's Website
data = pd.read_csv('Data/DFF_BE_With_Nonindust.csv')
data.replace(-99.9900, np.nan, inplace=True)
data.rename(
    columns={
        'Column2': 'CRSP_Permno',
        'Column3': 'First_Moody_Year',
        'Column4': 'Last_Moody_Year',
        }, inplace=True)

data = data.melt(id_vars=['CRSP_Permno', 'First_Moody_Year', 'Last_Moody_Year']).sort_values(by=['CRSP_Permno', 'First_Moody_Year', 'Last_Moody_Year']).dropna().drop('variable', axis=1)

data['year'] = data.groupby(['CRSP_Permno', 'First_Moody_Year', 'Last_Moody_Year']).cumcount()

data['year'] = data['year'] + data['First_Moody_Year']
data[data['CRSP_Permno'] == 10006]
data.drop(['First_Moody_Year', 'Last_Moody_Year'], axis=1, inplace=True)
data.rename(columns={
    'CRSP_Permno': 'permno',
    'value': 'BE'}, inplace=True)
data['BE'] = data['BE'].astype(float)
moody_data = data.copy()

# %%
merged_data = monthly_price_data.merge(moody_data, on=['permno', 'year'], how='left')
merged_data = merged_data.merge(book_wrds, on=['permno', 'year'], how='left')
merged_data['bkvlps'] = merged_data['bkvlps'].astype(float)
merged_data['bkvlps'] = merged_data['bkvlps'] * merged_data['shrout'] / 1e6
merged_data['BookEquity'] = merged_data['bkvlps']
merged_data['BookEquity'] = merged_data['BookEquity'].fillna(merged_data['BE'])
merged_data.drop(columns=['bkvlps', 'BE'], inplace=True)
merged_data.describe()
# %%
merged_data.to_csv('Out/monthly_return_book_value.csv', index=False)
#%%