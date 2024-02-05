#%%
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
# %%
# Load the cleaned and merged data
df = pd.read_csv('out/monthly_return_book_value.csv').drop(columns=['gvkey'])
df['B/M'] = df.BookEquity/df.mcap
df['B/M'] = df['B/M'].replace([np.inf, -np.inf], np.nan)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df.date.dt.month
df['yearMonth'] = df.date.dt.to_period('M')
df.drop_duplicates(subset=['date', 'permno'], inplace=True)
#%% Find the Annual portfolio at the end of June
portfolio_selection = df.loc[df.month == 6].dropna(subset=['B/M','mcap']).copy()
portfolio_selection = portfolio_selection.loc[portfolio_selection['B/M'] > 0]
High_threshold = portfolio_selection.groupby('year')['B/M'].quantile(0.7).to_dict()
Low_threshold = portfolio_selection.groupby('year')['B/M'].quantile(0.3).to_dict()
High_Big_portfolio = portfolio_selection.groupby('year')[['permno','B/M','mcap','hexcd']].apply(lambda x: x.loc[(x['B/M'] > x['B/M'].quantile(0.7))&(x.mcap >= x[x.hexcd == 1].mcap.quantile(0.5))].permno.tolist()).to_dict()
Low_Big_portfolio = portfolio_selection.groupby('year')[['permno','B/M','mcap','hexcd']].apply(lambda x: x.loc[(x['B/M'] <= x['B/M'].quantile(0.3))&(x.mcap >= x[x.hexcd == 1].mcap.quantile(0.5))].permno.tolist()).to_dict()
High_Small_portfolio = portfolio_selection.groupby('year')[['permno','B/M','mcap','hexcd']].apply(lambda x: x.loc[(x['B/M'] > x['B/M'].quantile(0.7))&(x.mcap < x[x.hexcd == 1].mcap.quantile(0.5))].permno.tolist()).to_dict()
Low_Small_portfolio = portfolio_selection.groupby('year')[['permno','B/M','mcap','hexcd']].apply(lambda x: x.loc[(x['B/M'] <= x['B/M'].quantile(0.3))&(x.mcap < x[x.hexcd == 1].mcap.quantile(0.5))].permno.tolist()).to_dict()

#%% Load the FF breakpoints
tempt = df.loc[df.month == 6].copy().drop_duplicates(subset=['date', 'permno']).reset_index(drop=True)
tempt['year'] = tempt.date.dt.year
positive_number = tempt.groupby('year')['B/M'].apply(lambda x: len(x[x >0])).to_dict()
total_number = tempt.groupby('year').size().to_dict()
negative_number = tempt.groupby('year')['B/M'].apply(lambda x: len(x[x <= 0])).to_dict()
tempt_df = pd.DataFrame([positive_number, negative_number, total_number], index=['positive', 'negative', 'total']).T
tempt_df.reset_index(inplace=True)
tempt_df.rename(columns={'index':'year'}, inplace=True)
tempt_df = tempt_df.melt(id_vars='year', value_vars=['positive', 'negative'])
tempt_df['data'] = 'our'
raw_df_breakpoint_FF = pd.read_csv('Data/BE-ME_Breakpoints.csv')
raw_df = raw_df_breakpoint_FF[['year','<=0',">0"]].copy().rename(columns={'<=0':'negative', '>0':'positive'})
raw_df['year'] = raw_df['year'].astype(int)
raw_df['data'] = 'FF'
raw_df = raw_df.melt(id_vars=['year', 'data'], value_vars=['positive', 'negative'])
merged_df = pd.concat([tempt_df, raw_df])
sns.lineplot(data=merged_df, x='year', y='value', hue='variable', style='data')
#%% 
df_breakpoint_FF = raw_df_breakpoint_FF[['year','0.3','0.7']].copy()
df_breakpoint_FF.columns = ['year','Growth','Value']
df_breakpoint_FF['year'] = df_breakpoint_FF['year'].astype(int)
df_breakpoint_FF = df_breakpoint_FF.melt(id_vars='year', value_vars=['Growth', 'Value'], var_name='Threshold', value_name='B/M').sort_values(by=['year', 'Threshold']).reset_index(drop=True)
df_breakpoint_FF['calculation'] = 'FF'
df_breakpoint = pd.DataFrame({'High_threshold': High_threshold, 'Low_threshold': Low_threshold})
df_breakpoint = df_breakpoint.reset_index()
df_breakpoint.rename(columns={'index': 'year'}, inplace=True)
df_breakpoint = df_breakpoint.melt(id_vars='year', value_vars=['High_threshold', 'Low_threshold'], var_name='Threshold', value_name='B/M')
df_breakpoint['Threshold'] = df_breakpoint.Threshold.map({'High_threshold': 'Value', 'Low_threshold': 'Growth'})
df_breakpoint['calculation'] = 'Our'
df_breakpoint = pd.concat([df_breakpoint, df_breakpoint_FF], ignore_index=True)
sns.set(style="whitegrid")
ax = sns.lineplot(x="year", y="B/M", hue="Threshold", style="calculation", data=df_breakpoint)
ax.set_title('B/M Breakpoint')
#%% Q2
#%% Q3
monthly_return_df = df.dropna(subset=['mcap','ret']).copy()
monthly_return_df[['permno','year','ret','yearMonth','month','mcap']]
#%%

def get_portfolio_return(df):
    df['mweight'] = df.groupby('yearMonth')['mcap'].transform(lambda x: x/sum(x))
    df['w_ret'] = df['mweight']*df['ret']
    return df.groupby('yearMonth').w_ret.sum()



value_big_returns = pd.DataFrame()
growth_big_returns = pd.DataFrame()
value_small_returns = pd.DataFrame()
growth_small_returns = pd.DataFrame()
for i in tqdm(High_Big_portfolio):
    annual_df = pd.DataFrame()
    annual_df = pd.concat([annual_df, monthly_return_df.loc[monthly_return_df.year == i]])


    value_big_portfolio = annual_df.loc[annual_df.permno.isin(High_Big_portfolio[i])].copy()
    if len(value_big_returns) == 0:
        value_big_returns = get_portfolio_return(value_big_portfolio)
    else:
        value_big_returns = pd.concat([value_big_returns, get_portfolio_return(value_big_portfolio)])
    
    growth_big_portfolio = annual_df.loc[annual_df.permno.isin(Low_Big_portfolio[i])].copy()
    if len(growth_big_returns) == 0:
        growth_big_returns = get_portfolio_return(growth_big_portfolio)
    else:
        growth_big_returns = pd.concat([growth_big_returns, get_portfolio_return(growth_big_portfolio)])
    
    value_small_portfolio = annual_df.loc[annual_df.permno.isin(High_Small_portfolio[i])].copy()
    if len(value_small_returns) == 0:
        value_small_returns = get_portfolio_return(value_small_portfolio)
    else:
        value_small_returns = pd.concat([value_small_returns, get_portfolio_return(value_small_portfolio)])
    
    growth_small_portfolio = annual_df.loc[annual_df.permno.isin(Low_Small_portfolio[i])].copy()
    if len(growth_small_returns) == 0:
        growth_small_returns = get_portfolio_return(growth_small_portfolio)
    else:
        growth_small_returns = pd.concat([growth_small_returns, get_portfolio_return(growth_small_portfolio)])

    ## The number of stocks in the portfolio are the same !!!!
    # print(len(value_portfolio), len(growth_portfolio))

    # break
return_df = pd.DataFrame({'Value_Big': value_big_returns, 'Growth_Big': growth_big_returns, 'Value_Small': value_small_returns, 'Growth_Small': growth_small_returns}).reset_index()
return_df['yearMonth'] = return_df['yearMonth'].dt.to_timestamp()
return_df['HML'] = 0.5 * (return_df['Value_Big'] + return_df['Value_Small'] - return_df['Growth_Big'] - return_df['Growth_Small'])
return_df['cum_HML'] = ((1+return_df['HML']/100).cumprod()-1)*100

#%% Load FF 
ff_df = pd.read_csv('Data/F-F_Research_Data_Factors.csv')
ff_df.iloc[:,1:] = ff_df.iloc[:,1:]/100
ff_df['yearMonth'] = pd.to_datetime(ff_df['yearMonth'], format='%Y%m')
ff_df['cum_HML'] = ((1+ff_df['HML']/100).cumprod()-1)*100
sns.lineplot(data=return_df, x='yearMonth', y='cum_HML', label='Our')
sns.lineplot(data=ff_df, x='yearMonth', y='cum_HML', label='FF')
#%%
return_df[['yearMonth','HML']].merge(ff_df[['yearMonth','HML']], on='yearMonth', suffixes=('_our', '_ff'))[['HML_our','HML_ff']].corr()

# %%
