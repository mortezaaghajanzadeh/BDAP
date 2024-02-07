#%%
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
# %%
# Load the cleaned and merged data
df = pd.read_csv('out/monthly_return_book_value.csv').drop(columns=['gvkey'])
df['B/M'] = df.BookEquity/df.mcap
df['B/M'] = df['B/M'].replace([np.inf, -np.inf], np.nan)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df.date.dt.month
df['yearMonth'] = df.date.dt.to_period('M')
df.drop_duplicates(subset=['date', 'permno'], inplace=True)
df.drop(df.loc[df.prc_flag == 1].index, inplace=True)
#
#%% Find the Annual portfolio at the end of June
portfolio_selection = df.loc[df.month.isin([7,12])].copy()
print(portfolio_selection.shape)
portfolio_selection['mcap_flag'] = portfolio_selection.groupby(['permno', 'year'])['mcap'].transform(lambda x: 1 if x.isna().sum() > 0 else 0)
portfolio_selection = portfolio_selection.loc[portfolio_selection.mcap_flag == 0].copy()
print(portfolio_selection.shape)

mapping_dict = dict(zip(portfolio_selection.loc[portfolio_selection.month == 12].set_index(['permno', 'year']).index, portfolio_selection.loc[portfolio_selection.month == 12].mcap))

portfolio_selection.drop(portfolio_selection[portfolio_selection.month == 12].index, inplace=True)
portfolio_selection['mcap'] = portfolio_selection.set_index(['permno', 'year']).index.map(mapping_dict)
print(portfolio_selection.shape)
portfolio_selection.dropna(subset=['mcap'], inplace=True)
print(portfolio_selection.shape)
#%%
portfolio_selection = portfolio_selection.loc[portfolio_selection.month == 7].dropna(subset=['B/M','mcap']).copy()
portfolio_selection.drop_duplicates(subset=['permno', 'year'], inplace=True)
plot_portfolio_selection = portfolio_selection.copy()
portfolio_selection = portfolio_selection.loc[portfolio_selection['B/M'] > 0]
High_threshold = portfolio_selection[(portfolio_selection.hexcd == 1)].groupby('year')['B/M'].quantile(0.7).to_dict()
Low_threshold = portfolio_selection[(portfolio_selection.hexcd == 1)].groupby('year')['B/M'].quantile(0.3).to_dict()


#%% Value and Growth Portfolios
High_Big_portfolio = portfolio_selection.groupby('year')[['permno','B/M','mcap','hexcd']].apply(
    lambda x: x.loc[
        (
            x['B/M'] > x['B/M'].quantile(0.7)
        )&(
            x.mcap >= x[x.hexcd == 1].mcap.quantile(0.5)
            )
            ].permno.tolist()
    ).to_dict()
High_Big_portfolio = {(i,j): 1 for i in High_Big_portfolio for j in High_Big_portfolio[i]}

Low_Big_portfolio = portfolio_selection.groupby('year')[['permno','B/M','mcap','hexcd']].apply(lambda x: x.loc[(x['B/M'] <= x['B/M'].quantile(0.3))&(x.mcap >= x[x.hexcd == 1].mcap.quantile(0.5))].permno.tolist()).to_dict()
Low_Big_portfolio = {(i,j): 1 for i in Low_Big_portfolio for j in Low_Big_portfolio[i]}

High_Small_portfolio = portfolio_selection.groupby('year')[['permno','B/M','mcap','hexcd']].apply(lambda x: x.loc[(x['B/M'] > x['B/M'].quantile(0.7))&(x.mcap < x[x.hexcd == 1].mcap.quantile(0.5))].permno.tolist()).to_dict()
High_Small_portfolio = {(i,j): 1 for i in High_Small_portfolio for j in High_Small_portfolio[i]}

Low_Small_portfolio = portfolio_selection.groupby('year')[['permno','B/M','mcap','hexcd']].apply(lambda x: x.loc[(x['B/M'] <= x['B/M'].quantile(0.3))&(x.mcap < x[x.hexcd == 1].mcap.quantile(0.5))].permno.tolist()).to_dict()
Low_Small_portfolio = {(i,j): 1 for i in Low_Small_portfolio for j in Low_Small_portfolio[i]}

#%% Load the FF breakpoints
tempt = plot_portfolio_selection.reset_index(drop=True).copy()
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
# sns.lineplot(data=merged_df, x='year', y='value', hue='variable', style='data')
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
monthly_return_df = df.dropna(subset=['mcap','ret'])[['permno','hexcd','year','ret','yearMonth','month','mcap','B/M']]
monthly_return_df.groupby('yearMonth').ret.std().plot()
mapping_dict = monthly_return_df.loc[monthly_return_df.hexcd == 1].groupby('yearMonth').mcap.quantile(0.8).to_dict()
monthly_return_df['high_cap'] = monthly_return_df.yearMonth.map(mapping_dict)
monthly_return_df
#% Q3
monthly_return_df['BH'] = monthly_return_df.set_index(['year','permno']).index.map(High_Big_portfolio)
monthly_return_df['SH'] = monthly_return_df.set_index(['year','permno']).index.map(High_Small_portfolio)
monthly_return_df['BL'] = monthly_return_df.set_index(['year','permno']).index.map(Low_Big_portfolio)
monthly_return_df['SL'] = monthly_return_df.set_index(['year','permno']).index.map(Low_Small_portfolio)
monthly_return_df[['BH','SH','BL','SL']] = monthly_return_df[['BH','SH','BL','SL']].fillna(0)
bh_df = monthly_return_df.loc[monthly_return_df.BH == 1].copy()
sh_df = monthly_return_df.loc[monthly_return_df.SH == 1].copy()
bl_df = monthly_return_df.loc[monthly_return_df.BL == 1].copy()
sl_df = monthly_return_df.loc[monthly_return_df.SL == 1].copy()
#%%
def get_portfolio_return(df,weight):
    if weight == 'equal':
        return df.groupby('yearMonth').ret.mean()
    elif weight == 'capped_value':
        df.loc[df.mcap > df.high_cap, 'mcap'] = df.loc[df.mcap > df.high_cap].high_cap
        df['mweight'] = df.groupby('yearMonth')['mcap'].transform(lambda x: x/sum(x))
    else:
        df['mweight'] = df.groupby('yearMonth')[weight].transform(lambda x: x/sum(x))
    df['w_ret'] = df['mweight']*df['ret']
    return df.groupby('yearMonth').w_ret.sum()


return_df = pd.DataFrame()
return_df['BH'] = get_portfolio_return(bh_df,weight = 'mcap')
return_df['SH'] = get_portfolio_return(sh_df,weight = 'mcap')
return_df['BL'] = get_portfolio_return(bl_df,weight = 'mcap')
return_df['SL'] = get_portfolio_return(sl_df,weight = 'mcap')
return_df.reset_index(inplace=True)
return_df['yearMonth'] = return_df['yearMonth'].dt.to_timestamp()
return_df['HML'] = 0.5 * (return_df['BH'] + return_df['SH'] - return_df['BL'] - return_df['SL'])
return_df['cum_HML'] = ((1+return_df['HML']).cumprod()-1)

#%% Load FF 
ff_df = pd.read_csv('Data/F-F_Research_Data_Factors.csv')
ff_df.iloc[:,1:] = ff_df.iloc[:,1:]/100
ff_df['yearMonth'] = pd.to_datetime(ff_df['yearMonth'], format='%Y%m')
ff_df['cum_HML'] = ((1+ff_df['HML']).cumprod()-1)
sns.lineplot(data=return_df, x='yearMonth', y='cum_HML', label='Our')
sns.lineplot(data=ff_df, x='yearMonth', y='cum_HML', label='FF')
#%%
return_df[['yearMonth','HML','cum_HML']].merge(ff_df[['yearMonth','HML','cum_HML']], on='yearMonth', suffixes=('_our', '_ff'))[['HML_our','HML_ff','cum_HML_our','cum_HML_ff']].corr()
# %% Q4

return_df['BH_equal'] = get_portfolio_return(bh_df, weight='equal').reset_index(drop=True)
return_df['SH_equal'] = get_portfolio_return(sh_df, weight='equal').reset_index(drop=True)
return_df['BL_equal'] = get_portfolio_return(bl_df, weight='equal').reset_index(drop=True)
return_df['SL_equal'] = get_portfolio_return(sl_df, weight='equal').reset_index(drop=True)
return_df['HML_equal'] = 0.5 * (return_df['BH_equal'] + return_df['SH_equal'] - return_df['BL_equal'] - return_df['SL_equal'])
return_df['cum_HML_equal'] = ((1+return_df['HML_equal']).cumprod()-1)


#%%

return_df['BH_capped_value'] = get_portfolio_return(bh_df, weight='capped_value').reset_index(drop=True)
return_df['SH_capped_value'] = get_portfolio_return(sh_df, weight='capped_value').reset_index(drop=True)
return_df['BL_capped_value'] = get_portfolio_return(bl_df, weight='capped_value').reset_index(drop=True)
return_df['SL_capped_value'] = get_portfolio_return(sl_df, weight='capped_value').reset_index(drop=True)
return_df['HML_capped_value'] = 0.5 * (return_df['BH_capped_value'] + return_df['SH_capped_value'] - return_df['BL_capped_value'] - return_df['SL_capped_value'])
return_df['cum_HML_capped_value'] = ((1+return_df['HML_capped_value']).cumprod()-1)

sns.lineplot(data=return_df, x='yearMonth', y='cum_HML', label='mcap')
sns.lineplot(data=return_df, x='yearMonth', y='cum_HML_equal', label='equal')
sns.lineplot(data=return_df, x='yearMonth', y='cum_HML_capped_value', label='capped_value')
# %% Load JKP data
jkp_ew = pd.read_csv('Data/JKP/[usa]_[be_me]_[monthly]_[ew].csv')[['date','ret']]
jkp_ew['date'] = pd.to_datetime(jkp_ew['date'])
jkp_vw = pd.read_csv('Data/JKP/[usa]_[be_me]_[monthly]_[vw].csv')[['date','ret']]
jkp_vw['date'] = pd.to_datetime(jkp_vw['date'])
jkp_vw_cap = pd.read_csv('Data/JKP/[usa]_[be_me]_[monthly]_[vw_cap].csv')[['date','ret']]
jkp_vw_cap['date'] = pd.to_datetime(jkp_vw_cap['date'])

jkp_return_df = jkp_ew.merge(jkp_vw, on='date', suffixes=('_ew', '_vw')).merge(jkp_vw_cap, on='date').rename(columns={'ret':'ret_vw_cap'})

jkp_return_df['cum_ew'] = ((1+jkp_return_df['ret_ew']).cumprod()-1)
jkp_return_df['cum_vw'] = ((1+jkp_return_df['ret_vw']).cumprod()-1)
jkp_return_df['cum_vw_cap'] = ((1+jkp_return_df['ret_vw_cap']).cumprod()-1)
#%%
return_df['cum_HML_jkp'] = return_df['HML']
return_df.loc[return_df.yearMonth < jkp_return_df.date.min(), 'cum_HML_jkp'] = np.nan
return_df['cum_HML_jkp'] = ((1+return_df['cum_HML_jkp']).cumprod()-1)

return_df['cum_HML_equal_jkp'] = return_df['HML_equal']
return_df.loc[return_df.yearMonth < jkp_return_df.date.min(), 'cum_HML_equal_jkp'] = np.nan
return_df['cum_HML_equal_jkp'] = ((1+return_df['cum_HML_equal_jkp']).cumprod()-1)

return_df['cum_HML_capped_value_jkp'] = return_df['HML_capped_value']
return_df.loc[return_df.yearMonth < jkp_return_df.date.min(), 'cum_HML_capped_value_jkp'] = np.nan
return_df['cum_HML_capped_value_jkp'] = ((1+return_df['cum_HML_capped_value_jkp']).cumprod()-1)


#%%
fig, ax = plt.subplots(3, sharex=True, figsize=(10,10))
sns.lineplot(data=return_df, x='yearMonth', y='cum_HML_jkp', label='Our', ax=ax[0])
sns.lineplot(data=jkp_return_df, x='date', y='cum_vw', label='JKP', ax=ax[0])
sns.lineplot(data=return_df, x='yearMonth', y='cum_HML_equal_jkp', label='Our', ax=ax[1])
sns.lineplot(data=jkp_return_df, x='date', y='cum_ew', label='JKP', ax=ax[1])
sns.lineplot(data=return_df, x='yearMonth', y='cum_HML_capped_value_jkp', label='Our', ax=ax[2])
sns.lineplot(data=jkp_return_df, x='date', y='cum_vw_cap', label='JKP', ax=ax[2])

# set axis labels
ax[0].set_title('Value-Weighted')
ax[1].set_title('Equal-Weighted')
ax[2].set_title('Value-Weighted with Cap')
ax[0].set_ylabel('HML Portfolio Return')
ax[1].set_ylabel('HML Portfolio Return')
ax[2].set_ylabel('HML Portfolio Return')
ax[2].set_xlabel('Year')

plt.show()
#%%
