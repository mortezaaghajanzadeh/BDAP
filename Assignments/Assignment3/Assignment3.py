#%%
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
import statsmodels.api as sm
from tqdm import tqdm
import matplotlib.pyplot as plt
#%%
data_path = 'data/{}'
out_path = 'out/{}'
#%%
data = ['ew','vw', 'vw_cap']

df = pd.DataFrame()
for key in data:
    df = pd.concat([df, pd.read_csv(data_path.format('[usa]_[all_factors]_[monthly]_[{}].csv'.format(key))).drop(columns = ['location','direction','n_stocks','n_stocks_min','freq'])])
df['model'] = df.weighting 
df.loc[df.model == 'vw_cap', 'model'] = 'vw'
df['date'] = pd.to_datetime(df.date)
# %% (1)
test_1 = df.groupby(['name','weighting']).ret.apply(lambda x: ttest_1samp(x, 0)[1]).to_frame().reset_index()
df_1 = pd.DataFrame()
df_1 = df.groupby(['name','weighting']).ret.mean().reset_index()
df_1 = df_1.merge(test_1, on = ['name','weighting'],suffixes=('_mean','_pval'))
df_1['replicate'] = 0
df_1.loc[(df_1.ret_mean > 0)&(abs(df_1.ret_pval) < 0.05), 'replicate'] = 1
# bar plot
res_1 = df_1.groupby(['weighting']).replicate.mean()
res_1.plot(kind = 'bar', color = 'skyblue', edgecolor = 'black', linewidth = 1.2, figsize = (10,6), rot = 0, fontsize = 12)
plt.xlabel('Weighting Scheme')
plt.ylabel('Proportion of Replicating Factors')
plt.title('Proportion of Replicating Factors by Weighting')
plt.xticks(rotation = 0)
plt.xticks(np.arange(3), ['Equal Weighted', 'Value Weighted', 'Value Weighted with Cap'])
vals = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in vals])
plt.savefig(out_path.format('1.png'), bbox_inches = 'tight')
plt.savefig(out_path.format('1.pdf'), bbox_inches = 'tight')
plt.show()

#%% (2)
factor_details = pd.read_excel('data/Factor Details.xlsx')[['abr_jkp','significance']].dropna()
significant_factor = list(factor_details.loc[factor_details.significance== 1].abr_jkp.unique())
# keep only significant factors in the original study
df = df.loc[df.name.isin(significant_factor)].copy()

test_2 = df.groupby(['name','weighting']).ret.apply(lambda x: ttest_1samp(x, 0)[1]).to_frame().reset_index()
df_2 = pd.DataFrame()
df_2 = df.groupby(['name','weighting']).ret.mean().reset_index()
df_2 = df_2.merge(test_2, on = ['name','weighting'],suffixes=('_mean','_pval'))
df_2['replicate'] = 0
df_2.loc[(df_2.ret_mean > 0)&(abs(df_2.ret_pval) < 0.05), 'replicate'] = 1
res_2 = df_2.groupby(['weighting']).replicate.mean()
res_2.plot(kind = 'bar', color = 'skyblue', edgecolor = 'black', linewidth = 1.2, figsize = (10,6), rot = 0, fontsize = 12)
plt.xlabel('Weighting Scheme')
plt.ylabel('Proportion of Replicating Factors')
plt.title('Proportion of Replicating Factors by Weighting (Originally Significant Factors Only)')
plt.xticks(rotation = 0)
plt.xticks(np.arange(3), ['Equal Weighted', 'Value Weighted', 'Value Weighted with Cap'])
vals = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in vals])
plt.savefig(out_path.format('2.png'), bbox_inches = 'tight')
plt.savefig(out_path.format('2.pdf'), bbox_inches = 'tight')
#%% (3)
market_return = pd.read_csv(data_path.format('market_returns.csv'))
market_return = market_return[market_return.excntry == 'USA'].reset_index(drop=True)[['eom','mkt_vw_exc','mkt_ew_exc']].copy()
market_return.rename(columns={'eom':'date','mkt_vw_exc':"vw",'mkt_ew_exc':"ew"}, inplace=True)
market_return['date'] = pd.to_datetime(market_return.date)

market_return_map = market_return.melt(id_vars = 'date', value_vars = ['ew','vw'], var_name = 'model', value_name = 'market_return')
mkt_mapping_dict = dict(zip(market_return_map.set_index(['date','model']).index,market_return_map.market_return))

def time_series_regression(portfolios, factors, FactorModel,alpha_test = True):
    portfolios = portfolios.merge(factors, on='date', how='left')
    portfolios = portfolios.dropna()
    X = portfolios[FactorModel]
    X = sm.add_constant(X)
    Y = portfolios['ret']
    # model = sm.OLS(Y, X).fit(cov_type='HAC',cov_kwds={'maxlags':int(len(Y)**0.25)}) 
    model = sm.OLS(Y, X).fit()
    pvalues = model.pvalues
    betas = model.params
    if alpha_test:
        return [betas.iloc[0],pvalues.iloc[0]]
    else:
        return [betas,pvalues]

df_3 = df.groupby(['name','weighting'])[['date','ret','model']].apply(lambda x: time_series_regression(x, market_return, list(x.model.unique()))).to_frame().reset_index()
df_3[['alpha','pval']] = pd.DataFrame(df_3[0].tolist(), index= df_3.index)
df_3 = df_3.drop(columns = [0])
df_3['replicate'] = 0
df_3.loc[(df_3.alpha > 0)&(abs(df_3.pval) < 0.05), 'replicate'] = 1
res_3 = df_3.groupby(['weighting']).replicate.mean()
res_3.plot(kind = 'bar', color = 'skyblue', edgecolor = 'black', linewidth = 1.2, figsize = (10,6), rot = 0, fontsize = 12)
plt.xlabel('Weighting Scheme')
plt.ylabel('Proportion of Replicating Factors')
plt.title('Proportion of Replicating Factors by Weighting (Factor Model)')
plt.xticks(rotation = 0)
plt.xticks(np.arange(3), ['Equal Weighted', 'Value Weighted', 'Value Weighted with Cap'])
vals = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in vals])
plt.savefig(out_path.format('3.png'), bbox_inches = 'tight')
plt.savefig(out_path.format('3.pdf'), bbox_inches = 'tight')
#%% (4)
# Bonferroni Adjustment
df_4 = df_3[['name','weighting','alpha','pval']].copy()
df_4['replicate'] = 0
df_4.loc[(df_4.alpha > 0)&(abs(df_4.pval) < 0.05/df_4.name.nunique()), 'replicate'] = 1
res_4 = df_4.groupby(['weighting']).replicate.mean()
res_4.plot(kind = 'bar', color = 'skyblue', edgecolor = 'black', linewidth = 1.2, figsize = (10,6), rot = 0, fontsize = 12)
plt.xlabel('Weighting Scheme')
plt.ylabel('Proportion of Replicating Factors')
plt.title('Proportion of Replicating Factors by Weighting (Bonferroni Adjustment)')
plt.xticks(rotation = 0)
plt.xticks(np.arange(3), ['Equal Weighted', 'Value Weighted', 'Value Weighted with Cap'])
vals = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in vals])
plt.savefig(out_path.format('4.png'), bbox_inches = 'tight')
plt.savefig(out_path.format('4.pdf'), bbox_inches = 'tight')

#%% (5)
# Benjamini-Hochberg
df_5 = df_3[['name','weighting','alpha','pval']].copy()
df_5 = df_5.groupby(['weighting'])[['name','alpha','pval']].apply(lambda x: x.sort_values('pval')).reset_index().drop(columns = ['level_1'])
df_5['ratio'] = df_5.groupby(['weighting']).cumcount()+1
df_5['ratio'] = df_5['ratio'] / df_5.groupby(['weighting'])['ratio'].transform('count') 
df_5['replicate'] = 0
df_5.loc[(df_5.alpha > 0)&(df_5.pval < df_5.ratio*0.05), 'replicate'] = 1
res_5 = df_5.groupby(['weighting']).replicate.mean()
res_5.plot(kind = 'bar', color = 'skyblue', edgecolor = 'black', linewidth = 1.2, figsize = (10,6), rot = 0, fontsize = 12)
plt.xlabel('Weighting Scheme')
plt.ylabel('Proportion of Replicating Factors')
plt.title('Proportion of Replicating Factors by Weighting (Benjamini-Hochberg Adjustment)')
plt.xticks(rotation = 0)
plt.xticks(np.arange(3), ['Equal Weighted', 'Value Weighted', 'Value Weighted with Cap'])
vals = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in vals])
plt.savefig(out_path.format('5.png'), bbox_inches = 'tight')
plt.savefig(out_path.format('5.pdf'), bbox_inches = 'tight')
#%% (6)
estimation_df = df.groupby(['name','weighting'])[['date','ret','model']].apply(lambda x: time_series_regression(x, market_return, list(x.model.unique()),alpha_test = False)).to_frame().reset_index()
estimation_df[['coef','pval']] = pd.DataFrame(estimation_df[0].tolist(), index= estimation_df.index)
estimation_df[['alpha','beta_1','beta_2']] = pd.DataFrame(estimation_df.coef.tolist(), index= estimation_df.index)
estimation_df['beta'] = estimation_df['beta_1'].fillna(0) + estimation_df['beta_2'].fillna(0)
estimation_df = estimation_df.drop(columns = [0,'pval','coef','beta_1','beta_2'])
alpha_mapping = dict(zip(estimation_df.set_index(['name','weighting']).index,estimation_df.alpha))
beta_mapping = dict(zip(estimation_df.set_index(['name','weighting']).index,estimation_df.beta))
# %%
cluster_df = pd.read_csv(data_path.format('Cluster Labels.csv'))
cluster_mapping_dict = dict(zip(cluster_df['characteristic'], cluster_df['cluster']))
df_6 = df.loc[(df.date.dt.year > 1971)&(df.date.dt.year < 2022)][['name', 'weighting', 'date', 'ret', 'model']].copy()
df_6['cluster'] = df_6.name.map(cluster_mapping_dict)
df_6['beta'] = df_6.set_index(['name','weighting']).index.map(beta_mapping)
df_6['mkt'] = df_6.set_index(['date','model']).index.map(mkt_mapping_dict)
df_6['mafr'] = df_6['ret'] - df_6['mkt'] * df_6['beta']
df_6['mafr_adjusted'] = df_6.groupby(['name','weighting']).mafr.transform(lambda x: x/x.std() * 0.1 / np.sqrt(12))
#%%
df_6['cluster_id'],cluster_key = pd.factorize(df_6['cluster'])
df_6['factor_id'],factor_key = pd.factorize(df_6['name'])
clusters = df_6.groupby('cluster_id').name.apply(lambda x: x.unique()).to_dict()

df_6_ew = df_6.loc[df_6.weighting == 'ew'].copy()
df_6_vw = df_6.loc[df_6.weighting == 'vw'].copy()
df_6_vw_cap = df_6.loc[df_6.weighting == 'vw_cap'].copy()

results_ew = np.zeros((len(clusters),len(clusters)))
results_vw = np.zeros((len(clusters),len(clusters)))
results_vw_cap = np.zeros((len(clusters),len(clusters)))
for cluster_1 in tqdm(clusters):
    for cluster_2 in clusters:
        tempt = 0
        tempt_vw = 0
        tempt_vw_cap = 0
        number = 0
        for i in clusters[cluster_1]:
            cluster_i_df = df_6_ew.loc[df_6_ew.name == i].mafr_adjusted
            cluster_i_df_vw = df_6_vw.loc[df_6_vw.name == i].mafr_adjusted
            cluster_i_df_vw_cap = df_6_vw_cap.loc[df_6_vw_cap.name == i].mafr_adjusted

            for j in clusters[cluster_2]:
                tempt += np.correlate(cluster_i_df, df_6_ew.loc[df_6_ew.name == j].mafr_adjusted)[0]
                tempt_vw += np.correlate(cluster_i_df_vw, df_6_vw.loc[df_6_vw.name == j].mafr_adjusted)[0]
                tempt_vw_cap += np.correlate(cluster_i_df_vw_cap, df_6_vw_cap.loc[df_6_vw_cap.name == j].mafr_adjusted)[0]
                number += 1

        results_ew[cluster_1,cluster_2] = tempt/number
        results_vw[cluster_1,cluster_2] = tempt_vw/number
        results_vw_cap[cluster_1,cluster_2] = tempt_vw_cap/number

# %%
pd.DataFrame(results_vw_cap, index = cluster_key, columns = cluster_key).to_latex(out_path.format('6.tex'), float_format="%.2f",column_format="l" + "c"*len(cluster_key))

#%%

cluster_key = list(cluster_key)
C_block = np.identity((len(factor_key)))
factor_indices = [cluster_key.index(cluster_mapping_dict[factor]) for factor in factor_key]
C_block = results_vw_cap[np.ix_(factor_indices, factor_indices)]
# set the diagonal to 1
np.fill_diagonal(C_block, 1)

sigma = np.diag(df_6_vw_cap.groupby(['factor_id']).mafr_adjusted.std().to_numpy())
S_block = sigma @ C_block @ sigma
S_block[:3,:3]
#%% (7)
