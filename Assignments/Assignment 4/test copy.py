#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%
data_path = 'data/'
out_path = 'out/'
characteristics = [
    'be_me', 'ret_12_1', 'market_equity', 'ret_1_0', 'rvol_252d', 'beta_252d', 'qmj_safety', 'rmax1_21d', 'chcsho_12m', 'ni_me', 'eq_dur', 'ret_60_12', 'ope_be', 'gp_at', 'ebit_sale', 'at_gr1', 'sale_gr1', 'at_be', 'cash_at', 'age', 'z_score'
]
in_sample_deadline = '2012-01-01'
start_time = '1991-12-31'
columns = ['id',
 'date',
 'eom',
 'source_crsp',
 'size_grp','ret','f_ret'] + characteristics
#%%
df = pd.read_csv(data_path + 'gbr.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
#%%
#%% (1)
# a
df = df.loc[df['date'] > start_time].copy()
# b
df.dropna(subset='market_equity', inplace=True)
df['tempt'] = df.groupby('id')['ret'].shift(-1)
df = df.loc[~df.tempt.isna()].drop(columns=['tempt']).copy()
# c
df['na_num'] = df[characteristics].isna().sum(axis=1)
df = df.loc[df['na_num'] < 5].drop(columns = ['na_num']).copy()
# d
df = df.loc[df.size_grp != 'nano'].copy()
# e
df[characteristics] = df.groupby('eom')[characteristics].transform(lambda x: x.fillna(x.median()))
# f
df[characteristics] = df.groupby('eom')[characteristics].transform(lambda x: (x - x.mean()) / x.std())
df['ret'] = df['ret_exc']
df['f_ret'] = df['ret_exc_lead1m']
df = df[columns].copy()
# %
df['month'] = df['date'].dt.to_period('M')
df.groupby('month').id.count().plot(title='Number of stocks per month', ylabel='Number of stocks', xlabel='Year', figsize=(10, 5), grid=True, legend=False, color='black',dashes=[6, 2])
plt.savefig(out_path + '1.png', dpi=300, bbox_inches='tight')
plt.savefig(out_path + '1.pdf', dpi=300, bbox_inches='tight')
plt.show()
# %%
in_sample_df = df[df['date'] < in_sample_deadline].dropna().copy()
out_sample_df = df[df['date'] >= in_sample_deadline].dropna().copy()
# %% (2)
prediction_model = characteristics
# a OLS regression (Fama-MacBeth)
from linearmodels import FamaMacBeth
etdata = in_sample_df.set_index(['id','date']).copy()
results = FamaMacBeth(etdata.f_ret, etdata[prediction_model]).fit(cov_type='kernel', kernel='newey-west', bandwidth=len(etdata) ** 0.25)

from linearmodels import PanelOLS

# Perform pooled OLS regression
pooled_results = PanelOLS.from_formula('f_ret ~ ' + ' + '.join(prediction_model), data=etdata).fit(cov_type='kernel', kernel='newey-west', bandwidth=len(etdata) ** 0.25)

# Assign the pooled OLS results to a variable
# model_a = results
model_a = pooled_results




# Save results as LaTeX output
latex_output = results.summary.tables[1].as_latex_tabular(center = False, column_format = 'l' + 'c'*len(prediction_model), longtable = True, escape = False,bold_rows = False,bold_cols = False)

# Save LaTeX output to a file
with open(out_path + '6_2.tex', 'w') as file:
    file.write(latex_output)
# %% 2(b)
# Ridge regression   
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

#define predictor and response variables
X = in_sample_df[prediction_model].values
y = in_sample_df['f_ret'].values


def find_mse(lambda_0):
    # initialize list to store mean squared errors
    mse_scores = []

    # perform 3-fold cross-validation
    kf = KFold(n_splits=3, shuffle=True, random_state=1)
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # define model

        model = Ridge(alpha=lambda_0)
        
        # fit model
        model.fit(X_train, y_train)
        
        # make predictions on validation set
        y_pred = model.predict(X_val)
        
        # compute mean squared error
        mse = mean_squared_error(y_val, y_pred)
        
        # store mse score
        mse_scores.append(mse)
    return np.mean(mse_scores)



ranges = [np.arange(0, 1e4, 1e2), np.arange(-1e2, 1e2, 10),np.arange(-1, 1, 0.01)]
min_mse_lambda = 0
for _,range_ in enumerate(ranges):
    range_ += min_mse_lambda
    mse_res = [find_mse(lambda_0) for lambda_0 in tqdm(range_)]
    if _ == 0:
        plt.plot(range_, mse_res)
        plt.xlabel('Lambda')
        plt.ylabel('Mean Squared Error')
        plt.title('Ridge Regression')
        plt.savefig(out_path + '6_2.png', dpi=300, bbox_inches='tight')
        plt.savefig(out_path + '6_2.pdf', dpi=300, bbox_inches='tight')
    # Find the minimum
    min_mse = np.min(mse_res)
    min_mse_index = np.argmin(mse_res)
    min_mse_lambda = range_[min_mse_index]
print(f'Minimum MSE: {min_mse} at lambda: {min_mse_lambda}')
model = Ridge(alpha=min_mse_lambda)
print(f'Coefficients based on the optimal lambda: {model.fit(X, y).coef_}')
model_b = model.fit(X, y)
# %% 2(c)
# Random Forest
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
# prediction_model = characteristics

# X = in_sample_df[prediction_model].values
# y = in_sample_df['f_ret'].values

dt = RandomForestRegressor(max_depth=4, max_features=3, min_samples_leaf=1, random_state=1)
dt.fit(X, y)

params_dt = {
    'max_features': [7, 14, 21],
    'max_depth': [1, 2, 3],
    'min_samples_leaf': [1, 10]
}

from sklearn.model_selection import GridSearchCV


# Instantiate grid_dt
grid_dt = GridSearchCV(estimator=dt,
                       param_grid=params_dt,
                       scoring='neg_mean_squared_error',
                       cv=5,
                       n_jobs=-1)

grid_dt.fit(X, y)
best_model = grid_dt.best_estimator_
print(f'Best model: {best_model}')

mse_results = []
for param in params_dt['max_features']:
    dt = RandomForestRegressor(max_depth= best_model.max_depth, max_features=param, min_samples_leaf=best_model.min_samples_leaf, random_state=1)
    dt.fit(X, y)
    y_pred = dt.predict(X)
    mse = mean_squared_error(y, y_pred)
    mse_results.append(mse)

plt.plot(params_dt['max_features'], mse_results)
plt.xlabel('Number of features')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs Number of Features')
plt.savefig(out_path + '6_3_1.png', dpi=300, bbox_inches='tight')
plt.savefig(out_path + '6_3_1.pdf', dpi=300, bbox_inches='tight')
plt.show()

mse_results = []
for param in params_dt['max_depth']:
    dt = RandomForestRegressor(max_depth=param, max_features=best_model.max_features, min_samples_leaf=best_model.min_samples_leaf, random_state=1)
    dt.fit(X, y)
    y_pred = dt.predict(X)
    mse = mean_squared_error(y, y_pred)
    mse_results.append(mse)

plt.plot(params_dt['max_depth'], mse_results)
plt.xlabel('Maximum Tree Depth')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs Maximum Tree Depth')
plt.savefig(out_path + '6_3_2.png', dpi=300, bbox_inches='tight')
plt.savefig(out_path + '6_3_2.pdf', dpi=300, bbox_inches='tight')
plt.show()

mse_results = []
for param in params_dt['min_samples_leaf']:
    dt = RandomForestRegressor(max_depth=best_model.max_depth, max_features=best_model.max_features, min_samples_leaf=param, random_state=1)
    dt.fit(X, y)
    y_pred = dt.predict(X)
    mse = mean_squared_error(y, y_pred)
    mse_results.append(mse)

plt.plot(params_dt['min_samples_leaf'], mse_results)
plt.xlabel('Minimum Samples per Leaf Node')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs Minimum Samples per Leaf Node')
plt.savefig(out_path + '6_3_3.png', dpi=300, bbox_inches='tight')
plt.savefig(out_path + '6_3_3.pdf', dpi=300, bbox_inches='tight')
plt.show()
#%% (3)
import tabulate

in_sample_r2_a = model_a.rsquared
in_sample_r2_b = model_b.score(X, y)
in_sample_r2_c = best_model.score(X, y)

table = [["Model", "In-sample R2"],
         ["Model A", in_sample_r2_a],
         ["Model B", in_sample_r2_b],
         ["Model C", in_sample_r2_c]]

table_str = tabulate.tabulate(table, headers="firstrow", tablefmt="latex", floatfmt=".4f", numalign="center", stralign="center", colalign=("center", "center"))

# Save table to a file
with open(out_path + "6_3.tex", "w") as file:
    file.write(table_str)

print(table_str)
# %% (4)
# Feature importance
importances_a = abs(model_a.params)
importances_b = abs(model_b.coef_)
importances_c = best_model.feature_importances_
importances_df = pd.DataFrame({'Model A': importances_a, 'Model B': importances_b, 'Model C': importances_c})
ax = importances_df.plot(kind='bar', title='Feature Importance', xlabel='Feature', ylabel='Importance', figsize=(10, 5), grid=True)
ax.set_xticklabels(importances_df.index, rotation=45)
plt.savefig(out_path + '6_4.png', dpi=300, bbox_inches='tight')
plt.savefig(out_path + '6_4.pdf', dpi=300, bbox_inches='tight')
plt.show()
# %% (5)
## (a)
# Out-of-sample R2
X = out_sample_df[prediction_model].values
y = out_sample_df['f_ret'].values
y_train = in_sample_df['f_ret'].values
out_sample_r2_a = 1 - (sum(y - np.array(model_a.params) @ X.T) / sum((y - y_train.mean()) ** 2))
out_sample_r2_b = model_b.score(X, y)
out_sample_r2_c = best_model.score(X, y)

table = [["Model", "Out-sample R2"],
         ["Model A", out_sample_r2_a],
         ["Model B", out_sample_r2_b],
         ["Model C", out_sample_r2_c]]

table_str = tabulate.tabulate(table, headers="firstrow", tablefmt="latex", floatfmt=".4f", numalign="center", stralign="center", colalign=("center", "center"))

# Save table to a file
with open(out_path + "6_5_a.tex", "w") as file:
    file.write(table_str)
# %% 5(b)
def OLS_predict(X, params):
    return np.array(params) @ X.T
def weighted_average(group):
    return np.average(group['f_ret'], weights=group['market_equity'])
X = out_sample_df[prediction_model].values
out_sample_df['pred_a'] = OLS_predict(X, model_a.params)
out_sample_df['pred_b'] = model_b.predict(X)
out_sample_df['pred_c'] = best_model.predict(X)
df_5 = out_sample_df[['id', 'date', 'eom', 'market_equity', 'f_ret', 'pred_a', 'pred_b', 'pred_c']].copy()
#%%
df_5['portfolio_a'] = df_5.groupby('eom')['pred_a'].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')).astype(int)+1
df_5['portfolio_b'] = df_5.groupby('eom')['pred_b'].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')).astype(int)+1
df_5['portfolio_c'] = df_5.groupby('eom')['pred_c'].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')).astype(int)+1
res5_df = pd.DataFrame(columns=['Model', 'Portfolio', 'Mean Return', 'Standard Deviation', 'Sharpe Ratio'])
for model in ['a', 'b', 'c']:
    mean_ret = df_5.groupby([f'portfolio_{model}','eom']).apply(weighted_average,include_groups=False).reset_index().rename(columns={f'portfolio_{model}': 'portfolio', 0: f'portfolio_{model}'})
    # print(mean_ret[mean_ret.portfolio == 5] )
    if len(res5_df) == 0:
        res5_df = mean_ret
    else:
        res5_df = pd.merge(res5_df, mean_ret, on=['eom', 'portfolio'], how='outer')

model = 'c'
df_5.groupby([f'portfolio_{model}','eom']).apply(weighted_average,include_groups=False).reset_index().rename(columns={f'portfolio_{model}': 'portfolio', 0: f'portfolio_{model}'}).portfolio.value_counts()
# res5_df[res5_df.portfolio == 5]
# df_5[df_5.eom == 20151231].portfolio_a.value_counts()
# weighted_average(df_5[(df_5.eom == 20151231)&(df_5.portfolio_a == 5)])
df_5[df_5.portfolio_c == 5].eom.value_counts()

res5_df = res5_df.pivot(index='portfolio', columns='eom').T.reset_index().rename(columns={'level_0': 'Model'}).rename_axis(None, axis=1)
res5_df['long_short'] = res5_df[5] - res5_df[1]
res5_df['eom'] = pd.to_datetime(res5_df['eom'], format='%Y%m%d')
res5_df.isnull().sum()

#%%
factors_df = pd.read_csv(data_path + 'market_returns.csv')
factors_df = factors_df.loc[factors_df.excntry == 'GBR'].copy()
factors_df['RF'] = factors_df.mkt_vw - factors_df.mkt_vw_exc
factors_df['mkt'] = factors_df.mkt_vw
factors_df['eom'] = pd.to_datetime(factors_df['eom'], format='%Y-%m-%d')

mapping_dict = dict(zip(factors_df.eom, factors_df.RF))
res5_df['rf'] = res5_df['eom'].map(mapping_dict)

mapping_dict = dict(zip(factors_df.eom, factors_df.mkt))
res5_df['mkt'] = res5_df['eom'].map(mapping_dict)
for i in range(1, 6):
    res5_df[i] = res5_df[i] - res5_df.rf
#%%
import statsmodels.api as sm

res5_df['excess_return'] = res5_df.long_short
res5_df['excess_mkt'] = res5_df.mkt - res5_df.rf


def CAPM_alpha(g,i):
    g = g.dropna()
    X = g['excess_mkt']
    Y = g[i]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit(cov_type='HAC',cov_kwds={'maxlags':int(len(Y)**0.25)}) 
    return model.params['const']
def CAPM_alpha_t(g):
    g = g.dropna()
    X = g['excess_mkt']
    Y = g[i]
    X = sm.add_constant(X)

    model = sm.OLS(Y, X).fit(cov_type='HAC',cov_kwds={'maxlags':int(len(Y)**0.25)}) 
    return model.tvalues['const']

def CAPM_residual(g):
    g = g.dropna()
    X = g['excess_mkt']
    Y = g[i]
    X = sm.add_constant(X)

    model = sm.OLS(Y, X).fit(cov_type='HAC',cov_kwds={'maxlags':int(len(Y)**0.25)}) 
    return model.resid.std()

g = res5_df[res5_df.Model == 'portfolio_a']
res_col = [ 1, 2, 3, 4, 5, 'long_short']
res_dict = {}
for i in res_col:
    if i == 'long_short':
        j = "LS"
    else:
        j = i
    res_dict[j] = [g[i].mean(), g[i].mean() / g[i].std() * np.sqrt(len(g)), CAPM_alpha(g,i), CAPM_alpha_t(g),g[i].mean()/g[i].std() ,CAPM_alpha(g,i)/CAPM_residual(g)]
res_df = pd.DataFrame(res_dict, index=['$r_i - r_f$', '$t_{stat}$', '$\\alpha_{CAPM}$', '$t_{\\alpha}$', 'Sharpe Ratio', 'IR'])
res_df = res_df.T
res_df.index.name = 'Portfolio'
res_df.reset_index().to_latex(out_path + '6_5_b_a.tex', float_format="%.3f", escape=False,column_format='c' + 'c'*6, index=False)

g = res5_df[res5_df.Model == 'portfolio_b']
res_dict = {}
for i in res_col:
    if i == 'long_short':
        j = "LS"
    else:
        j = i
    res_dict[j] = [g[i].mean(), g[i].mean() / g[i].std() * np.sqrt(len(g)), CAPM_alpha(g,i), CAPM_alpha_t(g),g[i].mean()/g[i].std() ,CAPM_alpha(g,i)/CAPM_residual(g)]
res_df = pd.DataFrame(res_dict, index=['$r_i - r_f$', '$t_{stat}$', '$\\alpha_{CAPM}$', '$t_{\\alpha}$', 'Sharpe Ratio', 'IR'])
res_df = res_df.T
res_df.index.name = 'Portfolio'
res_df.reset_index().to_latex(out_path + '6_5_b_b.tex', float_format="%.3f", escape=False,column_format='c' + 'c'*6,index=False)

g = res5_df[res5_df.Model == 'portfolio_c']
res_dict = {}
for i in res_col:
    if i == 'long_short':
        j = "LS"
    else:
        j = i
    res_dict[j] = [g[i].mean(), g[i].mean() / g[i].std() * np.sqrt(len(g)), CAPM_alpha(g,i), CAPM_alpha_t(g),g[i].mean()/g[i].std() ,CAPM_alpha(g,i)/CAPM_residual(g)]
res_df = pd.DataFrame(res_dict, index=['$r_i - r_f$', '$t_{stat}$', '$\\alpha_{CAPM}$', '$t_{\\alpha}$', 'Sharpe Ratio', 'IR'])
res_df = res_df.T
res_df.index.name = 'Portfolio'
res_df.reset_index().to_latex(out_path + '6_5_b_c.tex', float_format="%.3f", escape=False,column_format='c' + 'c'*6,index=False)



# %%
