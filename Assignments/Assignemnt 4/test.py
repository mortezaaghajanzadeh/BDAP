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
 'size_grp','ret'] + characteristics
#%%
df = pd.read_csv(data_path + 'gbr.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
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
df = df[columns].copy()
df['f_ret'] = df.groupby('id')['ret'].shift(-1)
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
prediction_model = ['be_me', 'ret_12_1','market_equity']
# a OLS regression (Fama-MacBeth)
from linearmodels import FamaMacBeth
etdata = in_sample_df.set_index(['id','date']).copy()
results = FamaMacBeth(etdata.f_ret, etdata[prediction_model]).fit(cov_type='kernel', kernel='newey-west', bandwidth=len(etdata) ** 0.25)

# Save results as LaTeX output
latex_output = results.summary.tables[1].as_latex_tabular(center = False, column_format = 'l' + 'c'*len(prediction_model), longtable = True, escape = False,bold_rows = False,bold_cols = False)

# Save LaTeX output to a file
with open(out_path + '2_1.tex', 'w') as file:
    file.write(latex_output)
# %%
# b Ridge regression   
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

range_ = np.arange(0, 1e4, 100)
mse_res = [find_mse(lambda_0) for lambda_0 in tqdm(range_)]
plt.plot(range_, mse_res)
# %%