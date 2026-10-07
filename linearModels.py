
# %%
# Testing pystout
import statsmodels.api as sm
import linearmodels as ln
from pystout import pystout
import pandas as pd

df = pd.read_csv('trainData.csv', usecols=['contractValueMw','projectIntensityMw', 'awardGrowth', 'timeDeviationPerc'])

#sm.datasets.webuse()
dta = df[['contractValueMw','projectIntensityMw', 'awardGrowth']]

y = df['timeDeviationPerc']*100
#%%
# =============================================================================
# First three models are from statsmodels
# =============================================================================
X = dta['contractValueMw']
X = sm.add_constant(X)
model1 = sm.OLS(y,X).fit()

#%%
X = dta['projectIntensityMw']
X = sm.add_constant(X)
model2 = sm.OLS(y,X).fit()

#%%
X = dta['awardGrowth']
X = sm.add_constant(X)
model3 = sm.OLS(y,X).fit()
#%%

X = dta[['contractValueMw','projectIntensityMw', 'awardGrowth']]
X = sm.add_constant(X)
model4 = sm.OLS(y,X).fit()
#%%
# =============================================================================
# Print result
# =============================================================================
pystout(models=[model1,model2,model3,model4],
        file='test_table.tex',
        #addnotes=['Here is a little note','And another one'],
        digits=4,
        endog_names=['Custom','Header','Please','Thanks'],
        #varlabels={'const':'Constant','displacement':'Disp','mpg':'MPG'},
        #addrows={'Test':['A','Test','Row','Here','Too']},
        mgroups={'Statsmodels':[1,4]},
        modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}','fvalue':'F-stat'}
        )

# %%
pd.concat([dta,y]).describe().to_latex('consultaLatex.tex', float_format="%.2f")
#pd.concat([dta,y]).describe()

# %%
'''
Este modelo es un modelo de regresión lineal simple, donde se busca \npredecir el porcentaje de desviación de tiempo en los contratos de construcción de proyectos de energía renovable.
'''
# %%
# =============================================================================
# 2. Using linearmodels
# =============================================================================
#%%
# Define model
model = ln.PanelOLS(y, X, entity_effects=True, time_effects=True)
# Fit model
results = model.fit()
# Print summary
print(results)