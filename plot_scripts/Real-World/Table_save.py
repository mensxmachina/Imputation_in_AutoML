from matplotlib import artist
import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
from scipy import stats
import numpy as np
import seaborn as sns
import sys


Results_JAD=pd.read_csv('Final-Real-World-Results.csv',sep=';',na_values='?')
res=Results_JAD
res.set_index('Dataset',inplace=True)

no_feature_selection_res = pd.DataFrame()
no_feature_selection_res = pd.concat([no_feature_selection_res,res['nofs']],axis=1)
for i in res.columns:
    if 'no_indicators_and_No_Feature_Selection' == i or 'Indicator_Variables_and_No_Feature_Selection' == i:
        continue
    if 'No_Feature_Selection' in i:
        no_feature_selection_res = pd.concat([no_feature_selection_res,res[i]],axis=1)

res_no_Ft=pd.DataFrame(no_feature_selection_res)
res_no_Ft.columns=['Best','GAIN','MF','MM','PPCA','SOFT','DAE','BI+MM','BI+MF','BI+PPCA','BI+GAIN','BI+SOFT','BI+DAE']
res_no_Ft.dropna(axis=1,how='all',inplace=True)
res_no_Ft = res_no_Ft[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA','DAE','BI+DAE']]

#With feature selection.
feature_selection_res = pd.DataFrame()
feature_selection_res = pd.concat([feature_selection_res,res['fs']],axis=1)
for i in res.columns:
    if 'no_indicators_and_Feature_Selection' == i or 'Indicator_Variables_and_Feature_Selection' == i:
        continue
    if 'and_Feature_Selection' in i:
        feature_selection_res = pd.concat([feature_selection_res,res[i]],axis=1)
res_Ft=pd.DataFrame(feature_selection_res)
res_Ft.columns=['Best','GAIN','MM','MF','SOFT','PPCA','DAE','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF','BI+DAE']
res_Ft.dropna(axis=1,how='all',inplace=True)
res_Ft = res_Ft[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA','DAE','BI+DAE']]



columns_best_overall = ['best',
       'no_indicators_and_Gain_Imputer',
       'no_indicators_and_Mean-Mode_Imputation',
       'no_indicators_and_Missing_Forest_Imputation',
       'no_indicators_and_Soft_Imputation',
       'no_indicators_and_PPCA_Imputation',
       'no_indicators_and_Denoise_autoencoder_imputation',
       'Indicator_Variables_and_Gain_Imputer',
       'Indicator_Variables_and_Mean-Mode_Imputation',
       'Indicator_Variables_and_Missing_Forest_Imputation',
       'Indicator_Variables_and_Soft_Imputation',
       'Indicator_Variables_and_PPCA_Imputation',
       'Indicator_Variables_and_Denoise_autoencoder_imputation']

best_overall_res = res[columns_best_overall]
best_overall=pd.DataFrame(best_overall_res,index=res.index)
best_overall.columns=['Best','GAIN','MM','MF','SOFT','PPCA','DAE','BI+GAIN','BI+MM','BI+MF','BI+SOFT','BI+PPCA','BI+DAE']
best_overall.dropna(axis=1,how='all',inplace=True)
best_overall = best_overall[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA','DAE','BI+DAE']]

res_no_Ft.round(3).to_csv('Tables/Real_no_FT.csv',sep=';')
res_Ft.round(3).to_csv('Tables/Real_FT.csv',sep=';')
best_overall.round(3).to_csv('Tables/Real_overall.csv',sep=';')