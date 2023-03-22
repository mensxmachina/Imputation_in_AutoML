from cmath import nan
from matplotlib import artist
import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
from scipy import stats
import numpy as np
import seaborn as sns
import sys


res=pd.read_csv('MAR_Holdout_Results.csv',sep=';')
row_id=list(res['Dataset'])

list_10 =  [x for x in [row_id.index(i) if '10' in i else np.nan for i in  row_id] if ~np.isnan(x)]
datalist_10 =  [x for x in [i if '10' in i else np.nan for i in  row_id]]
datalist_10 = [x for x in datalist_10 if x == x]


list_25 =  [x for x in [row_id.index(i) if '25' in i else np.nan for i in  row_id] if ~np.isnan(x)]
datalist_25 =  [x for x in [i if '25' in i else np.nan for i in  row_id]]
datalist_25 = [x for x in datalist_25 if x == x]


list_50 =  [x for x in [row_id.index(i) if '50' in i else np.nan for i in  row_id] if ~np.isnan(x)]
datalist_50 =  [x for x in [i if '50' in i else np.nan for i in  row_id]]
datalist_50 = [x for x in datalist_50 if x == x]



#Without feature selection.
no_feature_selection_res = pd.DataFrame()
no_feature_selection_res = pd.concat([no_feature_selection_res,res['nofs']],axis=1)

for i in res.columns:
    if 'no_indicators_and_No_Feature_Selection' == i or 'Indicator_Variables_and_No_Feature_Selection' == i:
        continue
    if 'No_Feature_Selection' in i:
        no_feature_selection_res = pd.concat([no_feature_selection_res,res[i]],axis=1)
res_no_Ft=pd.DataFrame(no_feature_selection_res)

res_no_Ft.columns=['Best','GAIN','MF','MM','PPCA','SOFT','DAE','BI+MM','BI+MF','BI+PPCA','BI+GAIN','BI+SOFT','BI+DAE']
res_no_Ft = res_no_Ft[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA','DAE','BI+DAE']]

res_no_Ft_10 = res_no_Ft.loc[list_10]
res_no_Ft_10.index = datalist_10
res_no_Ft_10.round(3).to_csv('Tables/10_MAR_no_FT.csv',sep=';',index=datalist_10)


res_no_Ft_25 = res_no_Ft.loc[list_25]
res_no_Ft_25.index = datalist_25
res_no_Ft_25.round(3).to_csv('Tables/25_MAR_no_FT.csv',sep=';',index=datalist_25)


res_no_Ft_50 = res_no_Ft.loc[list_50]
res_no_Ft_50.index = datalist_50
res_no_Ft_50.round(3).to_csv('Tables/50_MAR_no_FT.csv',sep=';',index=datalist_50)


feature_selection_res = pd.DataFrame()
feature_selection_res = pd.concat([feature_selection_res,res['fs']],axis=1)
for i in res.columns:
    if 'no_indicators_and_Feature_Selection' == i or 'Indicator_Variables_and_Feature_Selection' == i:
        continue
    if 'and_Feature_Selection' in i:
        feature_selection_res = pd.concat([feature_selection_res,res[i]],axis=1)
res_Ft=pd.DataFrame(feature_selection_res)

res_Ft.columns=['Best','GAIN','MM','MF','SOFT','PPCA','DAE','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF','BI+DAE']
res_Ft = res_Ft[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA','DAE','BI+DAE']]



res_Ft_10 = res_Ft.loc[list_10]
res_Ft_10.index = datalist_10
res_Ft_10.round(3).to_csv('Tables/10_MAR_FT.csv',sep=';',index=datalist_10)


res_Ft_25 = res_Ft.loc[list_25]
res_Ft_25.index = datalist_25
res_Ft_25.round(3).to_csv('Tables/25_MAR_FT.csv',sep=';',index=datalist_25)


res_Ft_50 = res_Ft.loc[list_50]
res_Ft_50.index = datalist_50
res_Ft_50.round(3).to_csv('Tables/50_MAR_FT.csv',sep=';',index=datalist_50)





#Best overall
columns_best_overall = [
        'best',
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
best_overall_res = pd.DataFrame()
for i in res.columns:
    if i in columns_best_overall:
        best_overall_res = pd.concat([best_overall_res,res[i]],axis=1)
best_overall=pd.DataFrame(best_overall_res)
best_overall.columns=['Best','GAIN','MM','MF','SOFT','PPCA','DAE','BI+GAIN','BI+MM','BI+MF','BI+SOFT','BI+PPCA','BI+DAE']
best_overall = best_overall[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA','DAE','BI+DAE']]





res_overall_10 = best_overall.loc[list_10]
res_overall_10.index = datalist_10
res_overall_10.round(3).to_csv('Tables/10_MAR_Overall.csv',sep=';',index=datalist_10)

res_overall_25 = best_overall.loc[list_25]
res_overall_25.index = datalist_25
res_overall_25.round(3).to_csv('Tables/25_MAR_Overall.csv',sep=';',index=datalist_25)


res_overall_50 = best_overall.loc[list_50]
res_overall_50.index = datalist_50
res_overall_50.round(3).to_csv('Tables/50_MAR_Overall.csv',sep=';',index=datalist_50)


