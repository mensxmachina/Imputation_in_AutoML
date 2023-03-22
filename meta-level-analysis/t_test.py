from re import L
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np


"""data = np.genfromtxt("TT.csv", delimiter=';')
print(data)

y  = data[:, -1]

names = []
for i in range(43):
    names.append(input())

for i in range(len(data[0])-1):
    r, p = spearmanr(data[:, i], y)
    if(p < 0.05):
        if(i<43):
            print(names[i], i, p, r)
        else:
            print('CORR', i, p, r)
"""

#Read Meta Features
data = pd.read_csv('DatasetDetails-Extended2.csv',sep=';')
#data = pd.read_csv('DatasetDetails-Extended.csv',sep=';')
data.set_index('Dataset',inplace=True)
data.dropna(how='any',axis=1)
#print(data)

#Read Results
data2 = pd.read_csv('Final-Real-World-Results.csv',sep=';')
data2.set_index('Dataset',inplace=True)
#print(data2)

#Select the columns of the results we want.
#Best overall
columns_best_overall = [
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

"""best_overall_res = pd.DataFrame()
for i in data2.columns:
    if i in columns_best_overall:
        best_overall_res = pd.concat([best_overall_res,data2[i]],axis=1)"""



best_overall=data2[[
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
       'Indicator_Variables_and_Denoise_autoencoder_imputation']]
best_overall.columns=['GAIN','MM','MF','SOFT','PPCA','DAE','BI+GAIN','BI+MM','BI+MF','BI+SOFT','BI+PPCA','BI+DAE']

best_overall = best_overall[['BI+GAIN','BI+MM','BI+MF','BI+SOFT','BI+PPCA','BI+DAE']]
best_overall.dropna(axis=1,how='all',inplace=True)
best_overall=best_overall.divide(best_overall['BI+MM'],axis=0)
best_overall.drop('BI+MM',axis=1,inplace=True)
meta_feature_correlations = list()
meta_feature_p_vals = list()
current_meta = list(data.columns) #Metafeatures are in the columns


new_cols = []
for str_text in current_meta:
    print(str_text)
    if str_text  == 'Missingness percentage':
        str_text = 'miss perce'
    elif str_text  ==  'Percentage_features_with_missing_values':
        str_text = 'perce_feat_w_miss_val'
    elif str_text  ==  'Percentage_samples_with_missing_values':
        str_text = 'perce_sampl_w_miss_val'
    elif str_text  ==  'Percentage_of_missing_values_per_feature_with_missing>1%':
        str_text = 'perce_miss_val_per_feat'
        
    new_cols.append(str_text)

current_meta = new_cols
index_list = list(best_overall.columns) #Each Imputor is in the index

sign_res = []

for i in best_overall.columns:
    current_correlations = list()
    current_p_vals = list()
    df3 = pd.merge(data, best_overall[i], left_index=True, right_index=True)
    new_data = df3.to_numpy()
    
    y  = df3.to_numpy()[:,-1]
    for idx in range(len(new_data[0])-1):
        r, p = spearmanr(new_data[:, idx], y)
        current_correlations.append(r)
        current_p_vals.append(p)
        if(p < 0.1):
            print(df3.columns[idx], idx, p, r,i)
            sign_res.append([i,df3.columns[idx],np.round(r,3),np.round(p,3)])
    meta_feature_correlations.append(current_correlations)
    meta_feature_p_vals.append(current_p_vals)



pd.DataFrame(meta_feature_correlations,index=index_list,columns = current_meta).to_csv('Correlations2.csv',sep=';')

pd.DataFrame(meta_feature_p_vals,index=index_list,columns = current_meta).to_csv('p-values2.csv',sep=';')

pd.DataFrame(sign_res,columns = ['Method','Meta-Feature','Correlation','p-value']).to_csv('Real-World-Results2.csv',sep=';')


linear_method_list = []
for i in range(len(meta_feature_p_vals)):
    linear_method_list = linear_method_list+current_meta

from statsmodels.stats.multitest import fdrcorrection
import statsmodels.stats.multitest as smm

"""`b`, `bonferroni` : one-step correction
    `s`, `sidak` : one-step correction
    `hs`, `holm-sidak` : step down method using Sidak adjustments
    `h`, `holm` : step-down method using Bonferroni adjustments
    `sh`, `simes-hochberg` : step-up method  (independent)
    `hommel` : closed method based on Simes tests (non-negative)
    `fdr_i`, `fdr_bh` : Benjamini/Hochberg  (non-negative)
    `fdr_n`, `fdr_by` : Benjamini/Yekutieli (negative)
    'fdr_tsbh' : two stage fdr correction (Benjamini/Hochberg)
    'fdr_tsbky' : two stage fdr correction (Benjamini/Krieger/Yekutieli)
    'fdr_gbs' : adaptive step-down fdr correction (Gavrilov, Benjamini, Sarkar)"""

rej, pval_corr = smm.multipletests(np.array(meta_feature_p_vals).flatten(), alpha=0.1, method='fdr_bh')[:2]
#rej, pval_corr = smm.multipletests([0.01,0.05,0.05,0.08,0.1], alpha=0.05, method='fdr_bh')[:2]
res = list(pval_corr)
res.sort()
print(res)
print(len(res))


import pandas as pd
import seaborn as sn



pvals_tbl=pd.DataFrame(meta_feature_p_vals,index=index_list,columns = current_meta)
print(pvals_tbl)
"""corr=pd.DataFrame(meta_feature_correlations,index=index_list,columns = current_meta)
#corr.drop(['freq_class.sd'],axis=1,inplace=True)
corr.dropna(how='any',axis=1,inplace=True)
plt.title('Meta-feature correlation')
sn.heatmap(corr, annot=True)
plt.xlabel('Metafeature')
plt.xticks(rotation = 60) # Rotates X-Axis Ticks by 45-degrees
plt.ylabel('Imputation method')
plt.show()




pvals_tbl=pd.DataFrame(meta_feature_p_vals,index=index_list,columns = current_meta)
#pvals_tbl.drop(['freq_class.sd'],axis=1,inplace=True)
pvals_tbl.dropna(how='any',axis=1,inplace=True)
plt.title('Meta-feature p-values')
sn.heatmap(pvals_tbl, annot=True)
plt.xticks(rotation = 60) # Rotates X-Axis Ticks by 45-degrees
plt.ylabel('Imputation method')
plt.xlabel('Metafeature',)
plt.show()"""