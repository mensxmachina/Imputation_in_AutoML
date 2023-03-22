import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
#from sympy import true
import sys
sys.path.append('../')
from Plot_parameters import colors,medianprops_diff,medianprops_normal,meanprops_normal,SMALL_SIZE,MEDIUM_SIZE,BIGGER_SIZE,colors_best,X_AXIS_SIZE,Y_AXIS_SIZE,LINE_WIDTH_SMALL,LEGEND_SIZE,ULTRA_SIZE,SCATTER_SMALL,SCATTER_BIG,LINE_WIDTH,MARKER_SIZE_SMALL,MARKER_SIZE_BIG


res=pd.read_csv('Final-Real-World-Results.csv',sep=';',na_values='?')


"""#Without feature selection.
no_feature_selection_res = pd.DataFrame()
for i in res.columns:
    if 'no_indicators_and_No_Feature_Selection' == i or 'Indicator_Variables_and_No_Feature_Selection' == i:
        continue
    if 'No_Feature_Selection' in i:
        no_feature_selection_res = pd.concat([no_feature_selection_res,res[i]],axis=1)

res_no_Ft=pd.DataFrame(no_feature_selection_res)
res_no_Ft.columns=['GAIN','MF','MM','PPCA','SOFT','DAE','BI+MM','BI+MF','BI+PPCA','BI+GAIN','BI+SOFT','BI+DAE']
print(res_no_Ft)
result = autorank(res_no_Ft, alpha=0.05, verbose=False) #force_mode='parametric'
print(result)
plot_stats(result,allow_insignificant=True)
plt.title('Ranking of Imputation methods without feature selection')
mng = plt.get_current_fig_manager()
plt.savefig('plots/ranking/real-world-ranking-without-fs.png',bbox_inches='tight')
plt.close()"""

#Without feature selection.
no_feature_selection_res = pd.DataFrame()
for i in res.columns:
    if 'no_indicators_and_No_Feature_Selection' == i or 'Indicator_Variables_and_No_Feature_Selection' == i:
        continue
    if 'No_Feature_Selection' in i:
        no_feature_selection_res = pd.concat([no_feature_selection_res,res[i]],axis=1)

res_no_Ft=pd.DataFrame(no_feature_selection_res)
res_no_Ft.columns=['GAIN','MF','MM','PPCA','SOFT','DAE','BI+MM','BI+MF','BI+PPCA','BI+GAIN','BI+SOFT','BI+DAE']
result = autorank(res_no_Ft[['BI+GAIN','BI+MM','BI+MF','BI+SOFT','BI+PPCA','BI+DAE']], alpha=0.05, verbose=False) #,force_mode='parametric'
print(res_no_Ft[['BI+GAIN','BI+MM','BI+MF','BI+SOFT','BI+PPCA','BI+DAE']])
print(result)
plot_stats(result,allow_insignificant=True)
plt.title('Ranking of Imputation methods without feature selection')
mng = plt.get_current_fig_manager()
plt.savefig('plots/ranking/real-world-ranking-without-fs.png',bbox_inches='tight')
plt.close()



"""#With feature selection.
feature_selection_res = pd.DataFrame()
for i in res.columns:
    if 'no_indicators_and_Feature_Selection' == i or 'Indicator_Variables_and_Feature_Selection' == i:
        continue
    if 'and_Feature_Selection' in i:
        feature_selection_res = pd.concat([feature_selection_res,res[i]],axis=1)
res_Ft=pd.DataFrame(feature_selection_res)
res_Ft.columns=['GAIN','MM','MF','SOFT','PPCA','DAE','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF','BI+DAE']
result = autorank(res_Ft, alpha=0.05, verbose=False) #,force_mode='parametric'
print(result)
plot_stats(result,allow_insignificant=True)
plt.title('Ranking of Imputation methods with feature selection')
mng = plt.get_current_fig_manager()
plt.savefig('plots/ranking/real-world-ranking-without-fs.png',bbox_inches='tight')
plt.close()"""


#With feature selection.
feature_selection_res = pd.DataFrame()
for i in res.columns:
    if 'no_indicators_and_Feature_Selection' == i or 'Indicator_Variables_and_Feature_Selection' == i:
        continue
    if 'and_Feature_Selection' in i:
        feature_selection_res = pd.concat([feature_selection_res,res[i]],axis=1)
res_Ft=pd.DataFrame(feature_selection_res)
res_Ft.columns=['GAIN','MM','MF','SOFT','PPCA','DAE','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF','BI+DAE']
result = autorank(res_Ft[['BI+GAIN','BI+MM','BI+MF','BI+SOFT','BI+PPCA','BI+DAE']], alpha=0.05, verbose=False) #,force_mode='parametric'
print(res_Ft[['BI+GAIN','BI+MM','BI+MF','BI+SOFT','BI+PPCA','BI+DAE']])
print(result)
plot_stats(result,allow_insignificant=True)
plt.title('Ranking of Imputation methods with feature selection')
mng = plt.get_current_fig_manager()
plt.savefig('plots/ranking/real-world-ranking-with-fs-Indicatoronly.png',bbox_inches='tight')
plt.close()


"""#Best overall
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
best_overall_res = pd.DataFrame()
for i in res.columns:
    if i in columns_best_overall:
        best_overall_res = pd.concat([best_overall_res,res[i]],axis=1)
best_overall=pd.DataFrame(best_overall_res)
best_overall.columns=['GAIN','MM','MF','SOFT','PPCA','DAE','BI+GAIN','BI+MM','BI+MF','BI+SOFT','BI+PPCA','BI+DAE']
result = autorank(best_overall, alpha=0.05, verbose=False) #,force_mode='parametric'
print(result)
plot_stats(result,allow_insignificant=True)
plt.title('Ranking of Imputation with and without feature selection')
mng = plt.get_current_fig_manager()
plt.savefig('plots/ranking/real-world-ranking-with-and-without-fs.png',bbox_inches='tight')
plt.close()"""



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
best_overall_res = pd.DataFrame()
for i in res.columns:
    if i in columns_best_overall:
        best_overall_res = pd.concat([best_overall_res,res[i]],axis=1)
best_overall=pd.DataFrame(best_overall_res)
best_overall.columns=['GAIN','MM','MF','SOFT','PPCA','DAE','BI+GAIN','BI+MM','BI+MF','BI+SOFT','BI+PPCA','BI+DAE']
#['GAIN','MM','MF','SOFT','PPCA','DAE']]
best_overall = best_overall[['BI+GAIN','BI+MM','BI+MF','BI+SOFT','BI+PPCA','BI+DAE']]
print(best_overall.mean())
result = autorank(best_overall, alpha=0.05, verbose=False,force_mode='parametric') #,
print(result)
create_report(result)

plot_stats(result,allow_insignificant=True)
#plt.title('Ranking of Imputation')

create_report(result)

mng = plt.get_current_fig_manager()
plt.savefig('plots/ranking/real-world-ranking-indicator-only.png',bbox_inches='tight')
plt.close()