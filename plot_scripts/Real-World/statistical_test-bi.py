from matplotlib import artist
import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
from scipy import stats
import numpy as np
import sys
sys.path.append('../')
from Plot_parameters import colors,medianprops_diff,medianprops_normal,meanprops_normal,SMALL_SIZE,MEDIUM_SIZE,BIGGER_SIZE,X_AXIS_SIZE,Y_AXIS_SIZE,LINE_WIDTH_SMALL,LEGEND_SIZE,ULTRA_SIZE,SCATTER_SMALL,SCATTER_BIG,LINE_WIDTH,MARKER_SIZE_SMALL,MARKER_SIZE_BIG
from scipy.stats import f_oneway

#Load datasets and set the index.
Results_JAD=pd.read_csv('Final-Real-World-Results.csv',sep=';',na_values='?')
#Results_JAD.set_index('Analysis',inplace=True)
#Exclude those not used in the holdout.
#further_drop = ['jad_vote','jad_colleges_aaup','jad_colleges_usnews','jad_cylinder-bands','jad_audiology'] #,'jad_soybean'
#Results_JAD.drop(further_drop,axis=0,inplace=True)
#Risky
res=Results_JAD

print(res.head())


def x(a,b):
    return a - b

"""colors_best = ['Lime','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','black','red','yellow']
medianprops_normal = dict(linestyle='-.', linewidth=2.5,color='white')
medianprops_diff = dict(linestyle='-.', linewidth=2.5,color='black')


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 20
Y_AXIS_SIZE = 24
ULTRA_SIZE = 28"""


plt.rc('font', size=ULTRA_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=Y_AXIS_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=Y_AXIS_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=Y_AXIS_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#Best overall
#columns_best_overall = ['no_indicators_and_Mean-Mode_Imputation','Indicator_Variables_and_Mean-Mode_Imputation','noindicators','indicators']
columns_best_overall = [
       'no_indicators_and_Gain_Imputer','Indicator_Variables_and_Gain_Imputer',
       'no_indicators_and_Mean-Mode_Imputation','Indicator_Variables_and_Mean-Mode_Imputation',
       'no_indicators_and_Missing_Forest_Imputation','Indicator_Variables_and_Missing_Forest_Imputation',
       'no_indicators_and_Soft_Imputation','Indicator_Variables_and_Soft_Imputation',
       'no_indicators_and_PPCA_Imputation','Indicator_Variables_and_PPCA_Imputation',
       'no_indicators_and_Denoise_autoencoder_imputation','Indicator_Variables_and_Denoise_autoencoder_imputation',
       'noindicators','indicators']

best_overall_res = pd.DataFrame()
for i in columns_best_overall:
    best_overall_res = pd.concat([best_overall_res,res[i]],axis=1)
best_overall=pd.DataFrame(best_overall_res)

columns_best_overall = ['GAIN','BI+GAIN','MM','BI+MM','MF','BI+MF','SOFT','BI+SOFT','PPCA','BI+PPCA','DAE','BI+DAE','Best','BI+Best']
best_overall.columns=columns_best_overall
best_overall.dropna(axis=1,how='all',inplace=True)


#print(best_overall)

################################
plt.title('Indicator performance gain/loss')
#('Best','BI+Best'),
"""list_pairs_imp = [('MM','BI+MM'),('DAE','BI+DAE'),('GAIN','BI+GAIN'),('MF','BI+MF'),('SOFT','BI+SOFT'),('PPCA','BI+PPCA')]
p_vals = []
group_means = []
for pair in list_pairs_imp:
    without_bi, with_bi  = pair[0],pair[1]
    auc_without_bi , auc_with_bi = best_overall[without_bi],best_overall[with_bi]
    group_means.append((pair[0],auc_with_bi-auc_without_bi))
    stat,pval = stats.ttest_rel(auc_without_bi, auc_with_bi,alternative='less')
    print(without_bi , stat, pval)
    stat,pval = stats.ttest_rel(auc_with_bi ,auc_without_bi,alternative='greater')
    print(without_bi , stat, pval)
    p_vals.append(pval)

import statsmodels
rej,q_values=statsmodels.stats.multitest.fdrcorrection(p_vals, alpha=0.05, method='indep', is_sorted=False)
print(rej,q_values)"""


"""#('Best','BI+Best'),
list_pairs_imp = [('MM','BI+MM'),('DAE','BI+DAE'),('GAIN','BI+GAIN'),('MF','BI+MF'),('SOFT','BI+SOFT'),('PPCA','BI+PPCA')]
p_vals = []
group_means = []
for pair in list_pairs_imp:
    without_bi, with_bi  = pair[0],pair[1]
    auc_without_bi , auc_with_bi = best_overall[without_bi],best_overall[with_bi]
    group_means.append((pair[0],auc_with_bi-auc_without_bi))


print(group_means)
performance = [i[1] for i in group_means] 
print(performance)
# MM , DAE , GAIN , MF , SOFT , PPCA
stat_f , p_val = f_oneway( performance[0] ,performance[1] , performance[2] , performance [ 3] , performance[4] )

print (p_val )"""



list_pairs_imp = [('MM','BI+MM'),('DAE','BI+DAE'),('GAIN','BI+GAIN'),('MF','BI+MF'),('SOFT','BI+SOFT'),('PPCA','BI+PPCA')]
p_vals = []
group_means = []
for pair in list_pairs_imp:
    without_bi, with_bi  = pair[0],pair[1]
    auc_without_bi , auc_with_bi = best_overall[without_bi],best_overall[with_bi]
    group_means.append((auc_with_bi,auc_without_bi))


performance_bi = [i[0] for i in group_means] 
performance_nobi = [i[1] for i in group_means] 

# MM , DAE , GAIN , MF , SOFT , PPCA
stat_f , p_val = f_oneway( np.array(performance_bi).T , np.array(performance_nobi ).T)

print (p_val )