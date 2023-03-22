from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
from scipy import stats
import numpy as np
import sys
sys.path.append('../')
from Plot_parameters import colors,medianprops_diff,medianprops_normal,meanprops_normal,SMALL_SIZE,MEDIUM_SIZE,BIGGER_SIZE,X_AXIS_SIZE,Y_AXIS_SIZE,LEGEND_SIZE,ULTRA_SIZE,SCATTER_SMALL,SCATTER_BIG,LINE_WIDTH,MARKER_SIZE_SMALL,MARKER_SIZE_BIG,LINE_WIDTH_SMALL


#Load datasets and set the index.
Results_JAD=pd.read_csv('MAR_Holdout_Results.csv',sep=';',na_values='?')
res_complete = pd.read_csv('Complete-Results.csv',sep=';',na_values='?')
#Risky
res=Results_JAD
"""
res=res.set_index('Dataset')
res.drop(['MAR_10_image.csv','MAR_25_image.csv','MAR_50_image.csv'],inplace=True)
res.reset_index(inplace=True)"""

print(res.head())

def x(a,b):
    return a - b

"""
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','black','red','yellow']
medianprops_normal = dict(linestyle='-.', linewidth=2.5,color='white')
medianprops_diff = dict(linestyle='-.', linewidth=2.5,color='black')


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 18
Y_AXIS_SIZE = 24
ULTRA_SIZE = 28
"""

plt.rc('font', size=ULTRA_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=ULTRA_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=Y_AXIS_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
"""
#Without feature selection.
no_feature_selection_res = pd.DataFrame()
no_feature_selection_res = pd.DataFrame(res['Dataset'])


no_feature_selection_res = pd.concat([no_feature_selection_res,res['nofs']],axis=1)
for i in res.columns:
    if 'no_indicators_and_No_Feature_Selection' == i or 'Indicator_Variables_and_No_Feature_Selection' == i:
        continue
    if 'No_Feature_Selection' in i:
        no_feature_selection_res = pd.concat([no_feature_selection_res,res[i]],axis=1)

res_no_Ft=pd.DataFrame(no_feature_selection_res)

res_no_Ft.columns=['Dataset','Best','GAIN','MF','MM','PPCA','SOFT','DAE','BI+MM','BI+MF','BI+PPCA','BI+GAIN','BI+SOFT','BI+DAE']
res_no_Ft.dropna(axis=1,how='all',inplace=True)


colors_best = ['Lime','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','black','red','yellow']"""

"""
row_id=list(res_no_Ft['Dataset'])
list_10 =  [x for x in [row_id.index(i) if '10' in i else np.nan for i in  row_id] if ~np.isnan(x)]
list_25 =  [x for x in [row_id.index(i) if '25' in i else np.nan for i in  row_id] if ~np.isnan(x)]
list_50 =  [x for x in [row_id.index(i) if '50' in i else np.nan for i in  row_id] if ~np.isnan(x)]
"""

"""res_no_Ft_tradeoff = res_no_Ft
res_no_Ft_tradeoff.index = res_no_Ft_tradeoff['Dataset']
res_no_Ft_tradeoff.drop('Dataset',axis=1 , inplace=True)"""
res_complete.index = res_complete['Dataset']
res_complete.drop('Dataset',axis=1 , inplace=True)



"""
for ind in res_no_Ft_tradeoff.index:
    for indx in res_complete.index:
        if indx in ind:
            res_no_Ft_tradeoff.loc[ind] = res_no_Ft_tradeoff.loc[ind].subtract(res_complete.loc[indx]['nofs'])
            break

print(res_no_Ft_tradeoff)

res_no_Ft_tradeoff.to_csv('MAR-From-Baseline-No-FS.csv')"""

"""
fig, ax = plt.subplots()
plt.title('AUC difference from complete dataset in MAR simulation when feature selection is excluded')
boxplots = list()
j=0
i=0
res_no_Ft_tradeoff.reset_index(inplace=True)
res_no_Ft = res_no_Ft_tradeoff[['Best','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA','DAE','BI+DAE']]
for x in res_no_Ft.columns:
    data =  list([np.array(res_no_Ft.loc[list_10][x].dropna(axis=0)),np.array(res_no_Ft.loc[list_25][x].dropna(axis=0)),np.array(res_no_Ft.loc[list_50][x].dropna(axis=0))])
    boxplots.append(ax.boxplot(data,patch_artist=True,positions=[0.5+j,2.7+j,4.9+j],widths = 0.15,boxprops=dict(facecolor=colors_best[i]), medianprops=medianprops_normal))
    j+= 0.16
    i+=1
plt.xlim(0,9)
locs, labels = plt.xticks()
plt.xticks([1.5, 3.7, 5.9],['10%','25%','50%'])
plt.ylabel('AUC_Difference = AUC of each imputor - AUC of Complete ')
ax.legend([boxplots[i]['boxes'][0] for i in range(len(boxplots))],[i for i in res_no_Ft.columns], loc='lower right')
plt.rcParams["savefig.bbox"] = "tight"
#plt.show()
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig('plots/MAR_Difference_From_baseline_without_fs.png',bbox_inches='tight')
plt.close()




#With feature selection.
feature_selection_res = pd.DataFrame()
feature_selection_res = pd.DataFrame(res['Dataset'])
feature_selection_res = pd.concat([feature_selection_res,res['fs']],axis=1)
for i in res.columns:
    if 'no_indicators_and_Feature_Selection' == i or 'Indicator_Variables_and_Feature_Selection' == i:
        continue
    if 'and_Feature_Selection' in i:
        feature_selection_res = pd.concat([feature_selection_res,res[i]],axis=1)
res_Ft=pd.DataFrame(feature_selection_res)
print(res_Ft.columns)
res_Ft.columns=['Dataset','Best','GAIN','MM','MF','SOFT','PPCA','DAE','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF','BI+DAE']
res_Ft.dropna(axis=1,how='all',inplace=True)"""



"""

colors_best = ['Lime','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','black','red','yellow']


row_id=list(res_Ft['Dataset'])
list_10 =  [x for x in [row_id.index(i) if '10' in i else np.nan for i in  row_id] if ~np.isnan(x)]
list_25 =  [x for x in [row_id.index(i) if '25' in i else np.nan for i in  row_id] if ~np.isnan(x)]
list_50 =  [x for x in [row_id.index(i) if '50' in i else np.nan for i in  row_id] if ~np.isnan(x)]


res_Ft_tradeoff = res_Ft
res_Ft_tradeoff.index = res_Ft_tradeoff['Dataset']
res_Ft_tradeoff.drop('Dataset',axis=1 , inplace=True)


for ind in res_Ft_tradeoff.index:
    for indx in res_complete.index:
        if indx in ind:
            res_Ft_tradeoff.loc[ind] = res_Ft_tradeoff.loc[ind].subtract(res_complete.loc[indx]['fs'])
            break
"""

"""
fig, ax = plt.subplots()
plt.title('AUC difference from complete dataset in MAR simulation when feature selection is enforced')
boxplots = list()
j=0
i=0
#['MM','BI+MM','GAIN','BI+GAIN','MF','BI+MF','SOFT','BI+SOFT','PPCA','BI+PPCA']
res_Ft_tradeoff.reset_index(inplace=True)
res_Ft = res_Ft_tradeoff[['Best','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA','DAE','BI+DAE']]

for x in res_Ft.columns:
    data =  list([np.array(res_Ft.loc[list_10][x].dropna(axis=0)),np.array(res_Ft.loc[list_25][x].dropna(axis=0)),np.array(res_Ft.loc[list_50][x].dropna(axis=0))])
    boxplots.append(ax.boxplot(data,patch_artist=True,positions=[0.5+j,2.7+j,4.9+j],widths = 0.15,boxprops=dict(facecolor=colors_best[i]), medianprops=medianprops_normal))
    j+= 0.16
    i+=1
plt.xlim(0,9)
locs, labels = plt.xticks()
plt.xticks([1.5, 3.7, 5.9],['10%','25%','50%'])
plt.ylabel('AUC_Difference = AUC of each imputor - AUC of Complete ')
ax.legend([boxplots[i]['boxes'][0] for i in range(len(boxplots))],[i for i in res_Ft.columns], loc='lower right')
plt.rcParams["savefig.bbox"] = "tight"
#plt.show()
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig('plots/MAR_Difference_From_baseline_with_fs.png',bbox_inches='tight')
plt.close()

"""

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
best_overall_res = pd.DataFrame(res['Dataset'])
best_overall_res = pd.concat([best_overall_res,res['best']],axis=1)
for i in res.columns:
    if i in columns_best_overall:
        best_overall_res = pd.concat([best_overall_res,res[i]],axis=1)



best_overall=pd.DataFrame(best_overall_res)

best_overall.columns=['Dataset','Best','GAIN','MM','MF','SOFT','PPCA','DAE','BI+GAIN','BI+MM','BI+MF','BI+SOFT','BI+PPCA','BI+DAE']
best_overall.dropna(axis=1,how='all',inplace=True)



colors_best = ['Lime','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','black','red','yellow']


row_id=list(best_overall['Dataset'])
list_10 =  [x for x in [row_id.index(i) if '10' in i else np.nan for i in  row_id] if ~np.isnan(x)]
list_25 =  [x for x in [row_id.index(i) if '25' in i else np.nan for i in  row_id] if ~np.isnan(x)]
list_50 =  [x for x in [row_id.index(i) if '50' in i else np.nan for i in  row_id] if ~np.isnan(x)]


res_overall_tradeoff = best_overall
res_overall_tradeoff.index = res_overall_tradeoff['Dataset']
res_overall_tradeoff.drop('Dataset',axis=1 , inplace=True)


def color_symbol_marker_column_match(column_name):
    line_style = ''
    color = ''
    marker = ''
    if 'Best' in column_name  :
        line_style = 'dashed'
        color = 'black'
        marker = '*'
    elif  'MM' in column_name :
        line_style = 'solid'
        color = 'green'
        marker='o'
    elif 'MF' in column_name  :
        line_style = 'solid'
        color = 'red'
        marker = '^'
    elif  'GAIN' in column_name  :
        line_style = 'dashed'
        color = 'blue'
        marker = 'X'
    elif  'DAE' in column_name :
        line_style = 'solid'
        color = 'orange'
        marker = 'd'
    elif 'SOFT' in column_name  :
        line_style = 'dashed'
        color = 'grey'
        marker = '+'
    elif 'PPCA' in column_name  :
        line_style = 'solid'
        color = 'purple'
        marker = 's'
    return (line_style,color,marker)

for ind in res_overall_tradeoff.index:
    for indx in res_complete.index:
        if indx in ind:
            res_overall_tradeoff.loc[ind] = res_overall_tradeoff.loc[ind].subtract(res_complete.loc[indx]['best'])
            break

plt.title('Average AUC difference from complete dataset in MAR simulation')
#['MM','BI+MM','GAIN','BI+GAIN','MF','BI+MF','SOFT','BI+SOFT','PPCA','BI+PPCA']
res_overall_tradeoff.reset_index(inplace=True)
#best_overall = res_overall_tradeoff[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA','DAE','BI+DAE']]
best_overall = res_overall_tradeoff[['BI+MM','BI+MF','BI+GAIN','BI+SOFT','BI+PPCA','BI+DAE']] #['BI+MF','BI+SOFT','BI+MM','BI+PPCA','BI+DAE','BI+GAIN']
legend_elements = []
auc_avg = []
for i in best_overall.columns:
    dt = [round(np.mean(best_overall.loc[list_10][i]),3),round(np.mean(best_overall.loc[list_25][i]),3),round(np.mean(best_overall.loc[list_50][i]),3)]
    auc_avg.append(dt)
    line_style,color,marker=color_symbol_marker_column_match(i)
    plt.scatter([0,1,2],dt,c=color,marker=marker,s=SCATTER_SMALL,zorder=2)
    plt.plot(dt,linestyle=line_style,color = color,linewidth = LINE_WIDTH_SMALL,zorder=1)
    legend_elements.append(Line2D([0], [0], marker=marker, color=color, label=i,markerfacecolor=color, markersize=MARKER_SIZE_SMALL))
plt.ylabel('Average AUC difference from complete')
plt.xticks([0,1,2],['10%','25%','50%'])
plt.xlabel('Amount of missingness')
plt.legend(handles = legend_elements,loc = 'lower left')    #,prop={'size': 15}

plt.rcParams["savefig.bbox"] = "tight"
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig('plots/MAR_Difference_From_baseline_with_and_without_fs-lineplot-NoIndi.png',bbox_inches='tight')
plt.close()



avg_features = pd.DataFrame(auc_avg,columns=['10%','25%','50%'],index=best_overall.columns)
print(avg_features)
avg_features.to_csv('MAR_Avg_auc.csv',sep=';')






