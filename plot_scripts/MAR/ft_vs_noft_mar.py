import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
from scipy import stats
import numpy as np
import sys
sys.path.append('../')
from Plot_parameters import colors,medianprops_diff,medianprops_normal,meanprops_normal,SMALL_SIZE,MEDIUM_SIZE,BIGGER_SIZE,X_AXIS_SIZE,Y_AXIS_SIZE,LINE_WIDTH_SMALL,LEGEND_SIZE,ULTRA_SIZE,SCATTER_SMALL,SCATTER_BIG,LINE_WIDTH,MARKER_SIZE_SMALL,MARKER_SIZE_BIG
from matplotlib.lines import Line2D


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
        line_style = 'dotted'
        color = 'purple'
        marker = 's'
    return (line_style,color,marker)


colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','black','red','yellow']
medianprops_normal = dict(linestyle='-.', linewidth=2.5,color='white')
medianprops_diff = dict(linestyle='-.', linewidth=2.5,color='black')
def x(a,b):
    return a - b

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 20
Y_AXIS_SIZE = 24
ULTRA_SIZE = 28


plt.rc('font', size=ULTRA_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=ULTRA_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=Y_AXIS_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

res=pd.read_csv('MAR_Holdout_Results.csv',sep=';',na_values='?')

print(res)

#Without feature selection.
"""no_feature_selection_res = pd.DataFrame()
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
res_no_Ft = res_no_Ft[['MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA','DAE','BI+DAE']]
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

best_overall_res = pd.DataFrame(res['Dataset'])
best_overall_res = pd.concat([best_overall_res,res['best']],axis=1)
for i in res.columns:
    if i in columns_best_overall:
        best_overall_res = pd.concat([best_overall_res,res[i]],axis=1)



best_overall=pd.DataFrame(best_overall_res)
best_overall.columns=['Dataset','Best','GAIN','MM','MF','SOFT','PPCA','DAE','BI+GAIN','BI+MM','BI+MF','BI+SOFT','BI+PPCA','BI+DAE']
best_overall.dropna(axis=1,how='all',inplace=True)
res_no_Ft = best_overall

feature_selection_res = pd.DataFrame()
feature_selection_res = pd.DataFrame(res['Dataset'])
feature_selection_res = pd.concat([feature_selection_res,res['fs']],axis=1)
for i in res.columns:
    if 'no_indicators_and_Feature_Selection' == i or 'Indicator_Variables_and_Feature_Selection' == i:
        continue
    if 'and_Feature_Selection' in i:
        feature_selection_res = pd.concat([feature_selection_res,res[i]],axis=1)
res_Ft=pd.DataFrame(feature_selection_res)
res_Ft.columns=['Dataset','Best','GAIN','MM','MF','SOFT','PPCA','DAE','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF','BI+DAE']
res_Ft.dropna(axis=1,how='all',inplace=True)

row_id=list(res_Ft['Dataset'])
list_10 =  [x for x in [row_id.index(i) if '10' in i else np.nan for i in  row_id] if ~np.isnan(x)]
list_25 =  [x for x in [row_id.index(i) if '25' in i else np.nan for i in  row_id] if ~np.isnan(x)]
list_50 =  [x for x in [row_id.index(i) if '50' in i else np.nan for i in  row_id] if ~np.isnan(x)]

res_Ft = res_Ft[['MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA','DAE','BI+DAE']]


print(res_Ft['BI+PPCA'].loc[list_10])
print(res_no_Ft['BI+PPCA'].loc[list_10])

for i in res_Ft.columns:
    res_Ft[i]=res_Ft[i].subtract(res_no_Ft[i],axis=0)


selected_features = ['BI+MM','BI+MF','BI+GAIN','BI+SOFT','BI+PPCA','BI+DAE']
res_Ft = res_Ft[selected_features]

plt.title('MAR : FS vs Best Overall AUC Difference')

legend_elements=list()
avg_ft = list()

for i in res_Ft.columns:
    dt = [round(np.nanmean(res_Ft.loc[list_10][i]),3),round(np.nanmean(res_Ft.loc[list_25][i]),3),round(np.nanmean(res_Ft.loc[list_50][i]),3)]
    avg_ft.append(dt)
    line_style,color,marker=color_symbol_marker_column_match(i)
    plt.scatter([0,1,2],dt,c=color,marker=marker,s=SCATTER_SMALL,zorder=2)
    plt.plot(dt,linestyle=line_style,color = color,linewidth=LINE_WIDTH_SMALL,zorder=1)
    legend_elements.append(Line2D([0], [0], marker=marker, color=color, label=i,markerfacecolor=color, markersize=MARKER_SIZE_SMALL))
    plt.xticks([0,1,2],['10%','25%','50%'])
plt.legend(handles = legend_elements,loc = 'lower left')    #,prop={'size': 15}

avg_features = pd.DataFrame(avg_ft,columns=['10%','25%','50%'],index=res_Ft.columns)
print(avg_features)
plt.xlim(-0.2,2.2)
locs, labels = plt.xticks()
plt.xlabel('Amount of missingness')
print(locs, labels)
plt.ylabel('Average AUC Difference Score (FS - NO FS)')
#plt.tight_layout()
#plt.rcParams["savefig.bbox"] = "tight"
plt.show()

"""mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig('plots/MCAR_FS_VS_NOFS.png',bbox_inches='tight')
#plt.close()"""
