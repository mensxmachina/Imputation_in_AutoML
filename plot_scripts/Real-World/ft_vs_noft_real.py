import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
from scipy import stats
import numpy as np
import sys

sys.path.append('../')
from Plot_parameters import colors,medianprops_diff,medianprops_normal,meanprops_normal,SMALL_SIZE,MEDIUM_SIZE,BIGGER_SIZE,X_AXIS_SIZE,Y_AXIS_SIZE,LINE_WIDTH_SMALL,LEGEND_SIZE,ULTRA_SIZE,SCATTER_SMALL,SCATTER_BIG,LINE_WIDTH,MARKER_SIZE_SMALL,MARKER_SIZE_BIG
from matplotlib.lines import Line2D


Results_JAD=pd.read_csv('Final-Real-World-Results.csv',sep=';',na_values='?')

res=Results_JAD

print(res.head())

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
features_used = ['BI+MM','BI+MF','BI+GAIN','BI+SOFT','BI+PPCA','BI+DAE']
res_no_Ft = best_overall[features_used]

"""#Without feature selection.
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
#features_used = ['MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA','DAE','BI+DAE']
features_used = ['BI+MM','BI+MF','BI+GAIN','BI+SOFT','BI+PPCA','BI+DAE']
res_no_Ft = res_no_Ft[features_used]"""



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
res_Ft = res_Ft[features_used]


"""for i in res_Ft.columns:
    res_Ft[i]=res_Ft[i].subtract(res_no_Ft[i],axis=0)
print(res_Ft)"""

#res_Ft.to_csv('FS_vs_No_FS.csv')


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

######Plot with horizontal lines
################################
plt.title('FS vs Best Overall performance gain/loss')
x_counter = 0.1 
x_offset = 0.01
x_ticks = []

for col in features_used:
    auc_without_fs , auc_with_fs = res_no_Ft.mean()[col], res_Ft.mean()[col]
    if auc_without_fs <= auc_with_fs:
        color = 'tab:green'
        min_auc = auc_without_fs
        max_auc = auc_with_fs
        plt.scatter(x_counter,min_auc,c='black',marker='o',s=SCATTER_SMALL,zorder=2)
        plt.scatter(x_counter,max_auc,c='blue',marker='*',s=SCATTER_BIG,zorder=2)
        pre= '+'
    else :
        color = 'tab:red'
        min_auc = auc_with_fs
        max_auc = auc_without_fs   
        plt.scatter(x_counter,min_auc,c='blue',marker='*',s=SCATTER_BIG,zorder=2)
        plt.scatter(x_counter,max_auc,c='black',marker='o',s=SCATTER_SMALL,zorder=2)
        pre=''
    plt.vlines(x_counter,ymin =min_auc ,ymax=max_auc,color=color,linewidth=LINE_WIDTH,zorder=1)
    plt.annotate(pre+ str( round((auc_with_fs - auc_without_fs),4 )),xy= (x_counter,max_auc+0.0015),fontsize = 25)
    x_ticks.append(x_counter)
    x_counter+=0.1


from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], color='green', lw=15, label='AUC increase using FS'),
                   Line2D([0], [0], color='red', lw=15, label='AUC decrease using FS'),
                   Line2D([0], [0], marker='*', color='white', label='NO FS avg. AUC',markerfacecolor='blue', markersize=MARKER_SIZE_BIG),
                   Line2D([0], [0], marker='o', color='white', label='FS avg. AUC',markerfacecolor='black', markersize=MARKER_SIZE_SMALL)]
plt.legend(handles = legend_elements,loc = 'lower left')    

plt.xlim(0,0.7) #0.8
plt.ylim(0.835,0.9)
x_tick_labels = list()
for  i  in features_used:
    x_tick_labels.append(i)

plt.xticks(x_ticks,x_tick_labels)
plt.xlabel('Imputation methods')
plt.ylabel('Average AUC score')

mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
#plt.savefig('plots/fs_vs_nofs.png',bbox_inches='tight')
plt.show()
plt.close()











