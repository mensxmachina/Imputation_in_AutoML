from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
from Plot_parameters import colors,medianprops_diff,medianprops_normal,meanprops_normal,SMALL_SIZE,MEDIUM_SIZE,BIGGER_SIZE,X_AXIS_SIZE,Y_AXIS_SIZE,LINE_WIDTH_SMALL,LEGEND_SIZE,ULTRA_SIZE,SCATTER_SMALL,SCATTER_BIG,LINE_WIDTH,MARKER_SIZE_SMALL,MARKER_SIZE_BIG

res_mcar=pd.read_csv('MAR-FT-Results.csv',sep=';',na_values='?')
res_complete = pd.read_csv('Complete-FT-Results.csv',sep=';',na_values='?')

res_mcar=res_mcar.set_index('Analysis')
res_mcar.drop(['MAR_10_image_outcome','MAR_25_image_outcome','MAR_50_image_outcome'],inplace=True)
res_mcar.reset_index(inplace=True)

print(res_mcar)


"""colors = ['Lime','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','black','red','yellow']
medianprops_normal = dict(linestyle='-.', linewidth=2.5,color='red')
medianprops_diff = dict(linestyle='-.', linewidth=2.5,color='black')
meanprops_normal = dict(linestyle='--', linewidth=2.5,color='black')

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 20
Y_AXIS_SIZE = 24
ULTRA_SIZE = 28"""


plt.rc('font', size=ULTRA_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=ULTRA_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=ULTRA_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=X_AXIS_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=Y_AXIS_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#With feature selection.
feature_selection_res = pd.DataFrame(res_mcar['Analysis'])
feature_selection_res = pd.concat([feature_selection_res,res_mcar['fs']],axis=1)
for i in res_mcar.columns:
    if 'no_indicators_and_Feature_Selection' == i or 'Indicator_Variables_and_Feature_Selection' == i:
        continue
    if 'and_Feature_Selection' in i:
        feature_selection_res = pd.concat([feature_selection_res,res_mcar[i]],axis=1)
res_Ft=pd.DataFrame(feature_selection_res)
res_Ft.columns=['Dataset','Best','GAIN','MM','MF','SOFT','PPCA','DAE','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF','BI+DAE']



row_id=list(res_Ft['Dataset'])
list_10 =  [x for x in [row_id.index(i) if '10' in i else np.nan for i in  row_id] if ~np.isnan(x)]
list_25 =  [x for x in [row_id.index(i) if '25' in i else np.nan for i in  row_id] if ~np.isnan(x)]
list_50 =  [x for x in [row_id.index(i) if '50' in i else np.nan for i in  row_id] if ~np.isnan(x)]

print(list_10,list_25,list_50)



fig, ax = plt.subplots()
plt.title('Number of features Selected for MAR data')
#['MM','BI+MM','GAIN','BI+GAIN','MF','BI+MF','SOFT','BI+SOFT','PPCA','BI+PPCA']
res_Ft = res_Ft[['MM','MF','DAE','GAIN','SOFT','PPCA']]#[['Best','MM','MF','DAE','GAIN','SOFT','PPCA']]

def color_symbol_marker_column_match(column_name):
    line_style = ''
    color = ''
    marker = ''
    if column_name == 'Best':
        line_style = 'dashed'
        color = 'black'
        marker = '*'
    elif column_name == 'MM':
        line_style = 'solid'
        color = 'green'
        marker='o'
    elif column_name == 'MF':
        line_style = 'solid'
        color = 'red'
        marker = '^'
    elif column_name == 'GAIN':
        line_style = 'dashed'
        color = 'blue'
        marker = 'X'
    elif column_name == 'DAE':
        line_style = 'solid'
        color = 'orange'
        marker = 'd'
    elif column_name == 'SOFT':
        line_style = 'dashed'
        color = 'grey'
        marker = '+'
    elif column_name == 'PPCA':
        line_style = 'solid'
        color = 'purple'
        marker = 's'
    return (line_style,color,marker)

res_complete=res_complete.set_index('Analysis')
res_complete.drop('50-Train-jad_image_outcome',inplace=True)
legend_elements = []
print(res_Ft)
avg_ft= list()
for i in res_Ft.columns:
    dt = [round(res_complete['fs'].mean(), 2),round(np.mean(res_Ft.loc[list_10][i]),2),round(np.mean(res_Ft.loc[list_25][i]),2),round(np.mean(res_Ft.loc[list_50][i]),2)]
    avg_ft.append(dt)
    line_style,color,marker=color_symbol_marker_column_match(i)
    plt.scatter([0,1,2,3],dt,c=color,marker=marker,s=SCATTER_SMALL,zorder=2)
    plt.plot(dt,linestyle=line_style,color = color,linewidth=LINE_WIDTH_SMALL,zorder=1)
    legend_elements.append(Line2D([0], [0], marker=marker, color=color, label=i,markerfacecolor=color, markersize=MARKER_SIZE_SMALL))
    plt.xticks([0,1,2,3],['0%','10%','25%','50%'])
plt.legend(handles = legend_elements,loc = 'upper left')     #prop={'size': 15}

plt.xlim(0,3)
locs, labels = plt.xticks()
plt.xlabel('Amount of missingness')
print(locs, labels)
plt.ylabel('Average Features Selected')
#plt.legend([i for i in res_Ft.columns],loc='upper left')
plt.tight_layout()
plt.rcParams["savefig.bbox"] = "tight"
#plt.show()
"""plt.ylabel('Number of Features Selected')
plt.tight_layout()
plt.rcParams["savefig.bbox"] = "tight"
#plt.show()
"""
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig('plots/MAR_features_line_selected.png',bbox_inches='tight')
#plt.close()

avg_features = pd.DataFrame(avg_ft,columns=['0%','10%','25%','50%'],index=res_Ft.columns)
print(avg_features)
avg_features.to_csv('MAR_Avg_features.csv',sep=';')