from turtle import st
import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
from scipy import stats
import numpy as np
import sys
sys.path.append('../')
from Plot_parameters import colors,medianprops_diff,medianprops_normal,meanprops_normal,SMALL_SIZE,MEDIUM_SIZE,BIGGER_SIZE,colors_best,X_AXIS_SIZE,Y_AXIS_SIZE,LINE_WIDTH_SMALL,LEGEND_SIZE,ULTRA_SIZE,SCATTER_SMALL,SCATTER_BIG,LINE_WIDTH,MARKER_SIZE_SMALL,MARKER_SIZE_BIG


#Load datasets and set the index.
Results_JAD=pd.read_csv('MCAR_Holdout_Results.csv',sep=';',na_values='?')
Results_JAD.fillna(0,inplace=True)

res=Results_JAD 
def x(a,b):
    return a - b




"""colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','black','red','yellow']
medianprops_normal = dict(linestyle='-.', linewidth=2.5,color='white')
medianprops_diff = dict(linestyle='-.', linewidth=2.5,color='black')


SMALL_SIZE = 8
MEDIUM_SIZE = 15
BIGGER_SIZE = 20
Y_AXIS_SIZE = 24
ULTRA_SIZE = 28
colors_best = ['Lime','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','black','red','yellow']"""


plt.rc('font', size=ULTRA_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=ULTRA_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=ULTRA_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=X_AXIS_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=Y_AXIS_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def color_symbol_marker_column_match(column_name):
    line_style = ''
    color = ''
    marker = ''
    if column_name == 'Best':
        line_style = 'dashed'
        color = 'black'
        marker = '*'
    elif column_name == 'MM' or column_name == 'BI+MM':
        line_style = 'solid'
        color = 'green'
        marker='o'
    elif column_name == 'MF' or column_name == 'BI+MF':
        line_style = 'solid'
        color = 'red'
        marker = '^'
    elif column_name == 'GAIN' or column_name == 'BI+GAIN':
        line_style = 'dashed'
        color = 'blue'
        marker = 'X'
    elif column_name == 'DAE':
        line_style = 'solid'
        color = 'orange'
        marker = 'd'
    elif column_name == 'BI+DAE':
        line_style = 'dashed'
        color = 'cyan'
        marker = '.'
    elif column_name == 'SOFT':
        line_style = 'dashed'
        color = 'grey'
        marker = '+'
    elif column_name == 'PPCA':
        line_style = 'solid'
        color = 'purple'
        marker = 's'
    return (line_style,color,marker)


def configuration_multiplier(list_of_imputors): 
    sum = 0 
    for column_name in list_of_imputors:   
        if column_name == 'MM' or column_name == 'BI+MM':
            sum+= 1
        elif column_name == 'MF' or column_name == 'BI+MF':
            sum+= 2
        elif column_name == 'GAIN' or column_name == 'BI+GAIN':
            sum+= 6
        elif column_name == 'DAE' or column_name == 'BI+DAE':
            sum+= 9
        else:
            sum+= 3
    return sum


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
best_overall_res = pd.concat([best_overall_res,res['best']],axis=1)
for i in res.columns:
    if i in columns_best_overall:
        best_overall_res = pd.concat([best_overall_res,res[i]],axis=1)



best_overall=pd.DataFrame(best_overall_res)
best_overall.columns=['Best','GAIN','MM','MF','SOFT','PPCA','DAE','BI+GAIN','BI+MM','BI+MF','BI+SOFT','BI+PPCA','BI+DAE']
best_overall.dropna(axis=1,how='all',inplace=True)


fig, ax = plt.subplots()
plt.title('Percentage of maximum AUC reached as we add extra imputation methods') 
best_overall = best_overall[['BI+GAIN','BI+MM','BI+MF','BI+SOFT','BI+PPCA','BI+DAE']] # Keep the best 5 overall. +MM
# 'GAIN','MM','MF','SOFT','PPCA','DAE',

def find_best_subset(dataframe_with_existing,dataframe_with_new_methods,existing_cols):
    existing_methods =  dataframe_with_existing.values
    extra_methods = dataframe_with_new_methods.values
    curr_max = 0
    curr_max_idx = -1
    #Only 1 new method found.
    if extra_methods.shape[1] == 1:
        return existing_cols + list(dataframe_with_new_methods.columns)

    for i in range(extra_methods.shape[1]):
        max_testing_mean = np.nanmean(np.max(np.column_stack((existing_methods, extra_methods[:,i])), axis=1) ,axis=0)
        if max_testing_mean > curr_max:
            curr_max = max_testing_mean
            curr_max_idx = i
    #Remove and add to the new dataframe
    col_selected = list(dataframe_with_new_methods.columns)[curr_max_idx]
    
    col_to_add = dataframe_with_new_methods[[col_selected]]
    new_methods=dataframe_with_new_methods.drop(col_selected,axis=1)
    data_existing = dataframe_with_existing.copy()
    data_existing[col_selected]  = col_to_add
    existing_cols.append(col_selected)

    new_list = find_best_subset(data_existing,new_methods,existing_cols)
    return new_list

def create_perce_best_plot(list_of_lists,best_overall_df):

    curr_l = list()
    x_axis = []
    y_axis = []
    for i in list_of_lists:
        curr_l.append(i)
        avg_gain= np.float(np.mean(np.max(best_overall_df[curr_l], axis=1) ,axis=0))
        #add the complexity.
        x_axis.append(configuration_multiplier(curr_l))
        
        y_axis.append(avg_gain)

    
    percentage_of_auc = [np.round(i/y_axis[-1],4)*100 for i in y_axis]
    p_auc_lower_99 = -1
    p_auc_lower_100 = -1
    for p_auc in range(len(percentage_of_auc)):
        if percentage_of_auc[p_auc] >= 99 and p_auc_lower_99==-1:
            p_auc_lower_99 = p_auc
        if percentage_of_auc[p_auc] >= 100 and p_auc_lower_100==-1:
            p_auc_lower_100 = p_auc


    #x_labels=[str(i)+'x' for i in x_axis]
    x_labels=[ str(curr_l[i]) + '(' + str(x_axis[i])+'x)' for i in range(len(x_axis))]
    # creating the bar plot
    
    plt.scatter(x_axis,percentage_of_auc,c='blue',marker='.',s=SCATTER_BIG,zorder=2)

    plt.scatter(x_axis[p_auc_lower_99],percentage_of_auc[p_auc_lower_99],c='orange',marker='p',s=SCATTER_BIG,zorder=3,label='99% of total AUC reached ')
    plt.scatter(x_axis[p_auc_lower_100],percentage_of_auc[p_auc_lower_100],c='red',marker='s',s=SCATTER_BIG,zorder=3,label = '100% of total AUC reached')
    plt.plot(x_axis,percentage_of_auc,linestyle='solid',color='black',linewidth = LINE_WIDTH_SMALL,zorder=1)
    print(percentage_of_auc)
    plt.xticks(x_axis,x_labels,rotation=45)
    plt.ylabel('Percentage of maximum AUC reached')
    plt.xlabel('Configurations')
    plt.legend(loc='lower right')
    #mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()
    plt.subplots_adjust(bottom=0.20)
    #plt.savefig('plots/mcar_best_overall_subset_configs.png',bbox_inches='tight')
    plt.show()






#with and without BI
list_of_lists=find_best_subset(best_overall[['BI+MM']],best_overall.drop('BI+MM',axis=1),['BI+MM'])
create_perce_best_plot(list_of_lists,best_overall)


"""# Keep only BI
best_overall = best_overall[['BI+GAIN','BI+MM','BI+MF','BI+SOFT','BI+PPCA','BI+DAE']] 
list_of_lists=find_best_subset(best_overall[['BI+MM']],best_overall.drop('BI+MM',axis=1),['BI+MM'])
print(list_of_lists)
create_perce_best_plot(list_of_lists,best_overall)"""