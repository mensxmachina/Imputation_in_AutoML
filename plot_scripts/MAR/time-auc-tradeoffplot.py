import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
from scipy import stats
import numpy as np
import sys
sys.path.append('../')
from Plot_parameters import colors,medianprops_diff,medianprops_normal,meanprops_normal,SMALL_SIZE,MEDIUM_SIZE,SCATTER_MEDIAN,BIGGER_SIZE,colors_best,X_AXIS_SIZE,Y_AXIS_SIZE,LINE_WIDTH_SMALL,LEGEND_SIZE,ULTRA_SIZE,SCATTER_SMALL,SCATTER_BIG,LINE_WIDTH,MARKER_SIZE_SMALL,MARKER_SIZE_BIG


time_df = pd.read_csv('Mar_50-Time-Execution-Results.csv',sep=';',na_values='?',float_precision='high')
Auc_df = pd.read_csv('MAR_Holdout_Results.csv',sep=';',na_values='?',float_precision='high')

time_df.set_index('Dataset',drop=True,inplace=True)
Auc_df.set_index('Dataset',drop=True,inplace=True)
time_df.sort_index(inplace=True)
Auc_df.sort_index(inplace=True)


print(time_df)
print(Auc_df)

"""time_df.drop(['Dataset'],axis=1,inplace=True)
Auc_df.drop(['Dataset'],axis=1,inplace=True)"""



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




#Without feature selection.
#no_feature_selection_res_time = pd.DataFrame()
no_feature_selection_auc = pd.DataFrame()

"""for i in time_df.columns:
    if 'no_indicators_and_No_Feature_Selection' == i or 'Indicator_Variables_and_No_Feature_Selection' == i:
        continue
    if 'No_Feature_Selection' in i:
        no_feature_selection_res_time = pd.concat([no_feature_selection_res_time,time_df[i]],axis=1)"""

for i in Auc_df.columns:
    if 'no_indicators_and_No_Feature_Selection' == i or 'Indicator_Variables_and_No_Feature_Selection' == i:
        continue
    if 'No_Feature_Selection' in i:
        no_feature_selection_auc = pd.concat([no_feature_selection_auc,Auc_df[i]],axis=1)

#res_no_Ft_time=pd.DataFrame(no_feature_selection_res_time)
#res_no_Ft_time.columns=['GAIN','MF','MM','PPCA','SOFT','BI+MM','BI+MF','BI+PPCA','BI+GAIN','BI+SOFT']
res_no_Ft_time = time_df
res_no_Ft_time.dropna(axis=1,how='all',inplace=True)
res_no_Ft_time_ready=res_no_Ft_time.mul(-1,axis=0).add(res_no_Ft_time['MM'],axis=0) 
res_no_Ft_time_ready=res_no_Ft_time_ready.divide(res_no_Ft_time['MM'],axis=0)
res_no_Ft_time_ready.drop('MM',axis=1,inplace=True)
res_no_Ft_time_ready = res_no_Ft_time_ready[['GAIN','MF','PPCA','SOFT','DAE','BI+MM','BI+GAIN','BI+MF','BI+PPCA','BI+SOFT','BI+DAE']]


res_no_Ft_auc=pd.DataFrame(no_feature_selection_auc)
res_no_Ft_auc.columns=['GAIN','MF','MM','PPCA','SOFT','DAE','BI+MM','BI+MF','BI+PPCA','BI+GAIN','BI+SOFT','BI+DAE']
#['GAIN','MF','MM','PPCA','SOFT','BI+MM','BI+MF','BI+PPCA','BI+GAIN','BI+SOFT']
res_no_Ft_auc.dropna(axis=1,how='all',inplace=True)
res_no_Ft_auc_ready=res_no_Ft_auc.subtract(res_no_Ft_auc['MM'],axis=0)
res_no_Ft_auc_ready=res_no_Ft_auc_ready.divide(res_no_Ft_auc['MM'],axis=0)
res_no_Ft_auc_ready.drop('MM',axis=1,inplace=True)
res_no_Ft_auc_ready = res_no_Ft_auc_ready[['GAIN','MF','PPCA','SOFT','DAE','BI+MM','BI+GAIN','BI+MF','BI+PPCA','BI+SOFT','BI+DAE']]


total = 0
better_slower = 0
worse_slower = 0
better_faster = 0
worse_faster =0
for i in res_no_Ft_auc_ready:
    for key,values in res_no_Ft_auc_ready[i].items():
        if values== np.nan and res_no_Ft_time_ready[i].loc[key] == np.nan:
            continue
        if values>0  and res_no_Ft_time_ready[i].loc[key] >0:
            better_faster = better_faster + 1
        elif values>0  and res_no_Ft_time_ready[i].loc[key] <0:
            better_slower = better_slower + 1
        elif values<=0  and res_no_Ft_time_ready[i].loc[key] <0:
            worse_slower = worse_slower + 1
        elif values <= 0 and res_no_Ft_time_ready[i].loc[key] >0:
            worse_faster = worse_faster + 1
        total  = total + 1


print('Total Runs',total)
print('better_faster Runs',better_faster)
print('better_slower Runs',better_slower)
print('worse_faster Runs',worse_faster)
print('worse_slower Runs',worse_slower)


plt.figure()
"""SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 20
ULTRA_SIZE = 30"""

"""plt.rc('font', size=ULTRA_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=ULTRA_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=ULTRA_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title"""



plt.rc('font', size=ULTRA_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=ULTRA_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=ULTRA_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=X_AXIS_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=Y_AXIS_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


plt.xlim(-0.5, 0.4)
plt.ylim(-70000,10000)
plt.axvline(x=0,color = 'black', linestyle = '-')
plt.axhline(y=0, color = 'black', linestyle = '-')




plt.fill_between([-0.5, 0],-70000,0,alpha=0.3, color='#CC0000')  # blue
plt.fill_between([0, 0.4], 0, 10000, alpha=0.3, color='#00CC00')  # yellow
plt.fill_between([-0.5, 0], 0 ,10000, alpha=0.3, color='#C0C0C0')  # orange
plt.fill_between([0, 0.4], -70000,0, alpha=0.3, color='#C0C0C0')  # red


plt.text(-0.35,-35000, 'MM Domination Quadrant')
plt.text(-0.35,-39000,str(worse_slower)+' points')
plt.text(0.1,-35000, str(better_slower)+' points')
plt.text(0.1,3000, str(better_faster)+' points')
plt.text(-0.35,3000, str(worse_faster)+' points')

markers = ['o', 'v', '^', 's', 'D', 'X', '<', '>', 'p','.','1']
for i in res_no_Ft_auc_ready.columns:
    marker_sele=markers[list(res_no_Ft_auc_ready.columns).index(i)]
    plt.scatter(res_no_Ft_auc_ready[i],res_no_Ft_time_ready[i],marker= marker_sele,label=i)
plt.legend()
plt.yticks([-70000,-60000,-50000,-40000,-30000,-20000,-10000,0,10000],['-7x$10^4$','-6x$10^4$','-5x$10^4$','-4x$10^4$','-3x$10^4$','-2x$10^4$','-$10^4$','0','$10^4$'])
plt.ylabel('Relative efficiency = ( ( Time_MM - Time_Imputor)/ Time_MM )')
plt.xlabel('Relative effectiveness = ( ( AUC_Imputor - AUC_MM ) / AUC_MM )')
plt.title('Trade-off plot when feature selection is excluded')
#plt.show()
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
#plt.savefig('plots/Time_Tradeoff_no_ft.png',bbox_inches='tight')
plt.close()












#FEATURE SELECTION 




#CHANGE THIS.
feature_selection_res_time = pd.DataFrame()
feature_selection_auc = pd.DataFrame()

"""for i in time_df.columns:
    if 'no_indicators_and_Feature_Selection' == i or 'Indicator_Variables_and_Feature_Selection' == i:
        continue
    if 'and_Feature_Selection' in i:
        feature_selection_res_time = pd.concat([feature_selection_res_time,time_df[i]],axis=1)"""

for i in Auc_df.columns:
    if 'no_indicators_and_Feature_Selection' == i or 'Indicator_Variables_and_Feature_Selection' == i:
        continue
    if 'and_Feature_Selection' in i:
        feature_selection_auc = pd.concat([feature_selection_auc,Auc_df[i]],axis=1)

#res_Ft_time=pd.DataFrame(feature_selection_res_time)
#res_Ft_time.columns=['GAIN','MM','MF','SOFT','PPCA','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF']
res_Ft_time = time_df
res_Ft_time.dropna(axis=1,how='all',inplace=True)
res_Ft_time_ready=res_Ft_time.mul(-1,axis=0).add(res_Ft_time['MM'],axis=0) 
res_Ft_time_ready=res_Ft_time_ready.divide(res_Ft_time['MM'],axis=0)
res_Ft_time_ready.drop('MM',axis=1,inplace=True)
res_Ft_time_ready = res_Ft_time_ready[['GAIN','MF','PPCA','SOFT','DAE','BI+MM','BI+GAIN','BI+MF','BI+PPCA','BI+SOFT','BI+DAE']]
#print(res_Ft_time_ready)

res_Ft_auc=pd.DataFrame(feature_selection_auc)
res_Ft_auc.columns=['GAIN','MM','MF','SOFT','PPCA','DAE','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF','BI+DAE']
#['GAIN','MM','MF','SOFT','PPCA','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF']
res_Ft_auc.dropna(axis=1,how='all',inplace=True)
res_Ft_auc_ready=res_Ft_auc.subtract(res_Ft_auc['MM'],axis=0)
res_Ft_auc_ready=res_Ft_auc_ready.divide(res_Ft_auc['MM'],axis=0)
res_Ft_auc_ready.drop('MM',axis=1,inplace=True)
res_Ft_auc_ready = res_Ft_auc_ready[['GAIN','MF','PPCA','SOFT','DAE','BI+MM','BI+GAIN','BI+MF','BI+PPCA','BI+SOFT','BI+DAE']]
#print(res_Ft_auc_ready)

total = 0
better_slower = 0
worse_slower = 0
better_faster = 0
worse_faster =0
for i in res_Ft_auc_ready:
    for key,values in res_Ft_auc_ready[i].items():
        if values== np.nan and res_Ft_time_ready[i].loc[key] == np.nan:
            continue
        if values>0  and res_Ft_time_ready[i].loc[key] >0:
            better_faster = better_faster + 1
        elif values>0  and res_Ft_time_ready[i].loc[key] <0:
            better_slower = better_slower + 1
        elif values<=0  and res_Ft_time_ready[i].loc[key] <0:
            worse_slower = worse_slower + 1
        elif values <= 0 and res_Ft_time_ready[i].loc[key] >0:
            worse_faster = worse_faster + 1
        total  = total + 1


print('Total Runs',total)
print('better_faster Runs',better_faster)
print('better_slower Runs',better_slower)
print('worse_faster Runs',worse_faster)
print('worse_slower Runs',worse_slower)


plt.figure()


"""plt.rc('font', size=ULTRA_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=ULTRA_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=ULTRA_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title"""


plt.xlim(-0.5, 0.4)
plt.ylim(-70000.0,10000.0)
plt.axvline(x=0,color = 'black', linestyle = '-')
plt.axhline(y=0, color = 'black', linestyle = '-')




plt.fill_between([-0.5, 0],-70000,0,alpha=0.3, color='#CC0000')  # blue
plt.fill_between([0, 0.4], 0, 10000, alpha=0.3, color='#00CC00')  # yellow
plt.fill_between([-0.5, 0], 0 ,10000, alpha=0.3, color='#C0C0C0')  # orange
plt.fill_between([0, 0.4], -70000,0, alpha=0.3, color='#C0C0C0')  # red


plt.text(-0.35,-35000, 'MM Domination Quadrant', style='italic')
plt.text(-0.35,-39000,str(worse_slower)+' points', style='italic')
plt.text(0.1,-35000, str(better_slower)+' points', style='italic')
plt.text(0.1,3000, str(better_faster)+' points', style='italic')
plt.text(-0.35,3000, str(worse_faster)+' points', style='italic')

markers = ['o', 'v', '^', 's', 'D', 'X', '<', '>', 'p','.','1']
for i in res_Ft_auc_ready.columns:
    marker_sele=markers[list(res_Ft_auc_ready.columns).index(i)]
    plt.scatter(res_Ft_auc_ready[i],res_Ft_time_ready[i],marker= marker_sele,label=i)
plt.legend()
plt.yticks([-70000,-60000,-50000,-40000,-30000,-20000,-10000,0,10000],['-7x$10^4$','-6x$10^4$','-5x$10^4$','-4x$10^4$','-3x$10^4$','-2x$10^4$','-$10^4$','0','$10^4$'])
plt.ylabel('Relative efficiency = ( (Time_MM - Time_Imputor)/ Time_MM )')
plt.xlabel('Relative effectiveness = ( ( AUC_Imputor - AUC_MM ) / AUC_MM )')
plt.title('Trade-off plot when feature selection is enforced')
#plt.show()
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
#plt.savefig('plots/Time_Tradeoff_with_ft.png',bbox_inches='tight')
plt.close()




#with and without feature selection





#CHANGE THIS.
feature_selection_res_time = pd.DataFrame()
feature_selection_auc = pd.DataFrame()

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
for i in Auc_df.columns:
    if i in columns_best_overall:
        best_overall_res = pd.concat([best_overall_res,Auc_df[i]],axis=1)

#res_Ft_time=pd.DataFrame(feature_selection_res_time)
#res_Ft_time.columns=['GAIN','MM','MF','SOFT','PPCA','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF']
res_Overall_time = time_df
#print(res_Overall_time)
res_Overall_time.dropna(axis=1,how='all',inplace=True)
#res_Ft_time_ready=res_Ft_time.mul(-1,axis=0)#.add(res_Ft_time['BI+MM'],axis=0) 
#res_Overall_time=res_Overall_time.subtract(res_Overall_time['BI+MM'],axis=0)
res_Overall_time_ready=res_Overall_time.divide(res_Overall_time['BI+MM'],axis=0)
res_Overall_time_ready.drop('BI+MM',axis=1,inplace=True)
#['GAIN','MF','PPCA','SOFT','DAE']
res_Overall_time_ready = res_Overall_time_ready[['BI+GAIN','BI+MF','BI+PPCA','BI+SOFT','BI+DAE']] #,'BI+MM',
#print(res_Overall_time_ready)


res_noft_and_Ft_auc=pd.DataFrame(best_overall_res)
res_noft_and_Ft_auc.columns=['GAIN','MM','MF','SOFT','PPCA','DAE','BI+GAIN','BI+MM','BI+MF','BI+SOFT','BI+PPCA','BI+DAE']
#['GAIN','MM','MF','SOFT','PPCA','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF']
res_noft_and_Ft_auc.dropna(axis=1,how='all',inplace=True)
#res_noft_and_Ft_auc_ready=res_noft_and_Ft_auc.subtract(res_noft_and_Ft_auc['BI+MM'],axis=0)
res_noft_and_Ft_auc_ready=res_noft_and_Ft_auc.divide(res_noft_and_Ft_auc['BI+MM'],axis=0)
res_noft_and_Ft_auc_ready.drop('BI+MM',axis=1,inplace=True)
#
res_noft_and_Ft_auc_ready = res_noft_and_Ft_auc_ready[['BI+GAIN','BI+MF','BI+SOFT','BI+PPCA','BI+DAE']]#['BI+MF','BI+PPCA','BI+SOFT','BI+DAE','BI+GAIN']] #'BI+MM',
#print(res_noft_and_Ft_auc_ready)

total = 0
better_slower = 0
worse_slower = 0
better_faster = 0
worse_faster =0
tied =0 
for i in res_noft_and_Ft_auc_ready:
    for key,values in res_noft_and_Ft_auc_ready[i].items():
        if values== np.nan and res_Overall_time_ready[i].loc[key] == np.nan:
            continue
        if values>1  and res_Overall_time_ready[i].loc[key] > 1:
            worse_faster = worse_faster + 1
        elif values>1  and res_Overall_time_ready[i].loc[key] < 1:
            worse_slower = worse_slower + 1
        elif values< 1  and res_Overall_time_ready[i].loc[key] > 1:
            better_faster = better_faster + 1
        elif values < 1 and res_Overall_time_ready[i].loc[key] < 1:
            better_slower = better_slower + 1
        elif values == 1 and res_Overall_time_ready[i].loc[key] < 1: #same score, slower tho.
            worse_slower = worse_slower + 1
        elif values == 1 and res_Overall_time_ready[i].loc[key] > 1: #same score but faster.
            better_faster = better_faster+1
        elif values == 1 and res_Overall_time_ready[i].loc[key] == 1: #same score but faster.
            tied = tied+1
        total  = total + 1


print('Total Runs',total)
print('better_faster Runs',better_faster)
print('better_slower Runs',better_slower)
print('worse_faster Runs',worse_faster)
print('worse_slower Runs',worse_slower)
print('tied Runs',tied)

plt.figure()


"""plt.rc('font', size=ULTRA_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=ULTRA_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=ULTRA_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title"""


plt.xlim(0.7, 1.2)
plt.ylim(-10000.0,90000.0)
plt.axvline(x=1,color = 'black', linestyle = '-')
plt.axhline(y=1, color = 'black', linestyle = '-')


plt.fill_between([0.7, 1],1,90000,alpha=0.3, color='#00CC00')  # blue
plt.fill_between([1, 1.2], -10000,1, alpha=0.3, color='#CC0000')  # yellow
plt.fill_between([0.7, 1], -10000,1, alpha=0.3, color='#C0C0C0')  # orange
plt.fill_between([1, 1.2], 1,90000, alpha=0.3, color='#C0C0C0')  # red

plt.text(0.8,45000, 'BI+MM Domination', style='italic')
plt.text(0.8,40000,str(better_faster)+' points', style='italic')
plt.text(1.05,40000, str(worse_faster)+' points', style='italic')
plt.text(1.05,-7000, str(worse_slower)+' points', style='italic')
plt.text(0.8,-7000, str(better_slower)+' points', style='italic')

markers = ['o', 'v', '^', 's', 'D', 'X', '<', '>', 'p','.','1']
for i in res_noft_and_Ft_auc_ready.columns:
    marker_sele=markers[list(res_noft_and_Ft_auc_ready.columns).index(i)]
    linestl,color,smth=color_symbol_marker_column_match(i)
    #print(res_noft_and_Ft_auc_ready[i],res_Overall_time_ready[i])
    print(res_noft_and_Ft_auc_ready[i].mean(skipna=True),res_Overall_time_ready[i].mean(skipna=True),i)
    plt.scatter(res_noft_and_Ft_auc_ready[i].mean(skipna=True),res_Overall_time_ready[i].mean(skipna=True),marker= marker_sele,s=SCATTER_MEDIAN,color=color,zorder=2,edgecolors='black',alpha=1)
    plt.scatter(res_noft_and_Ft_auc_ready[i],res_Overall_time_ready[i],marker= marker_sele,label=i,s=SCATTER_BIG,color=color,alpha = 0.6)
plt.legend(loc='upper right')
plt.yticks([90000,80000,70000,60000,50000,40000,30000,20000,10000,1.0,-10000],['9x$10^4$','8x$10^4$','7x$10^4$','6x$10^4$','5x$10^4$','4x$10^4$','3x$10^4$','2x$10^4$','$10^4$','1.0','-$10^4$'])
plt.ylabel('Efficiency Ratio') #= (Time Imputor - Time BI+MM) / Time BI+MM 
plt.xlabel('Effectiveness Ratio') #= (AUC Imputor - AUC BI+MM) / AUC BI+MM 
plt.title('Trade-off plot between AUC and Training time.')

print(res_noft_and_Ft_auc_ready)
print(res_Overall_time_ready)
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.show()
"""plt.savefig('plots/Time_Tradeoff_with_and_without_ft-BIMMBASELINE.png',bbox_inches='tight')#
plt.close()"""



results= res_noft_and_Ft_auc_ready.values
ctr = 0
for i in range( results.shape[0]):
    if np.any(results[i,:]>1):
        ctr+=1
print('BI+MM is losing in ' + str(ctr) + 'out of ' + str(results.shape[0]) + ' datasets')





"""
#================================================


#res_Ft_time=pd.DataFrame(feature_selection_res_time)
#res_Ft_time.columns=['GAIN','MM','MF','SOFT','PPCA','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF']
res_Overall_time = time_df
res_Overall_time.dropna(axis=1,how='all',inplace=True)
#res_Ft_time_ready=res_Ft_time.mul(-1,axis=0)#.add(res_Ft_time['BI+MM'],axis=0) 
#res_Overall_time=res_Overall_time.subtract(res_Overall_time['BI+MM'],axis=0)
res_Overall_time_ready=res_Overall_time.divide(res_Overall_time['BI+DAE'],axis=0)
res_Overall_time_ready.drop('BI+DAE',axis=1,inplace=True)
#['GAIN','MF','PPCA','SOFT','DAE']
res_Overall_time_ready = res_Overall_time_ready[['BI+GAIN','BI+MF','BI+PPCA','BI+SOFT','BI+MM']] #,'BI+MM',
print(res_Overall_time_ready)


res_noft_and_Ft_auc=pd.DataFrame(best_overall_res)
res_noft_and_Ft_auc.columns=['GAIN','MM','MF','SOFT','PPCA','DAE','BI+GAIN','BI+MM','BI+MF','BI+SOFT','BI+PPCA','BI+DAE']
#['GAIN','MM','MF','SOFT','PPCA','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF']
res_noft_and_Ft_auc.dropna(axis=1,how='all',inplace=True)
#res_noft_and_Ft_auc_ready=res_noft_and_Ft_auc.subtract(res_noft_and_Ft_auc['BI+MM'],axis=0)
res_noft_and_Ft_auc_ready=res_noft_and_Ft_auc.divide(res_noft_and_Ft_auc['BI+DAE'],axis=0)
res_noft_and_Ft_auc_ready.drop('BI+DAE',axis=1,inplace=True)
#
res_noft_and_Ft_auc_ready = res_noft_and_Ft_auc_ready[['BI+GAIN','BI+MF','BI+SOFT','BI+PPCA','BI+MM']]#['BI+MF','BI+PPCA','BI+SOFT','BI+DAE','BI+GAIN']] #'BI+MM',
print(res_noft_and_Ft_auc_ready)

total = 0
better_slower = 0
worse_slower = 0
better_faster = 0
worse_faster =0
tied =0
for i in res_noft_and_Ft_auc_ready:
    for key,values in res_noft_and_Ft_auc_ready[i].items():
        if values== np.nan and res_Overall_time_ready[i].loc[key] == np.nan:
            continue
        if values>1  and res_Overall_time_ready[i].loc[key] > 1:
            worse_faster = worse_faster + 1
        elif values>1  and res_Overall_time_ready[i].loc[key] < 1:
            worse_slower = worse_slower + 1
        elif values<1  and res_Overall_time_ready[i].loc[key] >1:
            better_faster = better_faster + 1
        elif values < 1 and res_Overall_time_ready[i].loc[key] <1:
            better_slower = better_slower + 1
        elif values == 1 and res_Overall_time_ready[i].loc[key] < 1: #same score, slower tho.
            worse_slower = worse_slower + 1
        elif values == 1 and res_Overall_time_ready[i].loc[key] > 1: #same score but faster.
            better_faster = better_faster+1
        elif values == 1 and res_Overall_time_ready[i].loc[key] == 1: #same score but faster.
            tied = tied+1
        total  = total + 1


print('Total Runs',total)
print('better_faster Runs',better_faster)
print('better_slower Runs',better_slower)
print('worse_faster Runs',worse_faster)
print('worse_slower Runs',worse_slower)
print('Tied Runs',tied)

plt.figure()


"""

"""
plt.rc('font', size=ULTRA_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=ULTRA_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=ULTRA_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title"""

"""
plt.xlim(0.7, 1.2)
plt.ylim(-10,60)
plt.axvline(x=1,color = 'black', linestyle = '-')
plt.axhline(y=1, color = 'black', linestyle = '-')




plt.fill_between([0.7, 1],1,60,alpha=0.3, color='#00CC00')  # blue
plt.fill_between([1, 1.2], -10,1, alpha=0.3, color='#CC0000')  # yellow
plt.fill_between([0.7, 1], -10,1, alpha=0.3, color='#C0C0C0')  # orange
plt.fill_between([1, 1.2], 1,60, alpha=0.3, color='#C0C0C0')  # red

plt.text(0.8,35, 'BI+DAE Domination', style='italic')
plt.text(0.8,30,str(better_faster)+' points', style='italic')
plt.text(1.05,30, str(worse_faster)+' points', style='italic')
plt.text(1.05,-6, str(worse_slower)+' points', style='italic')
plt.text(0.8,-6, str(better_slower)+' points', style='italic')

markers = ['o', 'v', '^', 's', 'D', 'X', '<', '>', 'p','.','1']
ctn = 0
for i in res_noft_and_Ft_auc_ready.columns:
    marker_sele=markers[list(res_noft_and_Ft_auc_ready.columns).index(i)]
    linestl,color,smth=color_symbol_marker_column_match(i)
    plt.scatter(res_noft_and_Ft_auc_ready[i].median(skipna=True),res_Overall_time_ready[i].median(skipna=True),marker= marker_sele,s=SCATTER_MEDIAN,color=color,zorder=2,edgecolors='black',alpha=0.8)
    plt.scatter(res_noft_and_Ft_auc_ready[i],res_Overall_time_ready[i],marker= marker_sele,label=i,s=SCATTER_BIG,color=color,zorder=1)

    

plt.legend(loc='upper right')
#plt.yticks([60000,50000,40000,30000,20000,10000,1,-10000],['6x$10^4$','5x$10^4$','4x$10^4$','3x$10^4$','2x$10^4$','$10^4$','1','-$10^4$'])
plt.yticks([60,40,20,1.0,-10])
plt.ylabel('Efficiency Ratio') #= (Time Imputor - Time BI+MM) / Time BI+MM 
plt.xlabel('Effectiveness Ratio') #= (AUC Imputor - AUC BI+MM) / AUC BI+MM 
plt.title('Trade-off plot between AUC and Training time.')
#plt.show()
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.show()
"""
"""
plt.savefig('plots/Time_Tradeoff_with_and_without_ft-BIMMBASELINE.png',bbox_inches='tight')#
plt.close()"""


"""
results= res_noft_and_Ft_auc_ready.values
ctr = 0
for i in range( results.shape[0]):
    if np.any(results[i,:]>1):
        ctr+=1
print('BI+DAE is losing in ' + str(ctr) + 'out of ' + str(results.shape[0]) + ' datasets')"""