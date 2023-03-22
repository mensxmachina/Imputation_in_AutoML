from matplotlib import artist
import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
from scipy import stats
import numpy as np
import sys
sys.path.append('../')
from Plot_parameters import colors,medianprops_diff,medianprops_normal,meanprops_normal,SMALL_SIZE,MEDIUM_SIZE,BIGGER_SIZE,X_AXIS_SIZE,Y_AXIS_SIZE,LINE_WIDTH_SMALL,LEGEND_SIZE,ULTRA_SIZE,SCATTER_SMALL,SCATTER_BIG,LINE_WIDTH,MARKER_SIZE_SMALL,MARKER_SIZE_BIG

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


print(best_overall)

"""print(best_overall.mean())


fig, (ax1,ax2) = plt.subplots(1,2,sharex=False,sharey=False)
fig.subplots_adjust(hspace=0.5)
fig.suptitle('Indicator performance plot')


ax1.set_ylim(0.87,0.9)
ax2.set_ylim(0.87,0.9)


ax1.plot([best_overall.mean()['MM'],best_overall.mean()['BI+MM']])
ax1.set_xticks(ticks=[0,1])
ax1.set_xticklabels(['MM','BI+MM'])


ax2.plot([best_overall.mean()['NO_BI'],best_overall.mean()['BI']])
ax2.set_xticks(ticks=[0,1])
ax2.set_xticklabels(['NO_BI','BI'])
#ax1.grid(True)
#ax2.grid(True)
#plt.xticks(ticks = [0, 1,2,3],labels = best_overall.columns)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off

plt.rcParams["savefig.bbox"] = "tight"
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig('plots/real-holdout-with_and_without_indicators.png',bbox_inches='tight')
plt.show()
plt.close()"""



######Plot with horizontal lines
################################
plt.title('Indicator performance gain/loss')
list_pairs_imp = [('MM','BI+MM'),('DAE','BI+DAE'),('GAIN','BI+GAIN'),('MF','BI+MF'),('SOFT','BI+SOFT'),('PPCA','BI+PPCA')] #('Best','BI+Best'),
x_counter = 0.1 
x_offset = 0.01
x_ticks = []
for pair in list_pairs_imp:
    without_bi, with_bi  = pair[0],pair[1]
    auc_without_bi , auc_with_bi = best_overall.mean()[without_bi],best_overall.mean()[with_bi]
    if auc_without_bi <= auc_with_bi:
        color = 'tab:green'
        min_auc = auc_without_bi
        max_auc = auc_with_bi
        plt.scatter(x_counter,min_auc,c='black',marker='o',s=SCATTER_SMALL,zorder=2)
        plt.scatter(x_counter,max_auc,c='blue',marker='*',s=SCATTER_BIG,zorder=2)
        pre= '+'
    else :
        color = 'tab:red'
        min_auc = auc_with_bi
        max_auc = auc_without_bi   
        plt.scatter(x_counter,min_auc,c='blue',marker='*',s=SCATTER_BIG,zorder=2)
        plt.scatter(x_counter,max_auc,c='black',marker='o',s=SCATTER_SMALL,zorder=2)
        pre=''
    plt.vlines(x_counter,ymin =min_auc ,ymax=max_auc,color=color,linewidth=LINE_WIDTH,zorder=1)
    plt.annotate(pre+ str( round((auc_with_bi - auc_without_bi),4 )),xy= (x_counter,max_auc+0.0015),fontsize = 25)
    x_ticks.append(x_counter)
    x_counter+=0.1


from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], color='green', lw=15, label='AUC increase using BI'),
                   Line2D([0], [0], color='red', lw=15, label='AUC decrease using BI'),
                   Line2D([0], [0], marker='*', color='white', label='Base+BI avg. AUC',markerfacecolor='blue', markersize=MARKER_SIZE_BIG),
                   Line2D([0], [0], marker='o', color='white', label='Base avg. AUC',markerfacecolor='black', markersize=MARKER_SIZE_SMALL)]
plt.legend(handles = legend_elements,loc = 'lower left')    

plt.xlim(0,0.7) #0.8
plt.ylim(0.835,0.895)
x_tick_labels = list()
for  i  in list_pairs_imp:
    x_tick_labels.append(i[0])

plt.xticks(x_ticks,x_tick_labels)
plt.xlabel('Imputation methods')
plt.ylabel('Average AUC Score')
#plt.show()
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig('plots/real-holdout-with_and_without_indicators-horizontal.png',bbox_inches='tight')
#plt.show()
plt.close()