import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
from scipy import stats
import numpy as np


#Load datasets and set the index.
res=pd.read_csv('Final-Real-World-Results.csv',sep=';',na_values='?')
time_res = pd.read_csv('Time-Execution-Results.csv',sep=';',na_values='?')

res.set_index('Dataset',drop=True,inplace=True)
time_res.set_index('Dataset',drop=True,inplace=True)


new_idx=[i.replace('preds-50-Test-Features-','') for i in list(res.index)]
res.index = new_idx

new_idx2=[i.replace('real_50\\50-Train-','') for i in list(time_res.index)]
time_res.index = new_idx2
time_res.reindex(new_idx,copy=False)

print(res.head())
print(time_res.head())
#Best overall
columns_best_overall = ['best',
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

best_overall_res = res[columns_best_overall]
best_overall=pd.DataFrame(best_overall_res,index=res.index)
best_overall.columns=['Best','GAIN','MM','MF','SOFT','PPCA','DAE','BI+GAIN','BI+MM','BI+MF','BI+SOFT','BI+PPCA','BI+DAE']
best_overall.dropna(axis=1,how='all',inplace=True)





def number_of_wins(dataframe,time_df,selected_columns):
    new_df=dataframe[selected_columns]
    new_df_time=time_df[selected_columns]
    arr = new_df.values
    arr_time = new_df_time.values
    max_val = arr.max(axis=1)
    count_deadlock = (arr == max_val[:, None]).sum(axis=1)
    count_deadlock_withcounts = (arr == max_val[:, None]).sum(axis=0)

    count_wins_before_ties = [0 for col in range(len(selected_columns))]
    average_diff = [0 for col in range(len(selected_columns))]
    for i in range(len(arr)):
        if count_deadlock[i] == 1:
            count_wins_before_ties[np.argmax(arr[i,:])] = count_wins_before_ties[np.argmax(arr[i,:])] + 1
            pos = np.argmax(arr[i,:])
        else:
            #save position of ties.
            tie_pos = list()
            for j in range(len(arr[i,:])):
                if arr[i,j]  == arr[i,:].max():
                    tie_pos.append(j)
            min_time = 100000000
            min_loc = -1 
            for loc in tie_pos:
                if arr_time[i,loc] < min_time:
                    min_loc = loc
                    min_time = arr_time[i,loc]
            count_wins_before_ties[min_loc]   = count_wins_before_ties[min_loc] + 1
            pos = min_loc
        for locs in range(len(selected_columns)):
            average_diff[locs]+=(max_val[i] -arr[i,locs]) #/max_val[i]
    for locs in range(len(selected_columns)):
        average_diff[locs]/=25
        #average_diff[locs]*=100
        average_diff[locs] = np.round(average_diff[locs],3)
    return count_wins_before_ties,average_diff,count_deadlock_withcounts


# Indicators
columns_wanted =['MM','DAE','GAIN','PPCA','SOFT','MF','BI+DAE','BI+MM','BI+MF','BI+PPCA','BI+SOFT']
n_wins,avg_diff,n_wins_withties=number_of_wins(best_overall,time_res,columns_wanted)
print(n_wins,avg_diff,n_wins_withties)
df = pd.DataFrame(list(zip(n_wins,n_wins_withties,avg_diff)),
               columns =['Number_of_Wins','Number_of_wins_withties', 'Average_Difference'],index=columns_wanted)
print(df)
print(best_overall.mean())


#Without

"""columns_wanted =['MM','DAE','GAIN','PPCA','SOFT','MF']
n_wins,avg_diff,n_wins_withties=number_of_wins(best_overall,time_res,columns_wanted)
print(n_wins,avg_diff,n_wins_withties)
df = pd.DataFrame(list(zip(n_wins,n_wins_withties,avg_diff)),
               columns =['Number_of_Wins','Number_of_wins_withties', 'Average_Difference'],index=columns_wanted)
print(df)
df.to_csv('N_Wins.csv',sep=';')

"""
columns_wanted =['BI+MM','BI+DAE','BI+GAIN','BI+PPCA','BI+SOFT','BI+MF']
print(best_overall[columns_wanted])
n_wins,avg_diff,n_wins_withties=number_of_wins(best_overall,time_res,columns_wanted)
print(n_wins,avg_diff,n_wins_withties)
df = pd.DataFrame(list(zip(n_wins,n_wins_withties,avg_diff)),
               columns =['Number_of_Wins','Number_of_wins_withties', 'Average_Difference'],index=columns_wanted)
print(df)
df.to_csv('N_Wins.csv',sep=';')

"""
columns_wanted =['MM','DAE','GAIN','PPCA','SOFT','MF','BI+MM','BI+DAE','BI+GAIN','BI+PPCA','BI+SOFT','BI+MF']
n_wins,avg_diff,n_wins_withties=number_of_wins(best_overall,time_res,columns_wanted)
print(n_wins,avg_diff,n_wins_withties)
df = pd.DataFrame(list(zip(n_wins,n_wins_withties,avg_diff)),
               columns =['Number_of_Wins','Number_of_wins_withties', 'Average_Difference'],index=columns_wanted)
print(df)
df.to_csv('N_Wins.csv',sep=';')"""