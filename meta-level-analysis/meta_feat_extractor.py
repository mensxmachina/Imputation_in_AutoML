# Load a dataset
from pymfe.mfe import MFE
from os import listdir
from os.path import isfile, join
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.metrics import silhouette_score
from numpy.core.numeric import NaN
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import KFold
import time
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import accuracy_score,roc_auc_score
import sklearn
#from dae_mix_encoding import DAE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,LabelBinarizer,OneHotEncoder
import glob
import datetime
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from meanmode import mm
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans


pd.set_option('display.max_columns', None)
def dataset_cat_flag(file_name):

    categorical_features = []
    flag = 0
    
    if file_name == '50-Train-jad_analcatdata_reviewer.csv':
        flag = 1
    elif file_name == '50-Train-jad_anneal.csv':
        categorical_features = ['family','product-type','steel','temper_rolling','condition','non-ageing','surface-finish','surface-quality','bc','bf','bt','bw/me','bl','chrom','phos','cbond','exptl','ferro','blue/bright/varn/clean','lustre','shape','oil']
    elif file_name == '50-Train-jad_audiology.csv':
        flag = 1
    elif file_name == '50-Train-jad_autoHorse.csv':
        categorical_features= ['fuel-type', 'aspiration', 'body-style', 'drive-wheels', 'make','engine-location', 'engine-type',  'fuel-system']
    elif file_name == '50-Train-jad_bridges.csv':
        categorical_features= ['RIVER', 'PURPOSE', 'CLEAR-G', 'T-OR-D', 'MATERIAL', 'SPAN', 'REL-L']
    elif file_name == '50-Train-jad_cjs.csv':
        categorical_features= [ 'TREE', 'BR']
    elif file_name == '50-Train-jad_colic.csv':
        categorical_features = ['surgery', 'Age', 'temp_extremities', 'peripheral_pulse', 'mucous_membranes', 'capillary_refill_time', 
        'pain', 'peristalsis', 'abdominal_distension', 'nasogastric_tube', 'nasogastric_reflux', 
        'rectal_examination', 'abdomen', 'abdominocentesis_appearance','outcome']
    elif file_name == '50-Train-jad_colleges_aaup.csv':
        categorical_features = [ 'State', 'Type' ]
    elif file_name == '50-Train-jad_cylinder-bands.csv':
        categorical_features = [
        'cylinder_number', 'customer', 'grain_screened', 'ink_color', 'proof_on_ctd_ink', 
        'blade_mfg', 'cylinder_division', 'paper_type', 'ink_type', 'direct_steam', 'solvent_type', 
        'type_on_cylinder', 'press_type', 'cylinder_size', 'paper_mill_location']
    elif file_name == '50-Train-jad_dresses-sales.csv':
        categorical_features = ['V2', 'V3', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13']
    elif file_name == '50-Train-jad_eucalyptus.csv':
        categorical_features = ['Abbrev', 'Locality', 'Map_Ref', 'Latitude', 'Sp']
    elif file_name == '50-Train-jad_hepatitis.csv':
        categorical_features = ['SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA', 
        'LIVER_BIG', 'LIVER_FIRM', 'SPLEEN_PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'HISTOLOGY']
    elif file_name == '50-Train-jad_mushroom.csv':
        flag = 1
    elif file_name == '50-Train-jad_pbcseq.csv':
        categorical_features = ['drug', 'sex', 'presence_of_asictes', 'presence_of_hepatomegaly', 
        'presence_of_spiders']
    elif file_name == '50-Train-jad_primary-tumor.csv':
        flag = 1
    elif file_name == '50-Train-jad_profb.csv':
        categorical_features = ['Favorite_Name', 'Underdog_name' ,'Weekday', 'Overtime']
    elif file_name == '50-Train-jad_schizo.csv':
        categorical_features = ['target', 'sex' ]
    elif file_name == '50-Train-jad_sick.csv':
        categorical_features = ['sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured','referral_source']
    elif file_name == '50-Train-jad_soybean.csv':
        flag = 1
    elif file_name == '50-Train-jad_stress.csv':
        categorical_features = ['Sexe', 'Consommation_tabac', 'type_consommation', 'Allergies']
    elif file_name == '50-Train-jad_vote.csv':
        flag = 1
    elif file_name == '50-Train-jad_hungarian.csv':
        categorical_features = ['sex' ,'fbs','exang']
    elif file_name == '50-Train-jad_braziltourism.csv':
        categorical_features = ['Sex', 'Access_road']
    else:
        categorical_features = []
    
    return (categorical_features,flag)

def encode_data(data,vmaps,flag):
    #preprocess the data 
    column_names = list(data.columns)
    
    
    if flag == 1:
        tr= OrdinalEncoder(unknown_value=np.nan,handle_unknown="use_encoded_value")
        X_train = tr.fit_transform(data)    
        # fit the data
        imputer = mm(parameters={},names=column_names,vmaps=vmaps)
        imputed_data = imputer.fit_transform(X_train)
        tr= OneHotEncoder(handle_unknown="ignore",sparse=False)
        Imputed_Train = tr.fit_transform(imputed_data)
    else:
        sclr = ColumnTransformer(
                transformers=[
                    ("std", 
                    StandardScaler(), 
                    [column_names.index(i) for i in column_names if i not in vmaps.keys()]),
                    ("ordi",
                    OrdinalEncoder(unknown_value=np.nan,handle_unknown="use_encoded_value"),
                    [column_names.index(i) for i in column_names if i in vmaps.keys()])
                    ],
                    remainder = 'passthrough'
            )
        column_names2 = [i for i in column_names if i not in vmaps.keys()] + [i for i in column_names if i in vmaps.keys()] 
        column_names = column_names2
        
        scaled_data = sclr.fit_transform(data)
        scaled_df=pd.DataFrame(scaled_data,columns=column_names)
        for col in scaled_df.columns:
            if len(scaled_df[col].unique()) == 1:
                scaled_df.drop(col,inplace=True,axis=1)
    
        column_names= list(scaled_df.columns)
        imputer = mm(parameters={},names=column_names,vmaps=vmaps)
        imputed_data = imputer.fit_transform(scaled_df.to_numpy())
        sclr = ColumnTransformer(
                    transformers=[
                        ("std", 
                        StandardScaler(), 
                        [column_names.index(i) for i in column_names if i not in vmaps.keys()]),
                        ("ordi",
                        OneHotEncoder(handle_unknown="ignore",sparse=False),
                        [column_names.index(i) for i in column_names if i in vmaps.keys()])
                        ],
                        remainder = 'passthrough'
                )
        Imputed_Train = sclr.fit_transform(imputed_data)
        
    return Imputed_Train


import csv
with open('DatasetDetails-Extended2.csv', 'w', newline='') as csvfile:
    index = 0
    dataset_path = "real_50/"
    dataset_names = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]
    for i in range(len(dataset_names)):
        if dataset_names[i] == '50-Test-jad_colleges_usnews.csv':
            continue
        data = pd.read_csv(dataset_path+dataset_names[i], delimiter=",")
        y = data["binaryClass"].to_numpy()
        del data["binaryClass"]
        X = data.to_numpy()
        
        # Extract default measures
        #mfe = MFE(groups=["general"]) #"general", "statistical", "info-theory"
        mfe = MFE(groups=["general"],features=['inst_to_attr','nr_attr','nr_inst'])
        mfe.fit(X, y)
        ft = mfe.extract()

        #Manual features
        data = pd.read_csv(dataset_path+dataset_names[i], delimiter=",",na_values=['?',np.nan])
        data.drop(['binaryClass'],axis=1,inplace=True)
        X =pd.DataFrame(data)
        
        #Imbalance ratio
        imbalance_ratio = min(Counter(y).values())*1.0/len(y)

        #Percentage of missing values per feature.
        percentage_of_missing_per_feature = X.isna().mean(axis=0).round(4)*100

        #Percentage of missing values per sample
        percentage_of_missing_per_sample = X.isna().mean(axis=1).round(4)*100

        #Overall percentage of missing values in the dataset.
        percentage_na = (np.asarray(percentage_of_missing_per_feature)).sum()/X.shape[1]

        #Number of features with more than 1% missing values.
        number_of_features = (np.asarray(percentage_of_missing_per_feature) >= 1 ).sum()

        #Number of features with more than 1% missing values.
        number_of_samples = (np.asarray(percentage_of_missing_per_sample)>0).sum()

        #percentage of features with missing values.
        percentage_of_features_with_more_than_1 = (number_of_features/X.shape[1]).round(4)*100

        #percentage of features with missing values.
        percentage_of_samples_with_missing_values = (number_of_samples/X.shape[0]).round(4)*100

        #Percentage of missing values for features that have over 1% missing values.
        missing_values_percentage_per_missing_feature = (np.asarray(percentage_of_missing_per_feature)[(np.asarray(percentage_of_missing_per_feature) >= 1 )].sum()/number_of_features)


        #preprocess 
        categorical_features,flag=dataset_cat_flag(dataset_names[i])

        if flag == 1 :
            numeric_data =  0
            categorical_data = X.shape[1]
        else:
            numeric_data = X.shape[1] - len(categorical_features)
            categorical_data = len(categorical_features)


        perce_of_cat_vars = categorical_data/ X.shape[1]
        perce_of_num_vars = numeric_data / X.shape[1]

        vmaps=dict(zip(categorical_features, ['' for i in categorical_features]))
        
        
        new_data=encode_data(X,vmaps,flag)
        
        #pca
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(new_data)
        cumsum = np.cumsum(pca.explained_variance_ratio_, dtype=float) 
        comp_50,comp_70,comp_90 = 0, 0, 0
        for k in range(len(cumsum)):
            if cumsum[k] >= 0.5 and comp_50 == 0:
                comp_50 = k+1
            if cumsum[k] >= 0.7 and comp_70 == 0:
                comp_70 = k+1
            if cumsum[k] >= 0.9 and comp_90 == 0:
                comp_90 = k+1
        # Components percentage.
        # 
        #  
        perce_comp_50 = comp_50/len(cumsum)
        perce_comp_70 = comp_70/len(cumsum)
        perce_comp_90 = comp_90/len(cumsum)
            

        #sihouette
        cl_scores =[]
        for n_cl in [2,3,4]:
            km = KMeans(n_clusters=n_cl, random_state=42)
            km.fit_predict(new_data)
            score = silhouette_score(new_data, km.labels_, metric='euclidean',random_state=42)
            print('Silhouetter Score: %.3f' % score)
            cl_scores.append(score)
        
        

        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if index == 0:
            ft[0].insert(0,'Dataset')
            ft[0].append('Minority Class %')
            ft[0].append('Missingness percentage')
            ft[0].append('Percentage_features_with_missing_values')
            ft[0].append('Percentage_samples_with_missing_values')
            ft[0].append('Percentage_of_missing_values_per_feature_with_missing>1%')
            ft[0].append('comp 50%var')
            ft[0].append('comp 70%var')
            ft[0].append('comp 90%var')
            ft[0].append('silhouette score k=2')
            ft[0].append('silhouette score k=3')
            ft[0].append('silhouette score k=4')
            ft[0].append('n_num')
            ft[0].append('n_cat')
            spamwriter.writerow(ft[0])
        index += 1
        ft[1].insert(0,dataset_names[i].replace('50-Train-jad_','').replace('.csv',''))
        ft[1].append(imbalance_ratio)
        ft[1].append(percentage_na)
        ft[1].append(percentage_of_features_with_more_than_1)
        ft[1].append(percentage_of_samples_with_missing_values)
        ft[1].append(missing_values_percentage_per_missing_feature)
        ft[1].append(comp_50)
        ft[1].append(comp_70)
        ft[1].append(comp_90)
        ft[1].append(cl_scores[0])
        ft[1].append(cl_scores[1])
        ft[1].append(cl_scores[2])
        ft[1].append(numeric_data)
        ft[1].append(categorical_data)


        spamwriter.writerow(ft[1])