import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
import scipy.stats as sstats

dataset_path = "output/loda_done/"
dataset_names = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]

print("Name #Samples, #Features, #OR, AVG Feature Range, AVG Feature mean, AVG Feature norm. mean, AVG Feature STD, AVG Feature IQR, AVG Feature Outside 1 & 99, AVG Features Skewness, AVG Feature Relative STD, \% of normally dist. Features",
"MAD")
for name in dataset_names:
    dataset = pd.read_csv(dataset_path+name, delimiter=",")
    nsamples = dataset.shape[0]
    nfeats = dataset.shape[1] - 1
    outliers = 0
    inliers = 0
    for val in dataset["is_anomaly"]:
        if(val == 1):
            outliers += 1
        else:
            inliers += 1

    outliers_ratio = (outliers) / (inliers + outliers)

    frange = 0
    fmean = 0
    fstd = 0
    fiqr = 0
    perc = []
    outside99and1 = 0
    p = 0
    cov = 0
    skewness = 0
    entropy = 0
    qcd = 0
    perc_normal = 0
    mad = 0
    rsd = 0
    norm_mean = 0
    for (cname, cvalues) in dataset.iteritems():
        if(cname == "is_anomaly"):
            break
        outside99and1_tmp = 0
        values = cvalues.values
        fstd += np.std(values)
        fmean += np.mean(values)
        frange += (max(values) - min(values))
        if(max(values != 0)):
            norm_mean += (np.mean(values) / max(values))
        fiqr += sstats.iqr(values)
        perc_tmp = np.percentile(values, [1,25,75,99])
        perc.append(perc_tmp)
        outside99and1_tmp += len(values[values > perc_tmp[3]])
        outside99and1_tmp += len(values[values < perc_tmp[0]])
        outside99and1 += (outside99and1_tmp / len(values))
        #qcd += ((perc_tmp[2] - perc_tmp[1]) / (perc_tmp[2] + perc_tmp[1]))
        skewness += cvalues.skew()
        #entropy += sstats.entropy(values, base=2) / np.log(len(values))
        mad += sstats.median_absolute_deviation(values)
        if(np.mean(values) != 0):
            rsd += np.std(values) / np.mean(values)

        stat, p = sstats.shapiro(values)
        alpha = 0.05
        if p > alpha:
            perc_normal += 1
        #for (cname_2, cvalues_2) in dataset.iteritems():
            #p_tmp = 0
            #cov_tmp = 0
            #if(cname == "is_anomaly" or cname == cname_2):
            #    break
            #p_tmp2, _ = sstats.pearsonr(cvalues.values, cvalues_2.values)
            #p_tmp += p_tmp2
            #cov_tmp += np.cov(cvalues.values, cvalues_2.values)
        #p += p_tmp
        #cov += cov_tmp
    print(name,",",nsamples,",",nfeats,",",outliers_ratio,",",(frange/nfeats),",",(fmean/nfeats),",",(norm_mean),",",(fstd/nfeats),",",(fiqr/nfeats),",",(outside99and1/nfeats),",",(skewness/nfeats),",",(rsd/nfeats),",", (perc_normal/nfeats)
    ,",",(mad/nfeats))