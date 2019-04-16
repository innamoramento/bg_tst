import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (5, 4)

def GetTrainDiagnosisDistributions(data, target_name):
    print('Train data ',target_name,' value counts:')
    print(data[target_name].value_counts())
    data[target_name].value_counts().plot(kind='bar', label=target_name)
    plt.legend()
    plt.title(target_name+' distributions')
    return

def CheckForNulls(data):
    total_null = 0
    for _ in data.columns:
        null_counts = data[_].isnull().sum()
        if null_counts > 0:
            total_null += 1
            print("Null values count in {} = {}".format(_, null_counts))
    if (total_null == 0):
        print("Null's was not found in data.")
    return


def FindHighlyCorrelatedFeatures(data, n=40):
    def get_pairs_2drop(data):
        # Drop diagonal and lower triangular elements
        pairs_to_drop = set()
        cols = data.columns
        for i in range(0, data.shape[1]):
            for j in range(0, i + 1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop

    def get_n_top_correlated_pairs(data, n):
        abs_corr = data.corr().abs().unstack()
        abs_corr = abs_corr.drop(labels=get_pairs_2drop(data)).sort_values(ascending=False)
        return abs_corr[0:n]

    print("Top-{} highly correlated feature pairs:".format(n))
    print(get_n_top_correlated_pairs(data, n))


def DrawBoxPlotsByDiagnosis(data, target_name, ncols = 4):
    features = list(set(data.columns)-set([target_name]))
    ncols = ncols
    nrows = int(np.ceil(len(features) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 40))
    for idx, feat in enumerate(features):
        sns.boxplot(x=target_name, y=feat, data=data, ax=axes[idx // ncols, idx % ncols])
        axes[idx // ncols, idx % ncols].set_xlabel(target_name)
        axes[idx // ncols, idx % ncols].set_ylabel(feat)
    return
