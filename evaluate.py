from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import numpy as np
models = ["Nuclea-seq","CRISPR-SGRU","CRISPR-IP","CRISPR-NET","CRISPR-MFH","CRISPR-BULGE"]
def evaluate_models(data_path="Data_sets/CRISPR_Test.csv"):
    global models
    if isinstance(data_path,str):
        data = pd.read_csv(data_path)
    labels = np.where(data['reads']>0,1,0)
    columns = data.columns
    models = [model for model in models if model in columns]
    rocs = {}
    prcs = {}
    for model in models:
        predictions = data[model].dropna()
        labels_ = labels[predictions.index] if len(predictions) < len(labels) else labels
        rocs[model] = roc_auc_score(labels_,predictions)
        prcs[model] = average_precision_score(labels_,predictions)
    
    data = pd.DataFrame(data=[rocs,prcs],index=['AUROC','AURPC'])
    data.to_csv('evaluation.csv')
evaluate_models()