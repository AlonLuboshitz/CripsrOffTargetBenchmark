import pandas as pd
import json
import sys
import traceback
import numpy as np
from Models.Nuclea_seq.modeling import log10_crispr_specificity
from Models.CRISPR_IP.encoding import my_encode_on_off_dim
from Models.SGRU.Encoder_sgRNA_off import Encoder
from Models.SGRU.MODEL import Crispr_SGRU
from Models.MFH.Csv2pkl import MFH_encoding
from Models.CRIPSR_net import Encoder_sgRNA_off
from Models.CRISPR_Bulge.train_and_predict_scripts.utilities import ensemble_predict

import os
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
#from tensorflow.python.keras.saving.saved_model.load import TensorFlowOpLayer

JSON_FILE = 'Running_args.json'
TEST_DATA = 'Data_sets/CRISPR_Test.csv'





def encordingXtest_sgru(Xtest):
    final_code = []
    for idx, row in Xtest.iterrows():
        on_seq = row[0]
        off_seq = row[1]
        en = Encoder(on_seq=on_seq, off_seq=off_seq, with_category=True, label=None)
        final_code.append(en.on_off_code)
    return np.array(final_code)


def get_sgrnas_otss(test_data , sgrna_columnm, off_target_column):
    """Return guide rnas aligned sequences and their off-target dna sequences.

    Args:
        test_data (data frame): pandas data frame
        sgrna_columnm (str): column of the sgrna
        off_target_column (str): column of the off-target

    Returns:
    sgrna,otss
    """
    if not isinstance(test_data,pd.DataFrame):
        test_data=pd.read_csv(test_data)
    return test_data[[sgrna_columnm,off_target_column]]

def create_inputs_to_models(model_name, sgrna_otss):
    """Given a data frame with sgrnas and otss create the matching input for each model

    Args:
        model_name (str): 'Nuclea-seq,..'
        sgrna_otss (data frame): pandas df with sgrna and otss

    """
    target_column,off_target_column = sgrna_otss.columns[0],sgrna_otss.columns[1]
    if model_name == 'Nuclea-seq': # only trained on TGG pam sequences
        available_pams = ['TGG', '-GG', 'T-G', 'TG-']
        filtered_sgrna_otss = sgrna_otss[sgrna_otss[off_target_column].str[-3:].isin(available_pams)]
        filtered_sgrna_otss[target_column] = filtered_sgrna_otss[target_column].str[:-3]
        filtered_sgrna_otss[off_target_column] = filtered_sgrna_otss[off_target_column].str[:-3]
        return filtered_sgrna_otss
    elif model_name == 'CRISPR-IP': # use interall crispr-ip function to encode
        
        xtest= np.array(sgrna_otss.apply(lambda row: my_encode_on_off_dim(row[target_column], row[off_target_column]), axis = 1).to_list())
        encoder_shape=(24,7)
        seq_len, coding_dim = encoder_shape
        xtest = xtest.reshape(xtest.shape[0], 1, seq_len, coding_dim)
        xtest = xtest.astype('float32')
        return xtest
    elif model_name == 'CRISPR-SGRU':
        sgrna_otss= encordingXtest_sgru(sgrna_otss)
        sgrna_otss = np.expand_dims(sgrna_otss, axis=1)
        return sgrna_otss
        
        
    elif model_name == 'CRISPR-MFH':
        X_predict_test,X_on_test,X_off_test = MFH_encoding(sgrna_otss,target_column,off_target_column)
        return X_predict_test,X_on_test,X_off_test
    elif model_name == 'MOFF':
        pass
    elif model_name == 'CRISPR-NET':
        input_codes = []
        for idx, row in sgrna_otss.iterrows():
            on_seq = row[target_column]
            off_seq = row[off_target_column]
            en = Encoder_sgRNA_off.Encoder(on_seq=on_seq, off_seq=off_seq)
            input_codes.append(en.on_off_code)
        input_codes = np.array(input_codes)
        input_codes = input_codes.reshape((len(input_codes), 1, 24, 7))
        return input_codes

    elif model_name == 'Aldit_Off':
        pass
    elif model_name == 'CRISPR-BULGE':
        sgrna_otss.rename(columns={"realigned_target":"Align.sgRNA","offtarget_sequence":"Align.off-target"},inplace=True)
        return sgrna_otss
def run_models(model_name, data, target_column = None, off_target_column = None):
    """Run the model on the data and outputs a series of predictions for each index in the data.

    Args:
        model_name (str): the model name
        data (pandas data frame): data frame with sgrnas, off targets
        target_column (str, optional): target column name. Defaults to None.
        off_target_column (str, optional): off target column. Defaults to None.

    Returns:
        pd.Series: Series of indexes and predictions from the given model
    """
    scores = []
    
    
    if model_name == "Nuclea-seq":
        for idx, row in data.iterrows():
            try:
                score = log10_crispr_specificity("WT", "TGG", row[target_column], row[off_target_column])
            except Exception as e:
                score = None  # or np.nan
                print(e)
                traceback.print_exc()
            scores.append((idx, score))
        score_series = pd.Series(dict(scores))
    elif model_name == 'CRISPR-IP':
        #custom_objects = {'TensorFlowOpLayer': TensorFlowOpLayer}
        
        model = load_model('Models/CRISPR_IP/example+crispr_ip.h5', safe_mode=False)
        score_series = model.predict(data)
        score_series = score_series[:,1]
    
    elif model_name == 'CRISPR-SGRU':
        models = [os.path.join('Models/SGRU/CHANGEseq',model) for model in os.listdir('Models/SGRU/CHANGEseq')]
        probs =[]
        for weighs_path in models:
            model=Crispr_SGRU()
            model.load_weights(weighs_path)
            y_pred=model.predict(data)
            y_prob = y_pred[:, 1]
            y_prob = np.array(y_prob)
            probs.append(y_prob)
        probs = np.array(probs)
        score_series  = probs.mean(axis=0)
       
    elif model_name == 'CRISPR-MFH':
        model = load_model('Models/MFH/CRISPR_MFH_best_model.h5')
        predictions = model.predict([data[0], data[1], data[2]])
        score_series = predictions[:, 1]
    elif model_name == 'MOFF':
        pass
    elif model_name == 'CRISPR-NET':
        json_file = open("Models/CRIPSR_net/scoring_models/CRISPR_Net_CIRCLE_elevation_SITE_structure.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("Models/CRIPSR_net/scoring_models/CRISPR_Net_CIRCLE_elevation_SITE_weights.h5")
        print("Loaded model from disk!")
        score_series = loaded_model.predict(data).flatten()
    elif model_name == 'Aldit_Off':
        pass
    elif model_name == 'CRISPR-BULGE':
        score_series = ensemble_predict(
        ensemble_components_file_path_and_name_list=[
            "Models/CRISPR_Bulge/files/bulges/1_folds/5_revision_ensemble_{}_exclude_RHAMPseq_continue_from_change_seq/"
            "read_ts_0/cleavage_models/aligned/FullGUIDEseq/classification/c_2/"
            "ln_x_plus_one_trans/model_fold_0".format(i) for i in range(5)],
        dataset_df=data)
        score_series = score_series['pred_averege_ensemble']
    return score_series


    
    

def assign_to_test_data_and_save(score_series, model_name):
    """Assigns the model predictions to the test data

    Args:
          
        score_series (pandas series): predictions with matching indexes
        model_name (str): model name - column to be filled with the series
    """
    test_data = pd.read_csv(TEST_DATA)
    test_data[model_name] = score_series
    test_data.to_csv(TEST_DATA,index=False)

def set_parameters(model_name):
    """Set model name, target column, offtarget column per model
    Uses Json file with corresponding argument matching.

    Args:
        model_name (str): the model name

    Returns:
        (tuple): model_name, target column, off-target column
    """
    with open(JSON_FILE,'r') as f:
        parameters = json.load(f)
    parameters = parameters[model_name]
    return parameters["Model_name"],parameters["Target_column"],parameters["Off_target_column"]

if __name__ == "__main__":
    model_name, target_column, off_target_column = set_parameters(sys.argv[1])
    test_data = get_sgrnas_otss(TEST_DATA,target_column,off_target_column)
    input_to_model = create_inputs_to_models(model_name,test_data)
    scores = run_models(model_name, input_to_model, target_column, off_target_column)
    assign_to_test_data_and_save(scores, model_name)





