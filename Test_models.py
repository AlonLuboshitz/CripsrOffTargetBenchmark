import pandas as pd
import json
import sys
import traceback

from Models.Nuclea_seq.modeling import log10_crispr_specificity
JSON_FILE = 'Running_args.json'
TEST_DATA = 'Data_sets/Test.csv'
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
    return score_series

def assign_to_test_data_and_save(test_data, score_series, model_name):
    """Assigns the model predictions to the test data

    Args:
        test_data (test.csv dataframe): The test data.csv file  
        score_series (pandas series): predictions with matching indexes
        model_name (str): model name - column to be filled with the series
    """
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
    assign_to_test_data_and_save(test_data, scores, model_name)