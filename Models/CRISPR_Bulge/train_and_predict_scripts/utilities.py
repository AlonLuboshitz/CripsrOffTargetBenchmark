
import pandas as pd
from OT_deep_score_src.predict_utilities import  predict
from OT_deep_score_src.general_utilities import Padding_type, Encoding_type
from OT_deep_score_src.models_inter import Model



def ensemble_predict(ensemble_components_file_path_and_name_list, dataset_df):
    """
    Generates predictions using an ensemble of pre-trained models and saves the results.

    Args:
        ensemble_components_file_path_and_name_list (list): A list of file paths and names
                                                            of the pre-trained ensemble component models.
        dataset_df (pd.DataFrame or str):  Either a DataFrame containing the test data or
                                               a file path to a CSV file containing the test data.

    Returns:
        pd.DataFrame: The DataFrame containing test data with ensemble predictions.
    """
    if isinstance(dataset_df, str):
        dataset_df = pd.read_csv(dataset_df)

    num_of_esemble_components = len(ensemble_components_file_path_and_name_list)
    for i, file_path_and_name in enumerate(ensemble_components_file_path_and_name_list):
        model = Model.load_model_instance(file_path_and_name)

        y_prediction = predict(
            test_dataset_df=dataset_df, model=model,
            include_distance_feature=False, include_sequence_features=True,
            include_gmt_score=False, include_nuclea_seq_score=False,
            padding_type=Padding_type.GAP, aligned=True,
            bulges=True, encoding_type=Encoding_type.ONE_HOT,
            flat_encoding=False  # Flat encoding for the NN models is False
            )

        if num_of_esemble_components > 1:
            dataset_df["pred_ensemble_component_{}".format(i)] = y_prediction
        else:
            dataset_df["pred"] = y_prediction

    if num_of_esemble_components > 1:
        dataset_df["pred_averege_ensemble"] = dataset_df[
            ["pred_ensemble_component_{}".format(i) for i in range(num_of_esemble_components)]].mean(axis=1)

    # save the predictions
    #dataset_df.to_csv("predictions.csv", index=False)

    return dataset_df
