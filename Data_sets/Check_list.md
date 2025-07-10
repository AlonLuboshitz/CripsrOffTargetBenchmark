Dataset in a CSV format
Contains two data files - train.csv, test.csv (can include name of the dataset as a prefix)
Csv file contains two mandatory columns - sequence and label (can include additional columns)
CSV file with predictions from state-of-the-art tools on the test set

Test file with added columns - one per tool
Name of the column must be the name of the tool

Metadata file for a dataset, describing
Columns in the dataset
Dataset and the problem it is handling
Preprocessing of positive samples
A method for generating negative samples
Predictions:
Name of each tested tool
Doi of a publication of a tool
Is higher or lower predicted score better - [ascending - descending]
