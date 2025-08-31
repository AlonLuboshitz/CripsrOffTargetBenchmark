import pandas as pd
import numpy as np
from tqdm import tqdm


def MFH_encoding(dataset_path, on_col, off_col):
    dim=7
    print(f"Processing dataset: {dataset_path}")
    if isinstance(dataset_path,str):
        df = pd.read_csv(dataset_path)
    else: df = dataset_path

    encoded_predict, encoded_on, encoded_off = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        on_off_encoded, on_encoded, off_encoded = MFH(row[on_col], row[off_col],
                                                                                     dim=dim)
        encoded_predict.append(on_off_encoded)
        encoded_on.append(on_encoded)
        encoded_off.append(off_encoded)

    X_predict = np.array(encoded_predict, dtype=np.float32).reshape((len(encoded_predict), 1, 24, 7))
    X_on = np.array(encoded_on, dtype=np.float32).reshape((len(encoded_on), 1, 24, 5))
    X_off = np.array(encoded_off, dtype=np.float32).reshape((len(encoded_off), 1, 24, 5))
    #labels = df[label_col].values.astype('int')
    return X_predict,X_on,X_off


def MFH(guide_seq, off_seq, dim=5):
    code_dict = {'A': [1, 0, 0, 0, 0], 'T': [0, 1, 0, 0, 0], 'G': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0],
                 '-': [0, 0, 0, 0, 1]}
    direction_dict = {'A': 5, 'G': 4, 'C': 3, 'T': 2, '-': 1}
    tlen = 24
    guide_seq = "-" * (tlen - len(guide_seq)) + guide_seq.upper()
    off_seq = "-" * (tlen - len(off_seq)) + off_seq.upper()

    gRNA_list = list(guide_seq)
    off_list = list(off_seq)
    pair_code = []
    on_encoded_matrix = np.zeros((24, 5), dtype=np.float32)
    off_encoded_matrix = np.zeros((24, 5), dtype=np.float32)

    on_off_dim7_codes = []
    for i in range(len(gRNA_list)):
        if gRNA_list[i] == 'N':
            gRNA_list[i] = off_list[i]

        if gRNA_list[i] == '_':
            gRNA_list[i] = '-'

        if off_list[i] == '_':
            off_list[i] = '-'

        gRNA_base_code = code_dict[gRNA_list[i].upper()]
        DNA_based_code = code_dict[off_list[i].upper()]
        diff_code = np.bitwise_or(gRNA_base_code, DNA_based_code)

        if(dim==7):
            dir_code = np.zeros(2)
            if gRNA_list[i] == "-" or off_list[i] == "-" or direction_dict[gRNA_list[i]] == direction_dict[off_list[i]]:
                pass
            else:
                if direction_dict[gRNA_list[i]] > direction_dict[off_list[i]]:
                    dir_code[0] = 1
                else:
                    dir_code[1] = 1

            pair_code.append(np.concatenate((diff_code, dir_code)))
        else:
            pair_code.append(diff_code)
        on_encoded_matrix[i] = code_dict[gRNA_list[i]]
        off_encoded_matrix[i] = code_dict[off_list[i]]

    pair_code_matrix = np.array(pair_code, dtype=np.float32).reshape(1, 1, 24, dim)

    return pair_code_matrix, on_encoded_matrix.reshape(1, 1, 24, 5), off_encoded_matrix.reshape(1, 1, 24, 5)
