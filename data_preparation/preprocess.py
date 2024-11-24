import os

import mne
import numpy as np
from tqdm import tqdm

from DataMaker import DataMakerCont2

# Numpyにstr属性がない場合は追加
if not hasattr(np, 'str'):
    np.str = str

class CFG:
    input_directory = 'dataset/EDF_labeled_CECTS50_JBHI'
    output_directory = 'dataset/preprocessed_EDF_labeled_CECTS50_JBHI'
    n_files = 50
    n_channels = 16
    sampling_rate = 256
    ch_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6']


def get_file_names():
    edf_file_names = []
    for root, _, files in tqdm(os.walk(CFG.input_directory)):
        for file in files:
            if file.endswith('.edf'):
                file_path = os.path.join(root, file)
                try:
                    raw = mne.io.read_raw_edf(file_path, preload=True)
                    print(f"Successfully loaded {file_path}", raw.info)
                    edf_file_names.append(file_path)
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")
    return edf_file_names


def preprocess_data(dm):
    preprocessed_datas = []
    for i in tqdm(range(CFG.n_files)):
        pt = dm.patients[i]
        data = dm[pt].all
        eeg_data = data['data']
        eeg_data = eeg_data.transpose(2, 1, 0).reshape(CFG.n_channels, -1)
        info = mne.create_info(ch_names=CFG.ch_names, sfreq=CFG.sampling_rate, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data, info)
        preprocessed_datas.append(raw)
    return preprocessed_datas


def save_edf_files(datas, file_names):
    for data, file_name in zip(datas, file_names):
        base_name = os.path.basename(file_name)
        sub_directory = os.path.join(CFG.output_directory, os.path.splitext(base_name)[0])
        os.makedirs(sub_directory, exist_ok=True)
        name_without_ext = os.path.splitext(base_name)[0]
        save_path = os.path.join(sub_directory, f'{name_without_ext}.fif')
        data.save(save_path, overwrite=True)
        print(f"Successfully saved {save_path}")


if __name__ == '__main__':
    # Input directory containing the EDF files
    file_names = get_file_names()
    # Preprocess the EDF files
    dm = DataMakerCont2()
    preprocessed_datas = preprocess_data(dm)
    # Save the preprocessed data
    save_edf_files(datas=preprocessed_datas, file_names=file_names)