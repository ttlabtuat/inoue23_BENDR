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
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6']

def get_file_names(input_directory):
    edf_file_names = []
    for root, _, files in tqdm(os.walk(input_directory)):
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

def preprocess_data(dm, output_directory, n_files, ch_names, sampling_rate):
    preprocessed_datas = []
    for i in tqdm(range(n_files)):
        pt = dm.patients[i]
        data = dm[pt].all
        eeg_data = data['data']
        n_segments, n_samples, n_channels = eeg_data.shape

        # データの形状を (n_channels, n_samples * n_segments) に変換
        eeg_data = eeg_data.transpose(2, 1, 0).reshape(n_channels, -1)

        info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data, info)
        output_file = os.path.join(output_directory, f'{pt}.fif')
        raw.save(output_file, overwrite=True)
        preprocessed_datas.append(output_file)
        print(f"Saved preprocessed data to {output_file}")

if __name__ == '__main__':
    # ディレクトリが存在しない場合は作成
    os.makedirs(CFG.output_directory, exist_ok=True)

    # EDFファイルのリストを取得
    edf_file_names = get_file_names(CFG.input_directory)

    # DataMakerのインスタンスを作成
    dm = DataMakerCont2()

    # データの前処理を実行
    preprocessed_datas = preprocess_data(dm, CFG.output_directory, CFG.n_files, CFG.ch_names, CFG.sampling_rate)