import os

import mne
from tqdm import tqdm


def load_edf_files(directory):
    edf_files = []
    edf_file_names = []
    for root, _, files in tqdm(os.walk(directory)):
        for file in files:
            if file.endswith('.edf'):
                file_path = os.path.join(root, file)
                try:
                    raw = mne.io.read_raw_edf(file_path, preload=True)
                    print(f"Successfully loaded {file_path}", raw.info)
                    edf_files.append(raw)
                    edf_file_names.append(file)
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")
    return edf_files, edf_file_names


def preprocess_edf_files(edf_files):
    preprocessed_edf_files = []
    for edf_file in edf_files:
        # Preprocess the EDF file
        # 1. Apply notch filter
        filtered_edf_file = edf_file.copy().notch_filter(freqs=50, n_jobs=-1)
        # 2. Downsample the EDF file
        downsampled_edf_file = filtered_edf_file.copy().resample(sfreq=256, n_jobs=-1)
        preprocessed_edf_files.append(downsampled_edf_file)
    return preprocessed_edf_files


def save_edf_files(edf_files, directory, file_names):
    for edf_file, file_name in zip(edf_files, file_names):
        base_name = os.path.basename(file_name)
        sub_directory = os.path.join(directory, os.path.splitext(base_name)[0])
        os.makedirs(sub_directory, exist_ok=True)
        name_without_ext = os.path.splitext(base_name)[0]
        save_path = os.path.join(sub_directory, f'{name_without_ext}.fif')
        edf_file.save(save_path, overwrite=True)
        print(f"Successfully saved {save_path}")


if __name__ == '__main__':
    # Input directory containing the EDF files
    input_directory = 'dataset/EDF_labeled_CECTS50_JBHI'
    edf_files, edf_file_names = load_edf_files(input_directory)
    # Preprocess the EDF files
    preprocessed_edf_files = preprocess_edf_files(edf_files)
    # Save the preprocessed EDF files
    output_directory = 'dataset/preprocessed_EDF_labeled_CECTS50_JBHI'
    save_edf_files(edf_files=preprocessed_edf_files, directory=output_directory, file_names=edf_file_names)
