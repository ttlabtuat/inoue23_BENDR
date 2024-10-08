import copy
import itertools
import os

import numpy as np
import pandas as pd
import scipy.stats
from EDFLIB import CREDF
from sklearn.model_selection import train_test_split

'''
脳波セグメントとラベルのペアを生成するクラスたちです。

- DataMaker
    セグメント切り出し方法: 注釈されたラベル位置に基づいて、前後に合計1秒間で切り出す
    前処理: セグメントごとに Z-score normalization を適用
- DataMakerRAW
    セグメント切り出し方法: DataMakerと同じ
    前処理: なし（生の波形そのまま）
- DataMakerCont
    セグメント切り出し方法: 脳波記録の先頭から1秒ずつ連続的に取得
    前処理: セグメントごとに Z-score normalization を適用
- DataMakerRawCont
    セグメント切り出し方法: DataMakerContと同じ
    前処理: なし（生の波形そのまま）
- DataMakerCont2
    セグメント切り出し方法: DataMakerContと同じ
    前処理: ノッチフィルタを適用
'''


class DataMaker(object):
    '''
        edf, csv を機械学習用のデータ形式に整形して，
        イテレータとして利用できるクラスです．
    '''
    # {}内には患者IDが入る
    _EDF_PATH = os.path.join(os.environ['HOME'], 'epilepsy_detect/input/EDF_labeled_CECTS50_JBHI/{0}/{0}.edf')
    
    # 1セグメントのサンプル数
    _SEGMENT_SIZE = 512
    # 1セグメント内におけるピーク点前後のサンプル数の比率
    _SEGMENT_RATIO = (3, 7)

    _patients = [
        'RJ92606C', 'RJ926063', 'RJ92601Q', 'RJ92606N','DJ002001', 
        'CJ493EXT', 'FJ9633JW', 'CJ493FG6', 'FJ96313K', 'FJ96305Y',
        'CJ493BSD', 'CJ493E4N', 'FJ9630I9', 'CJ493C0P', 'FJ9631MG',
        'FJ9631YW', 'FJ9631LS', 'FJ9633M4', 'CJ493EJC', 'FJ9631CT',
        'FJ9632RU', 'FJ9632QY', 'CJ493FCW', 'CJ493FY9', 'CJ493CD7',
        'FJ9631VS', 'FJ963181', 'CJ4939BP', 'FJ9632Q2', 'CJ493ERL',
        'FJ9632S8', 'FJ96322V', 'CJ493EZB', 'CJ493AT9', 'FJ963497',
        'CJ493G8S', 'CJ493FR9', 'FJ96352X', 'CJ493DUV', 'FJ96356G',
        'FJ9633G0', 'CJ493AC6', 'CJ493DDU', 'CJ493E6J', 'CJ493FPQ',
        'CJ4939WZ', 'FJ9632AH', 'FJ9634LP', 'FJ96344S', 'FJ96351H',
        ]
    
    _MP_PATTERN = {
        "Fp1": "A1", "Fp2": "A2", "F3": "A1", "F4": "A2", "C3": "A1",
        "C4": "A2", "P3": "A1", "P4": "A2", "O1": "A1", "O2": "A2",
        "F7": "A1", "F8": "A2", "T3": "A1", "T4": "A2", "T5": "A1",
        "T6": "A2",
        #"X1": "X2", "E": "E", "X1": "X2", "X3": "X4", "X5": "X6",
        }

    def __init__(self, valid_rate=None):
        # データ保持に関する変数
        self._valid_rate = valid_rate
        self._data, self._data_orig = {}, {}  # 無印には前処理を適用したものを格納
        self._labels = {}
        self._times = {}
        self._channels = {}
        self._annotators = {}

        for pt in self._patients:
            data, labels, times, channels, annotators\
                = self._make_formated_data(pt)

            # 0=Artifact と 1=Spikeのみ使う
            useable_idx = labels <= 1
            data = data[useable_idx]
            labels = labels[useable_idx]
            times = times[useable_idx]
            channels = channels[useable_idx]
            annotators = annotators[useable_idx]

            self._data_orig[pt] = data
            self._data[pt] = self._apply_preprocessing(to=data, patient=pt)
            self._labels[pt] = labels
            self._times[pt] = times
            self._channels[pt] = channels
            self._annotators[pt] = annotators
        
        # タイムスタンプファイルを更新
        #self._upadte_timestamp()

    def _apply_preprocessing(self, to: np.ndarray, patient):
        data = to
        orig_shape = data.shape
        data = data.reshape(orig_shape[0], -1)
        data_std = scipy.stats.zscore(data, axis=1)
        #mean = data.mean(1)
        #std = data.std(1) + 1E-12
        #data_std = (data - mean[:,None]) / std[:,None]
        data_std = data_std.reshape(*orig_shape)

        # kerasのconv1dの入力は (batch_size, steps, input_dim)
        # 戻り値はこの形状であること
        return data_std

    def _get_label_num(self, label):
        str_to_num = {'Artifact': 0, 'Open_Eye': 0,
                        'Parox_Discharge': 1, 'SpikeWave': 1,
                        'SlowWave': 0, 'NormalWave': 0, 'Seizure': 0,
                        'Stage 2': 0,
                        }
        return str_to_num[label] if label in str_to_num else 99

    def _parse_csv(self, csv_path):
        df = pd.read_csv(csv_path, encoding="shift_jis")

        # 注釈時刻で並べ替える
        df = df.sort_values(by=df.columns.values[1], ascending=True)

        # 4つのリストにアノテーション情報を追加していく
        labels = []   # ラベル
        times = []   # 時刻(s)
        channels = []   # チャンネル
        annotators = []   # 注釈者

        for row in df.values:
            # 注釈ファイルにはヘッダーやデータセットアノテーションがあるので、
            # その場合は読み込みをスキップする
            if 'Hz' not in row[8]:
                continue
            
            time = float(row[1])
            label_idx = self._get_label_num(row[3])
            ch = row[5]
            annotator = row[0]
            
            # 前回のラベルの近傍にあって、且つ異なるラベルの注釈の場合を考慮する
            if len(times) > 0 and time - times[-1] < 1.0\
                              and label_idx != labels[-1]:
                # もし直前のラベルが非てんかん性であれば、それを削除する
                if labels[-1] == 0:
                    labels = labels[:-1]   # ラベル
                    times = times[:-1]   # 時刻(s)
                    channels = channels[:-1]   # チャンネル
                    annotators = annotators[:-1]   # 注釈者
                # もし直前がてんかん性であれば、今回のラベルはスキップする
                else:
                    continue

            labels.append(label_idx)
            times.append(time)
            channels.append(ch)
            annotators.append(annotator)

        return np.array(labels), np.array(times), np.array(channels), np.array(annotators)

    def _make_formated_data(self, patient):
        # 整形済みデータを保存していなかった場合は新規作成
        edf_path = self._EDF_PATH.format(patient)
        csv_path = edf_path.replace('.edf', '.csv')
        
        # CSVからラベル情報を解析
        labels, times, channels, annotators = self._parse_csv(csv_path)
        
        # CSVファイルに出現する全チャンネル名を重複なく取得
        active_chs = list(self._MP_PATTERN.keys())
        
        # active_chs をもとにEDFを読み込み
        edf = CREDF(edf_path, active_matches=active_chs)

        # [n_samples, n_channels]の形で raw を取得
        raw = edf.raw.reshape(edf.n_channels, edf.n_samples).T
        
        # times配列をx軸index化
        times_idx = (times * edf.samp_freq).astype(int)

        # 前後1秒分のデータ配列を作成
        n_segments = times_idx.shape[0]
        data = np.empty((n_segments, self._SEGMENT_SIZE, edf.n_channels))

        n_head_sample = int(
            self._SEGMENT_RATIO[0] / (self._SEGMENT_RATIO[0] + self._SEGMENT_RATIO[1]) * self._SEGMENT_SIZE)
        n_tail_sample = int(self._SEGMENT_SIZE - n_head_sample)

        for i, t_idx in enumerate(times_idx):
            start_idx = t_idx - n_head_sample
            end_idx = t_idx + n_tail_sample
            data[i] = raw[start_idx:end_idx]

        return (data, labels, times, channels, annotators)

    def __iter__(self):
        self._counter = 0
        self._stop = len(self._patients)
        n_trains = len(self._patients) - 1
        self._c_trains = itertools.combinations(self._patients, n_trains)
        return self

    def __next__(self):
        if self._counter > self._stop:
            raise StopIteration()
        self._counter += 1
        trains = list(next(self._c_trains))
        tests = list(set(self._patients)-set(trains))
        X_train = np.vstack([self._data[patient] for patient in trains])
        y_train = np.hstack([self._labels[patient] for patient in trains])
        X_test = np.vstack([self._data[patient] for patient in tests])
        y_test = np.hstack([self._labels[patient] for patient in tests])
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        if self._valid_rate is None:
            Xs = (X_train, X_test)
            ys = (y_train, y_test)
        else:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_size=self._valid_rate, random_state=0)
            Xs = (X_train, X_valid, X_test)
            ys = (y_train, y_valid, y_test)
        return trains, tests, Xs, ys

    def next(self):
        # Python 2 対応
        return self.__next__()

    def __getitem__(self, item):
        if item in self._patients:
            dm_child = copy.copy(self)
            dm_child._patients = [item]
            dm_child._data_orig = {item: self._data_orig[item]}
            dm_child._data = {item: self._data[item]}
            dm_child._labels = {item: self._labels[item]}
            dm_child._times = {item: self._times[item]}
            dm_child._channels = {item: self._channels[item]}
            dm_child._annotators = {item: self._annotators[item]}
            return dm_child

        elif (isinstance(item, list) or isinstance(item, set)) and set(item) <= set(self._patients):
            dm_child = copy.copy(self)
            dm_child._patients = list(item)
            dm_child._data_orig = {pt: self._data_orig[pt] for pt in item}
            dm_child._data = {pt: self._data[pt] for pt in item}
            dm_child._labels = {pt: self._labels[pt] for pt in item}
            dm_child._times = {pt: self._times[pt] for pt in item}
            dm_child._channels = {pt: self._channels[pt] for pt in item}
            dm_child._annotators = {pt: self._annotators[pt] for pt in item}
            return dm_child

        elif item in self.annotators:
            dm_child = copy.copy(self)
            dm_child._patients = []
            dm_child._data_orig = {}
            dm_child._data = {}
            dm_child._labels = {}
            dm_child._times = {}
            dm_child._channels = {}
            dm_child._annotators = {}

            for pt in self._patients:
                ants = self._annotators[pt]
                if item in ants:
                    dm_child._data_orig[pt] = self._data_orig[pt][item==ants]
                    dm_child._data[pt] = self._data[pt][item==ants]
                    dm_child._labels[pt] = self._labels[pt][item==ants]
                    dm_child._times[pt] = self._times[pt][item==ants]
                    dm_child._channels[pt] = self._channels[pt][item==ants]
                    dm_child._annotators[pt] = self._annotators[pt][item==ants]
                    dm_child._patients.append(pt)
            return dm_child
        else:
            raise IndexError('Index is not a valid patient or annotator')

    @property
    def all(self):
        pts = self._patients
        return {'data': np.vstack([self._data[pt] for pt in pts]),
                'label': np.hstack([self._labels[pt] for pt in pts]),
                'time': np.hstack([self._times[pt] for pt in pts]),
                'channel': np.hstack([self._channels[pt] for pt in pts]),
                'annotator': np.hstack([self._annotators[pt] for pt in pts])}

    @property
    def output_dim(self):
        return 1

    @property
    def input_dim(self):
        return list(self._data.values())[0].shape[2]

    @property
    def len_dat(self):
        return list(self._data.values())[0].shape[1]

    @property
    def patients(self):
        return self._patients

    @property
    def n_patients(self):
        return len(self._patients)

    @property
    def annotators(self):
        annotators = set()
        for a in self._annotators.values():
            annotators |= set(a)
        return list(annotators)

    @property
    def n_annotators(self):
        return len(self.annotators)

    @property
    def n_annotations(self):
        return np.sum([len(self._labels[pt]) for pt in self._patients])

class DataMakerRAW(DataMaker):
    def _apply_preprocessing(self, to, patient):
        return to

class DataMakerCont(DataMaker):
    def _make_formated_data(self, patient):
        # 整形済みデータを保存していなかった場合は新規作成
        edf_path = self._EDF_PATH.format(patient)
        csv_path = edf_path.replace('.edf', '.csv')
        
        # CSVからラベル情報を解析
        labels, times, channels, annotators = self._parse_csv(csv_path)
        
        # CSVファイルに出現する全チャンネル名を重複なく取得
        active_chs = list(self._MP_PATTERN.keys())
        
        # active_chs をもとにEDFを読み込み
        edf = CREDF(edf_path, active_matches=active_chs)

        # [n_samples, n_channels]の形で raw を取得
        raw = edf.raw.reshape(edf.n_channels, edf.n_samples).T

        # times配列をx軸index化
        times_idx = (times * edf.samp_freq).astype(int)

        # 前後1秒分のデータ配列を作成
        n_segments = times_idx.shape[0]
        data = np.empty((n_segments, self._SEGMENT_SIZE, edf.n_channels))

        n_head_sample = int(
            self._SEGMENT_RATIO[0] / (self._SEGMENT_RATIO[0] + self._SEGMENT_RATIO[1]) * self._SEGMENT_SIZE)
        n_tail_sample = int(self._SEGMENT_SIZE - n_head_sample)

        for i, t_idx in enumerate(times_idx):
            start_idx = (t_idx // self._SEGMENT_SIZE) * self._SEGMENT_SIZE
            end_idx = start_idx + self._SEGMENT_SIZE
            data[i] = raw[start_idx:end_idx]

        return (data, labels, times, channels, annotators)

class DataMakerRawCont(DataMakerCont):
    _PATH_data = os.path.dirname(__file__)\
        + '/__pycache__/{}_data_raw_stride.npy'

    def _make_formated_data(self, patient):
        # 整形済みデータを保存していなかった場合は新規作成
        edf_path = self._EDF_PATH.format(patient)
        csv_path = edf_path.replace('.edf', '.csv')
        
        # CSVからラベル情報を解析
        labels, times, channels, annotators = self._parse_csv(csv_path)
        
        # CSVファイルに出現する全チャンネル名を重複なく取得
        active_chs = list(self._MP_PATTERN.keys())
        
        # active_chs をもとにEDFを読み込み
        edf = CREDF(edf_path, active_matches=active_chs)

        # [n_samples, n_channels]の形で raw を取得
        raw = edf.raw.reshape(edf.n_channels, edf.n_samples).T

        # モノポーラ適用
        edf_r = CREDF(edf_path, active_matches=['A1', 'A2'])
        raw_r = edf_r.raw.reshape(2, edf.n_samples).T
        for i, ch in enumerate(active_chs):
            if int(ch[-1]) % 2 == 1:
                raw[:, i] = raw[:, i] - raw_r[:, 0]
            else:
                raw[:, i] = raw[:, i] - raw_r[:, 1]
        
        # times配列をx軸index化
        times_idx = (times * edf.samp_freq).astype(int)

        # 前後1秒分のデータ配列を作成
        n_segments = times_idx.shape[0]
        data = np.empty((n_segments, self._SEGMENT_SIZE, edf.n_channels))

        n_head_sample = int(
            self._SEGMENT_RATIO[0] / (self._SEGMENT_RATIO[0] + self._SEGMENT_RATIO[1]) * self._SEGMENT_SIZE)
        n_tail_sample = int(self._SEGMENT_SIZE - n_head_sample)

        for i, t_idx in enumerate(times_idx):
            start_idx = (t_idx // self._SEGMENT_SIZE) * self._SEGMENT_SIZE
            end_idx = start_idx + self._SEGMENT_SIZE
            data[i] = raw[start_idx:end_idx]

        return (data, labels, times, channels, annotators)

    def _apply_preprocessing(self, to, patient):
        return to


class DataMakerCont2(DataMakerCont):
    _PATH_data = os.path.dirname(__file__)\
        + '/__pycache__/{}_data_raw_stride.npy'

    def _make_formated_data(self, patient):
        # 整形済みデータを保存していなかった場合は新規作成
        edf_path = self._EDF_PATH.format(patient)
        csv_path = edf_path.replace('.edf', '.csv')
        
        # CSVからラベル情報を解析
        labels, times, channels, annotators = self._parse_csv(csv_path)
        
        # CSVファイルに出現する全チャンネル名を重複なく取得
        active_chs = list(self._MP_PATTERN.keys())
        
        # active_chs をもとにEDFを読み込み
        edf = CREDF(edf_path, active_matches=active_chs)
        # バンドパスフィルタを適用
        edf = edf.apply_notch(50)

        # [n_samples, n_channels]の形で raw を取得
        raw = edf.raw.reshape(edf.n_channels, edf.n_samples).T

        # モノポーラ適用
        edf_r = CREDF(edf_path, active_matches=['A1', 'A2'])
        raw_r = edf_r.raw.reshape(2, edf.n_samples).T
        for i, ch in enumerate(active_chs):
            if int(ch[-1]) % 2 == 1:
                raw[:, i] = raw[:, i] - raw_r[:, 0]
            else:
                raw[:, i] = raw[:, i] - raw_r[:, 1]
        
        # times配列をx軸index化
        times_idx = (times * edf.samp_freq).astype(int)

        # 前後1秒分のデータ配列を作成
        n_segments = times_idx.shape[0]
        data = np.empty((n_segments, self._SEGMENT_SIZE, edf.n_channels))

        n_head_sample = int(
            self._SEGMENT_RATIO[0] / (self._SEGMENT_RATIO[0] + self._SEGMENT_RATIO[1]) * self._SEGMENT_SIZE)
        n_tail_sample = int(self._SEGMENT_SIZE - n_head_sample)

        for i, t_idx in enumerate(times_idx):
            start_idx = (t_idx // self._SEGMENT_SIZE) * self._SEGMENT_SIZE
            end_idx = start_idx + self._SEGMENT_SIZE
            data[i] = raw[start_idx:end_idx]

        return (data, labels, times, channels, annotators)

    def _apply_preprocessing(self, to, patient):
        return to