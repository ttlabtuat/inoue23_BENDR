import argparse

import mne
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dn3.configuratron import ExperimentConfig
from dn3.trainable.processes import StandardClassification
from dn3_ext import BENDRClassification

mne.set_log_level(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EEGデータの二値分類を行う転移学習���ログラム")
    parser.add_argument('--config', default="configs/transferlearning.yml", help="実験設定のYAMLファイル")
    parser.add_argument('--debug', action='store_true', help='デバッグモードで実行')
    args = parser.parse_args()

    # 実験設定の読み込み
    experiment = ExperimentConfig(args.config)

    # デバッグモードかどうかでパラメータを切り替える
    if args.debug:
        training_params = experiment.debug_training_params
    else:
        training_params = experiment.training_params

    # パラメータの取得
    epochs = getattr(training_params, 'epochs', 50)
    batch_size = getattr(training_params, 'batch_size', 32)
    learning_rate = getattr(training_params, 'learning_rate', 0.0001)
    weight_decay = getattr(training_params, 'weight_decay', 0.0001)

    # 設��ファイルから事前学習済みモデルのパスを取得
    encoder_weights = experiment.encoder_weights
    context_weights = experiment.context_weights

    # データセットの構築
    dataset = None
    for ds_name, ds in experiment.datasets.items():
        try:
            dataset = ds.auto_construct_dataset()
            # チャンネルの確認
            print(f"使用しているチャンネル数: {dataset.identically_spaced_channels}")
        except ValueError as e:
            print(f"データセットの構築中にエラーが発生しました: {e}")
            exit(1)
        except Exception as e:
            print(f"予期しないエラーが発生しました: {e}")
            exit(1)
        break  # 一つのデータセットのみを使用

    # データセットをトレーニングとバリデーションに分割
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # モデルの初期化（チャンネル数を指定）
    n_channels = dataset.sequence_length
    model = BENDRClassification.from_dataset(
        dataset,
        multi_gpu=True,
        enc_model_hparams={'n_channels': n_channels}
    )

    # 事前学習済みの重みをロード（strict=Falseを指定）
    model.load_pretrained_modules(
        encoder_weights,
        context_weights,
        freeze_encoder=True,
        strict=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # データローダーの作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # モデルのトレーニング
    process = StandardClassification(
        model,
        device=device,
        metrics=['accuracy']
    )
    process.set_optimizer(
        torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    )

    process.fit(train_loader, val_loader, epochs=epochs)
