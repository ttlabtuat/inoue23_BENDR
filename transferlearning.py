import argparse

import mne
import torch
from sklearn.model_selection import train_test_split

from dn3.configuratron import ExperimentConfig
from dn3.trainable.processes import StandardClassification
from dn3_ext import BENDRClassification

mne.set_log_level(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EEGデータの二値分類を行う転移学習プログラム")
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
    learning_rate = getattr(training_params, 'learning_rate', 1e-4)
    weight_decay = getattr(training_params, 'weight_decay', 1e-4)
    debug_subset_size = getattr(training_params, 'debug_subset_size', None)

    # 設定ファイルから事前学習済みモデルのパスを取得
    encoder_weights = experiment.encoder_weights
    context_weights = experiment.context_weights

    # データセットの構築
    dataset_name = next(iter(experiment.datasets))
    if args.debug:
        # デバッグモードのときは最初のデータセットのみを使用
        dataset_name = dataset_name[0]
    dataset_config = experiment.datasets[dataset_name]
    dataset = dataset_config.auto_construct_dataset()

    # データセットをトレーニングとバリデーションに分割
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # モデルの初期化
    model = BENDRClassification.from_dataset(dataset, multi_gpu=True)

    # 事前学習済みの重みをロード
    model.load_pretrained_modules(encoder_weights, context_weights, freeze_encoder=True)

    # 複数GPUの設定
    if torch.cuda.device_count() > 1:
        print("使用可能なGPU数:", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 損失関数とオプティマイザの設定
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # トレーニングプロセスの定義
    process = StandardClassification(model, loss=criterion, metrics=['accuracy'])
    process.set_optimizer(optimizer)

    # データローダーの作成
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # モデルの訓練
    process.fit(
        training_dataset=train_loader,
        validation_dataset=val_loader,
        epochs=args.epochs,
        num_workers=args.num_workers,
        device=device,
        verbose=True,
    )

    # テストデータでの評価（必要に応じて）
    if 'test' in experiment.datasets:
        test_dataset = experiment.datasets['test'].auto_construct_dataset()
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        test_results = process.evaluate(test_loader)
        print("Test Results:", test_results)
