import torch
import argparse
import tqdm
import time
import utils

from dn3.configuratron import ExperimentConfig
from dn3.trainable.processes import StandardClassification
from dn3_ext import BENDRClassification, LinearHeadBENDR
from result_tracking import ThinkerwiseResultTracker

import mne
mne.set_log_level(False)

def fine_tune_bendr(args):
    # 実験設定をロード
    experiment = ExperimentConfig(args.ds_config)

    # 結果を追跡するインスタンスを初期化
    results = ThinkerwiseResultTracker() if args.results_filename else None

    # データセットごとにファインチューニングを実行
    for ds_name, ds in tqdm.tqdm(experiment.datasets.items(), total=len(experiment.datasets.items()), desc='Datasets'):
        added_metrics, retain_best, _ = utils.get_ds_added_metrics(ds_name, args.metrics_config)
        for fold, (training, validation, test) in enumerate(tqdm.tqdm(utils.get_lmoso_iterator(ds_name, ds))):
            
            # GPUメモリの状態を表示
            tqdm.tqdm.write(torch.cuda.memory_summary())

            # モデルの選択
            if args.model == utils.MODEL_CHOICES[0]:
                model = BENDRClassification.from_dataset(training, multi_gpu=args.multi_gpu)
            else:
                model = LinearHeadBENDR.from_dataset(training)

            # モデルの事前学習済み重みをロード
            if not args.random_init:
                model.load_pretrained_modules(experiment.encoder_weights, experiment.context_weights, freeze_encoder=args.freeze_encoder)

            # トレーニングプロセスを設定
            process = StandardClassification(model, metrics=added_metrics)
            process.set_optimizer(torch.optim.Adam(process.parameters(), ds.lr, weight_decay=0.01))

            # ファインチューニングを実行
            process.fit(training_dataset=training, validation_dataset=validation, warmup_frac=0.1, retain_best=retain_best, pin_memory=False, **ds.train_params)

            # 結果を保存
            if results:
                if isinstance(test, Thinker):
                    results.add_results_thinker(process, ds_name, test)
                else:
                    results.add_results_all_thinkers(process, ds_name, test, Fold=fold+1)
                results.to_spreadsheet(args.results_filename)

            # メモリ解放のための明示的なガベージコレクション
            del process
            del model
            torch.cuda.synchronize()
            time.sleep(10)

        # データセットごとの最終結果を保存
        if results:
            results.performance_summary(ds_name)
            results.to_spreadsheet(args.results_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BENDRモデルのファインチューニング")
    parser.add_argument('model', choices=utils.MODEL_CHOICES)
    parser.add_argument('--ds-config', default="configs/downstream.yml", help="DN3の設定ファイル")
    parser.add_argument('--metrics-config', default="configs/metrics.yml", help="評価指標の設定ファイル")
    parser.add_argument('--subject-specific', action='store_true', help="ターゲットの被験者に特化してファインチューニングを行う")
    parser.add_argument('--mdl', action='store_true', help="すべての追加データを使用してターゲット被験者のファインチューニングを行う")
    parser.add_argument('--freeze-encoder', action='store_true', help="エンコーダ部分を凍結するかどうか")
    parser.add_argument('--random-init', action='store_true', help="事前学習なしでランダム初期化したBENDRで比較する")
    parser.add_argument('--multi-gpu', action='store_true', help='複数GPUでBENDRを分散処理する')
    parser.add_argument('--num-workers', default=4, type=int, help='データローダーのワーカー数')
    parser.add_argument('--results-filename', default=None, help='結果を保存するスプレッドシートのファイル名')
    args = parser.parse_args()

    fine_tune_bendr(args)

