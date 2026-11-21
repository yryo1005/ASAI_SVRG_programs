"""
E5（実験計画8章）の問題の定義（データ，モデル，損失，評価指標）．

データとモデルは既存の実験のモジュールをそのまま用い，コピーや変更はしない．
- ImageWoof：`ex009_e3_imagewoof_walltime` の `data.py`，`model.py`（ResNet18，LayerNorm）．
- CIFAR-10：`ex002_e0_cifar10_alexnet_layernorm` の `data.py`，`model.py`（AlexNet，LayerNorm）．

2つの実験のモジュール名はどちらも `data`，`model` である．同じプロセスで両方を読み込めるように，
`importlib` でファイルのパスから別の名前（`e5_imagewoof_data` 等）を付けて読み込む．
"""

import functools
import importlib.util
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROGRAMS_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, os.path.join(_PROGRAMS_DIR, "common"))

DATASETS = ("imagewoof", "cifar10")
SOURCE_EXPERIMENTS = {
    "imagewoof": "ex009_e3_imagewoof_walltime",
    "cifar10": "ex002_e0_cifar10_alexnet_layernorm",
}
CONFIG_EXTRA = {
    "imagewoof": {"dataset": "ImageWoof (imagewoof2-320)", "model": "ResNet18LayerNorm"},
    "cifar10": {"dataset": "CIFAR-10", "model": "AlexNetCIFARNorm", "norm_type": "layernorm"},
}
N_TRAIN = {"imagewoof": 9025, "cifar10": 50000}
REG_LAMBDA = 5e-4


def load_source_module(dataset: str, name: str):
    """
    概要: 既存の実験のモジュール（`data` または `model`）を，データセットごとに別の名前で読み込む．
        一度読み込んだモジュールは `sys.modules` から返す．
    引数:
        dataset (str)．"imagewoof" または "cifar10"．
        name (str)．"data" または "model"．
    戻り値: module (module)．
    """
    module_name = f"e5_{dataset}_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = os.path.join(_PROGRAMS_DIR, SOURCE_EXPERIMENTS[dataset], f"{name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_model(dataset: str, seed: int):
    """
    概要: 既存の SGD と同じ方法で，指定したシードでモデルを初期化する．
    引数: dataset (str)．seed (int)．
    戻り値: model (torch.nn.Module)．
    """
    model_module = load_source_module(dataset, "model")
    if dataset == "imagewoof":
        return model_module.load_model(model_module.ResNet18LayerNorm, seed=seed)
    return model_module.load_model(model_module.AlexNetCIFARNorm, seed=seed, norm_type="layernorm")


def build_problem(dataset: str) -> dict:
    """
    概要: 問題の定義を辞書にまとめる．各プロセスで呼ぶ．
    引数: dataset (str)．"imagewoof" または "cifar10"．
    戻り値: problem (dict)．`load_dataloader(seed, batch_size, device)`，`build_model(seed)`，`loss_func`，
        `metrics_func`，`reg_lambda`，`prepare_cache`，`config_extra` を含む．
    """
    assert dataset in DATASETS
    data_module = load_source_module(dataset, "data")
    model_module = load_source_module(dataset, "model")
    return {
        "load_dataloader": data_module.load_dataloader,
        "prepare_cache": data_module.prepare_cache,
        "build_model": functools.partial(build_model, dataset),
        "loss_func": model_module.loss_func,
        "metrics_func": model_module.metrics_func,
        "reg_lambda": REG_LAMBDA,
        "config_extra": {**CONFIG_EXTRA[dataset], "source_experiment": SOURCE_EXPERIMENTS[dataset]},
    }
