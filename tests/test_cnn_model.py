"""
programs/ex002_cifar10_cnn/model.py の単体テスト．
"""

import importlib.util
import os

import torch

# programs/ex001_mushroom_svrg/model.py と programs/ex002_cifar10_cnn/model.py は
# 同名（model.py）のため，sys.path経由の `import model` ではモジュール名が衝突する．
# importlib.util により一意なモジュール名で明示的に読み込む．
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "programs",
    "ex002_cifar10_cnn",
    "model.py",
)
_spec = importlib.util.spec_from_file_location("ex002_model", _MODEL_PATH)
_ex002_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ex002_model)

CNNModel = _ex002_model.CNNModel
load_model = _ex002_model.load_model
set_model_params = _ex002_model.set_model_params


def test_cnn_model_output_shape():
    """CNNModelの出力形状が (B, num_classes) であることを確認する．"""
    model = CNNModel(num_classes=10)
    x = torch.randn(4, 3, 32, 32)

    logits = model(x)

    assert logits.shape == (4, 10)


def test_load_model_is_reproducible_with_same_seed():
    """同じseedでload_modelを呼び出すと，同一の初期パラメータが得られることを確認する．"""
    model_a = load_model(CNNModel, seed=0)
    model_b = load_model(CNNModel, seed=0)

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(p_a, p_b)


def test_load_model_differs_across_seeds():
    """異なるseedでload_modelを呼び出すと，異なる初期パラメータが得られることを確認する．"""
    model_a = load_model(CNNModel, seed=0)
    model_b = load_model(CNNModel, seed=1)

    params_a = list(model_a.parameters())
    params_b = list(model_b.parameters())
    assert not torch.allclose(params_a[0], params_b[0])


def test_set_model_params_overwrites_values():
    """set_model_paramsが，与えたテンソル列でモデルのパラメータを正しく上書きすることを確認する．"""
    source_model = CNNModel()
    target_model = CNNModel()

    source_values = [p.detach().clone() for p in source_model.parameters()]
    set_model_params(target_model, source_values)

    for p_target, p_source in zip(target_model.parameters(), source_values):
        assert torch.allclose(p_target, p_source)
