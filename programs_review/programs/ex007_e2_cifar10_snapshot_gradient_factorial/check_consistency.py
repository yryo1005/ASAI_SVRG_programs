"""
E2 の本実行の前に行う一致の確認（`実験指示E2.md` 3節）の結果を Markdown にまとめるスクリプト．

- `e0`：E2 のコードで実行した短い試行（`SVRG`，`NFG_SVRG`，`ASAI_SVRG`，シード0，3エポック）を，E0 のログと比較する．
  `compare_with_existing_logs.py` の表（共通する先頭の評価点の最大誤差）に加えて，評価点ごとの値と差の表を作り，
  0エポック目（初期値と全勾配の計算だけの値）と1〜3エポック目を区別して判定する．
  `--same-gpu-root` を与えた場合は，E0 の既存のコード（`ex002_e0_cifar10_alexnet_layernorm/train.py`）を同じ GPU で実行した
  結果との比較（ビット単位で一致するか）も加える．`--precision-note` を与えた場合は，`precision` の結果を引用する．
- `precision`：初期点 $ {\\mathbf z}_{0} $（シード0）の全勾配のノルムの2乗と学習損失を，TF32 あり（PyTorch の既定）・TF32 なし（FP32）・
  float64 の3通りで計算し，GPU による丸め誤差の大きさを示す．
- `omega`：$ \\omega $ の記録を有効にした試行と無効にした試行（`ASAI_SVRG`，シード0，2エポック）を比較し，
  `omega_*` と `elapsed_time` 以外の全ての記録がビット単位で一致するかを確認する．

比較先のパスはコマンドライン引数で受け取る．

実行例（`programs_review/programs` で実行する）:
    python ex007_e2_cifar10_snapshot_gradient_factorial/check_consistency.py e0 \\
        --new-root ../outputs/_verify/ex007 --existing-root ../outputs/ex002_e0_cifar10_alexnet_layernorm \\
        --hp-map inner_snapshot_lr0.001_bs128_lambda0.0005_epochs3=inner_snapshot_lr0.001_bs128_lambda0.0005_epochs48 \\
        --output ../outputs/ex007_e2_cifar10_snapshot_gradient_factorial/consistency_with_e0.md
    python ex007_e2_cifar10_snapshot_gradient_factorial/check_consistency.py omega \\
        --on-root ../outputs/_verify/ex007_omega_on --off-root ../outputs/_verify/ex007_omega_off \\
        --output ../outputs/ex007_e2_cifar10_snapshot_gradient_factorial/consistency_omega.md
"""

import argparse
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROGRAMS_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROGRAMS_DIR)

from compare_with_existing_logs import compare_run, format_markdown, iterate_pairs, load_log  # noqa: E402

PER_EPOCH_KEYS = ("oracle_calls", "train_loss", "test_accuracy", "test_correct", "approx_error",
                  "full_grad_norm_sq", "approx_cos")
EPOCH0_REL_TOL = 1e-5
SKIPPED_IN_OMEGA_CHECK = ("elapsed_time", "omega_snapshot", "omega_average")


def _rel_diff(new: float, old: float) -> float:
    """
    概要: 相対差 $ |a-b|/\\max(|b|, \\mathrm{tiny}) $ を返す．両方とも NaN の場合は 0．
    引数: new (float)，old (float)．
    戻り値: value (float)．
    """
    if np.isnan(new) and np.isnan(old):
        return 0.0
    return abs(new - old) / max(abs(old), np.finfo(np.float64).tiny)


def per_epoch_table(new_log: dict, old_log: dict, keys=PER_EPOCH_KEYS) -> tuple:
    """
    概要: 共通する評価点ごとに，新旧の値・絶対差・相対差の表を作る．
    引数:
        new_log (dict)．新しい実装の `log.json`．
        old_log (dict)．比較先の `log.json`．
        keys (tuple of str)．比較する指標．
    戻り値:
        text (str)．Markdown の表．
        epoch0_max_rel (float)．0エポック目の `oracle_calls` 以外の相対差の最大値．
        oracle_exact (bool)．`oracle_calls` が全ての評価点で完全に一致したか．
    """
    length = min(len(new_log["epoch"]), len(old_log["epoch"]))
    lines = ["| エポック | 指標 | 新しい実装 | E0 のログ | 絶対差 | 相対差 |", "|---:|---|---:|---:|---:|---:|"]
    epoch0_max_rel, oracle_exact = 0.0, True
    for epoch in range(length):
        for key in keys:
            if key not in new_log or key not in old_log:
                continue
            new, old = float(new_log[key][epoch]), float(old_log[key][epoch])
            rel = _rel_diff(new, old)
            if key == "oracle_calls":
                oracle_exact &= new == old
            elif epoch == 0 and key != "approx_cos":
                epoch0_max_rel = max(epoch0_max_rel, rel)
            lines.append(f"| {epoch} | {key} | {new:.10g} | {old:.10g} | {abs(new - old):.3e} | {rel:.3e} |")
    return "\n".join(lines) + "\n", epoch0_max_rel, oracle_exact


def _environment_line(run_dir: str) -> str:
    """
    概要: `config.json` の実行環境を1行の文字列にする．
    引数: run_dir (str)．試行のディレクトリ．
    戻り値: text (str)．
    """
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        return "（config.json なし）"
    with open(path) as f:
        env = json.load(f).get("environment", {})
    return "，".join(f"{k}={v}" for k, v in env.items())


def check_e0(args) -> str:
    """
    概要: E0 のログとの一致の確認の結果を Markdown にする．
    引数: args (argparse.Namespace)．`new_root`，`existing_root`，`hp_map`．
    戻り値: text (str)．
    """
    hp_map = dict(item.split("=", 1) for item in args.hp_map)
    pairs = list(iterate_pairs(args.new_root, args.existing_root, hp_map))
    rows = [(method, seed, key, result) for method, _, seed, new_dir, old_dir in pairs
            for key, result in compare_run(new_dir, old_dir).items()]
    sections = ["# E2：E0 のログとの一致の確認\n",
                f"- 新しい実装：`{os.path.relpath(args.new_root)}`（E2 の `train.py`，3エポック）",
                f"- 比較先：`{os.path.relpath(args.existing_root)}`（E0，48エポック．先頭の4点を比較）",
                f"- 条件名の対応：{', '.join(f'`{k}` → `{v}`' for k, v in hp_map.items())}",
                f"- 判定：0エポック目は `oracle_calls` 以外（`approx_cos` を除く）の相対差が {EPOCH0_REL_TOL:.0e} 以下，"
                "`oracle_calls` は全ての評価点で完全一致であれば合格とする．1〜3エポック目は差の大きさを記録する．\n",
                "## `compare_with_existing_logs.py` の結果（共通する先頭の評価点の最大誤差）\n",
                format_markdown(rows),
                f"比較した系列数: {len(rows)}，全て完全一致: {all(r['exact'] for *_, r in rows)}\n",
                "## 判定\n",
                "| 手法 | シード | 0エポック目の最大相対差 | 0エポック目の判定 | `oracle_calls` の完全一致 | "
                "1〜3エポック目の `train_loss` の最大相対差 | 1〜3エポック目の `test_accuracy` の最大絶対差 |",
                "|---|---:|---:|---|---|---:|---:|"]
    details = []
    for method, _, seed, new_dir, old_dir in pairs:
        new_log, old_log = load_log(os.path.join(new_dir, "log.json")), load_log(os.path.join(old_dir, "log.json"))
        table, epoch0_rel, oracle_exact = per_epoch_table(new_log, old_log)
        n = min(len(new_log["epoch"]), len(old_log["epoch"]))
        loss_rel = max((_rel_diff(new_log["train_loss"][e], old_log["train_loss"][e]) for e in range(1, n)), default=0.0)
        acc_abs = max((abs(new_log["test_accuracy"][e] - old_log["test_accuracy"][e]) for e in range(1, n)), default=0.0)
        sections.append(f"| {method} | {seed} | {epoch0_rel:.3e} | {'合格' if epoch0_rel <= EPOCH0_REL_TOL else '不合格'} | "
                        f"{'○' if oracle_exact else '×'} | {loss_rel:.3e} | {acc_abs:.4f} |")
        details.append(f"### {method}，シード {seed}\n\n新しい実装の環境：{_environment_line(new_dir)}\n\n"
                       f"E0 の環境：{_environment_line(old_dir)}\n\n{table}")
    if args.same_gpu_root:
        same_rows = [(method, seed, key, result)
                     for method, _, seed, new_dir, old_dir in iterate_pairs(args.new_root, args.same_gpu_root,
                                                                            {k: k for k in hp_map})
                     for key, result in compare_run(new_dir, old_dir).items()]
        sections += ["\n## 同じ GPU で実行した E0 の既存のコードとの比較\n",
                     f"比較先：`{os.path.relpath(args.same_gpu_root)}`（`ex002_e0_cifar10_alexnet_layernorm/train.py` を変更せずに，"
                     "このPCで同じ条件（シード0，3エポック）で実行した結果）．\n",
                     format_markdown(same_rows),
                     f"比較した系列数: {len(same_rows)}，全て完全一致: {all(r['exact'] for *_, r in same_rows)}\n"]
    if args.precision_note and os.path.exists(args.precision_note):
        with open(args.precision_note) as f:
            sections += ["\n## GPU の丸め誤差の大きさ（`check_consistency.py precision`）\n", f.read()]
    sections.append("\n## 評価点ごとの値\n")
    sections.extend(details)
    return "\n".join(sections)


def check_precision(args) -> str:
    """
    概要: 初期点 $ {\\mathbf z}_{0} $（シード0）における全勾配のノルムの2乗と学習損失を，TF32 あり・TF32 なし・float64 で計算し，
        float64 に対する相対差を表にする．E0 のログの値を与えた場合は，それも float64 と比べる．
    引数: args (argparse.Namespace)．`reference`（"名前=値" の文字列のリスト）．
    戻り値: text (str)．Markdown．
    """
    import torch

    sys.path.insert(0, _THIS_DIR)
    sys.path.insert(0, os.path.join(_PROGRAMS_DIR, "common"))
    import importlib.util

    from vr_image_training import compute_full_gradient_and_metrics

    module_spec = importlib.util.spec_from_file_location("ex007_train", os.path.join(_THIS_DIR, "train.py"))
    e2_train = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(e2_train)
    problem = e2_train.build_problem()
    device = torch.device("cuda")
    default_tf32 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)

    def full_grad_norm(tf32: bool, dtype) -> tuple:
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cuda.matmul.allow_tf32 = tf32 and default_tf32[1]
        train_loader, _ = problem["load_dataloader"](seed=0, batch_size=128, device=device)
        train_loader.inputs = train_loader.inputs.to(dtype)
        model = problem["build_model"](0).to(device=device, dtype=dtype)
        grads, metrics = compute_full_gradient_and_metrics(model, train_loader, problem["loss_func"],
                                                           problem["metrics_func"], problem["reg_lambda"])
        return sum(torch.sum(g.double() ** 2).item() for g in grads), metrics["loss"]

    results = {"TF32 あり（既定：cudnn.allow_tf32=True）": full_grad_norm(True, torch.float32),
               "TF32 なし（FP32）": full_grad_norm(False, torch.float32),
               "float64": full_grad_norm(False, torch.float64)}
    ref_norm, ref_loss = results["float64"]
    lines = [f"GPU：{torch.cuda.get_device_name(0)}，cuDNN {torch.backends.cudnn.version()}，torch {torch.__version__}．"
             f"PyTorch の既定は cudnn.allow_tf32={default_tf32[0]}，matmul.allow_tf32={default_tf32[1]} である．\n",
             "| 計算 | $ \\|\\nabla f({\\mathbf z}_{0})\\|^{2} $ | float64 との相対差 | 学習損失 $ f({\\mathbf z}_{0}) $ | float64 との相対差 |",
             "|---|---:|---:|---:|---:|"]
    for name, (norm, loss) in results.items():
        lines.append(f"| {name} | {norm:.10f} | {abs(norm - ref_norm) / ref_norm:.3e} | {loss:.10f} | "
                     f"{abs(loss - ref_loss) / ref_loss:.3e} |")
    for item in args.reference or []:
        name, value = item.split("=", 1)
        value = float(value)
        lines.append(f"| {name}（ログの値） | {value:.10f} | {abs(value - ref_norm) / ref_norm:.3e} | － | － |")
    return "\n".join(lines) + "\n"


def check_omega(args) -> str:
    """
    概要: $ \\omega $ の記録の有無による差の確認の結果を Markdown にする．
    引数: args (argparse.Namespace)．`on_root`，`off_root`．
    戻り値: text (str)．
    """
    sections = ["# E2：$ \\omega $ の記録が学習に影響しないことの確認\n",
                f"- ω の記録あり：`{os.path.relpath(args.on_root)}`",
                f"- ω の記録なし（`--no-omega`）：`{os.path.relpath(args.off_root)}`",
                f"- `{'`，`'.join(SKIPPED_IN_OMEGA_CHECK)}` 以外の全ての記録がビット単位で一致するかを確認する．\n",
                "| 手法 | 条件 | シード | 指標 | 点数 | ビット単位で一致 | 最大絶対差 |", "|---|---|---:|---|---:|---|---:|"]
    all_equal, n_series = True, 0
    for method in sorted(os.listdir(args.on_root)):
        for hp in sorted(os.listdir(os.path.join(args.on_root, method))):
            for seed in sorted(os.listdir(os.path.join(args.on_root, method, hp))):
                on_path = os.path.join(args.on_root, method, hp, seed, "log.json")
                off_path = os.path.join(args.off_root, method, hp, seed, "log.json")
                if not (os.path.exists(on_path) and os.path.exists(off_path)):
                    continue
                on_log, off_log = load_log(on_path), load_log(off_path)
                for key in on_log:
                    if key in SKIPPED_IN_OMEGA_CHECK:
                        continue
                    a = np.asarray(on_log[key], dtype=np.float64)
                    b = np.asarray(off_log[key], dtype=np.float64)
                    equal = a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
                    all_equal &= equal
                    n_series += 1
                    max_abs = float(np.nanmax(np.abs(a - b))) if a.shape == b.shape and a.size else float("nan")
                    sections.append(f"| {method} | {hp} | {seed} | {key} | {a.size} | {'○' if equal else '×'} | {max_abs:.3e} |")
                sections.append(f"| {method} | {hp} | {seed} | omega_snapshot（記録あり） | {len(on_log['omega_snapshot'])} | "
                                f"{', '.join(f'{v:.6g}' for v in on_log['omega_snapshot'])} | － |")
                sections.append(f"| {method} | {hp} | {seed} | omega_snapshot（記録なし） | {len(off_log['omega_snapshot'])} | "
                                f"{', '.join(f'{v:.6g}' for v in off_log['omega_snapshot'])} | － |")
    sections.append(f"\n比較した系列数: {n_series}，全てビット単位で一致: {all_equal}\n")
    return "\n".join(sections)


def parse_args() -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．
    引数: なし
    戻り値: args (argparse.Namespace)．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_e0 = sub.add_parser("e0", help="E0 のログとの一致の確認")
    p_e0.add_argument("--new-root", required=True)
    p_e0.add_argument("--existing-root", required=True)
    p_e0.add_argument("--hp-map", nargs="+", required=True, help="新しい条件名=E0 の条件名")
    p_e0.add_argument("--same-gpu-root", default=None, help="同じ GPU で E0 の既存のコードを実行した結果")
    p_e0.add_argument("--precision-note", default=None, help="`precision` の出力（Markdown）のパス")
    p_e0.add_argument("--output", required=True)
    p_pr = sub.add_parser("precision", help="初期点の全勾配の丸め誤差の大きさ")
    p_pr.add_argument("--reference", nargs="*", help="比較するログの値（名前=値）")
    p_pr.add_argument("--output", required=True)
    p_om = sub.add_parser("omega", help="ω の記録の有無による差の確認")
    p_om.add_argument("--on-root", required=True)
    p_om.add_argument("--off-root", required=True)
    p_om.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    """
    概要: 指定した確認を行い，結果を表示・保存する．
    引数: なし
    戻り値: なし
    """
    args = parse_args()
    commands = {"e0": check_e0, "omega": check_omega, "precision": check_precision}
    text = commands[args.command](args)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
