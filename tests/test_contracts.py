"""Interface contracts shared by train.py, configs, checkpoints and the model."""

import ast
import dataclasses
import glob
import inspect
import os
import subprocess
import sys

from model import GPTConfig
from optim.schedules import get_lr_schedule

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _train_ast():
    with open(os.path.join(ROOT, "train.py"), encoding="utf-8") as f:
        return ast.parse(f.read())


def _top_level_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _build_model_args_keys():
    for node in _train_ast().body:
        if isinstance(node, ast.FunctionDef) and node.name == "_build_model_args":
            call = node.body[-1].value
            return {kw.arg for kw in call.keywords}
    raise AssertionError("_build_model_args not found in train.py")


def test_build_model_args_keys_are_gptconfig_fields():
    fields = {f.name for f in dataclasses.fields(GPTConfig)}
    assert _build_model_args_keys() <= fields


def test_config_files_only_set_known_train_globals():
    known = _top_level_names(_train_ast())
    for path in glob.glob(os.path.join(ROOT, "config", "*.py")):
        with open(path, encoding="utf-8") as f:
            names = _top_level_names(ast.parse(f.read()))
        unknown = names - known
        assert not unknown, f"{os.path.basename(path)} sets unknown keys: {sorted(unknown)}"


def test_checkpoint_model_args_roundtrip():
    cfg = GPTConfig(
        n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100,
        attention_type="gqa", n_kv_heads=2, pos_embedding="rope", rope_theta=500.0,
        norm_type="rmsnorm", activation="swiglu", use_moe=True, num_experts=4,
        moe_aux_coef=0.02,
    )
    assert GPTConfig(**dataclasses.asdict(cfg)) == cfg


def test_moe_aux_coef_default():
    assert GPTConfig().moe_aux_coef == 0.01


def test_get_lr_schedule_signature():
    params = list(inspect.signature(get_lr_schedule).parameters)
    assert params[:5] == ["schedule_type", "learning_rate", "min_lr", "warmup_iters", "lr_decay_iters"]


def test_configurator_applies_overrides_on_import(tmp_path):
    cfg = tmp_path / "cfg.py"
    cfg.write_text("x = 2\n")
    script = tmp_path / "run.py"
    script.write_text("x = 1\nimport configurator\nprint(x)\n")
    env = dict(os.environ, PYTHONPATH=ROOT)
    out = subprocess.run(
        [sys.executable, str(script), str(cfg), "--x=5"],
        capture_output=True, text=True, env=env, check=True,
    ).stdout
    assert out.strip() == "5"
