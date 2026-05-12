"""
Configuration management for nanoGPT Extended.

Supports Python config files and command-line overrides. Config files are
executed as Python scripts in the context of the caller's global namespace,
allowing them to set and override any variable. CLI args of the form
``--key=value`` or ``--key value`` are then applied on top.

Usage
-----
In a training script::

    from configurator import get_config
    cfg = get_config()           # reads sys.argv automatically
    learning_rate = cfg['learning_rate']

Or the nanoGPT-style inline pattern (just import the module)::

    import configurator  # silently applies sys.argv overrides to globals

CLI usage::

    python train.py config/train_gpt2.py --learning_rate=3e-4 --n_layer=12
    python train.py config/train_gpt2.py --compile=False --wandb_log=True
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Optional YAML support (gracefully degraded when PyYAML is not installed)
# ---------------------------------------------------------------------------
try:
    import yaml as _yaml  # type: ignore
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------

def _coerce_value(raw: str) -> Any:
    """Convert a raw string CLI value to an appropriate Python type.

    Priority order:
    1. Python literals (``True``, ``False``, ``None``, numbers, lists, dicts)
    2. Plain string fallback
    """
    # Fast-path for booleans written as Python identifiers
    if raw == "True":
        return True
    if raw == "False":
        return False
    if raw == "None":
        return None

    # Try a safe literal evaluation (handles int, float, list, dict, tuple)
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        pass

    # Fall through to bare string
    return raw


def _cast_to_existing(value: Any, existing: Any) -> Any:
    """Cast *value* to match the type of *existing* when both are scalars.

    This ensures ``--dropout=0.1`` stays a float even when ``_coerce_value``
    would already produce one, but also handles ``--n_layer=12`` (int).
    """
    if existing is None or isinstance(existing, bool):
        # Booleans must be handled before int because bool is a subclass
        return value
    if isinstance(existing, int) and not isinstance(existing, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    if isinstance(existing, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return value


# ---------------------------------------------------------------------------
# Config-file loading
# ---------------------------------------------------------------------------

def override_from_config_file(
    filename: str | Path,
    namespace: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute *filename* as a Python (or YAML) config file.

    For Python files the file is executed with ``exec`` in a copy of
    *namespace*, so it can reference existing variables (e.g. ``n_embd =
    n_layer * 64``).  For YAML files the top-level keys are merged directly.

    Parameters
    ----------
    filename:
        Path to the config file.  Recognised extensions: ``.py``, ``.yaml``,
        ``.yml``.
    namespace:
        The current configuration dictionary.  **Modified in-place.**

    Returns
    -------
    Dict[str, Any]
        The updated *namespace*.

    Raises
    ------
    FileNotFoundError
        If *filename* does not exist.
    ImportError
        If a YAML file is given but PyYAML is not installed.
    ValueError
        If the file extension is not recognised.
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".py":
        with path.open("r", encoding="utf-8") as fh:
            source = fh.read()
        # Execute in a temporary dict that starts as a copy of the namespace
        # so that the config file can reference existing defaults.
        local_ns: Dict[str, Any] = dict(namespace)
        exec(compile(source, str(path), "exec"), local_ns)  # noqa: S102
        # Merge back: skip dunder names and module-level imports
        for k, v in local_ns.items():
            if not k.startswith("__"):
                namespace[k] = v

    elif suffix in {".yaml", ".yml"}:
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required to load YAML config files: "
                "pip install pyyaml"
            )
        with path.open("r", encoding="utf-8") as fh:
            data = _yaml.safe_load(fh)
        if data and isinstance(data, dict):
            namespace.update(data)

    else:
        raise ValueError(
            f"Unsupported config file extension '{suffix}'. "
            "Supported: .py, .yaml, .yml"
        )

    print(f"[configurator] Loaded config file: {path}", file=sys.stderr)
    return namespace


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def override_from_args(
    args: Sequence[str],
    namespace: Dict[str, Any],
) -> Dict[str, Any]:
    """Parse *args* (e.g. ``sys.argv[1:]``) and apply overrides to *namespace*.

    Accepted formats::

        --key=value
        --key value   (two-token form)
        --key         (bare flag, treated as True)

    The value string is coerced to a Python type via :func:`_coerce_value`.
    When a key already exists in *namespace* the value is additionally cast
    to match the existing type.

    Parameters
    ----------
    args:
        Sequence of argument strings (typically ``sys.argv[1:]`` minus config
        file paths which should have been consumed already).
    namespace:
        The current configuration dictionary.  **Modified in-place.**

    Returns
    -------
    Dict[str, Any]
        The updated *namespace*.
    """
    args = list(args)
    i = 0
    while i < len(args):
        arg = args[i]
        if not arg.startswith("--"):
            i += 1
            continue

        # Strip leading --
        arg = arg[2:]

        if "=" in arg:
            key, raw = arg.split("=", 1)
        else:
            key = arg
            # Peek ahead for a value token that is not itself a flag
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                i += 1
                raw = args[i]
            else:
                # Bare flag: treat as True
                raw = "True"

        value = _coerce_value(raw)
        if key in namespace:
            value = _cast_to_existing(value, namespace[key])
        namespace[key] = value
        i += 1

    return namespace


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def get_config(
    argv: Optional[List[str]] = None,
    defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a configuration dictionary from config files and CLI overrides.

    Parameters
    ----------
    argv:
        Argument list to parse.  Defaults to ``sys.argv[1:]``.
    defaults:
        Optional baseline configuration dictionary.  If *None* an empty dict
        is used (the caller must supply defaults, or use the configurator as a
        module against an existing globals() dict).

    Returns
    -------
    Dict[str, Any]
        Fully resolved configuration dictionary.

    Examples
    --------
    ::

        cfg = get_config()
        learning_rate = cfg['learning_rate']

    Typical ``sys.argv``::

        python train.py config/train_gpt2.py --learning_rate=3e-4

    The function will:

    1. Iterate over positional arguments (those not starting with ``--``)
       and treat each as a config file path.
    2. Apply ``--key=value`` overrides from the remaining arguments.
    """
    if argv is None:
        argv = sys.argv[1:]

    namespace: Dict[str, Any] = dict(defaults or {})

    # Separate config file paths from key=value overrides
    config_files: List[str] = []
    remaining: List[str] = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--"):
            remaining.append(arg)
            # If this is a two-token form (--key value), carry the value along
            if "=" not in arg and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                i += 1
                remaining.append(argv[i])
        else:
            config_files.append(arg)
        i += 1

    # 1. Load config files in order
    for cf in config_files:
        override_from_config_file(cf, namespace)

    # 2. Apply CLI overrides
    override_from_args(remaining, namespace)

    return namespace


# ---------------------------------------------------------------------------
# nanoGPT-style "import configurator" magic
# ---------------------------------------------------------------------------
# When a training script does ``import configurator`` at module level *after*
# defining its defaults as globals, this block will automatically apply
# sys.argv overrides to the caller's global namespace.  This preserves
# backward-compatibility with the original nanoGPT pattern.

def _apply_to_caller_globals() -> None:
    """Apply sys.argv config files and overrides to the *caller's* globals.

    This is called automatically when ``configurator`` is imported as a
    module (not via ``from configurator import get_config``).

    The function walks up the call stack to find the first frame that is
    *not* this module and treats that frame's ``f_globals`` dict as the
    configuration namespace.
    """
    import inspect

    # Find the first caller frame outside this module
    current_file = os.path.abspath(__file__)
    frame = inspect.currentframe()
    target_globals: Optional[Dict[str, Any]] = None
    try:
        while frame is not None:
            frame = frame.f_back
            if frame is None:
                break
            caller_file = os.path.abspath(frame.f_globals.get("__file__", ""))
            if caller_file != current_file:
                target_globals = frame.f_globals
                break
    finally:
        del frame  # avoid reference cycles

    if target_globals is None:
        return

    argv = sys.argv[1:]

    # Separate positional config files from flag overrides
    config_files: List[str] = []
    remaining: List[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--"):
            remaining.append(arg)
            if "=" not in arg and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                i += 1
                remaining.append(argv[i])
        else:
            config_files.append(arg)
        i += 1

    for cf in config_files:
        override_from_config_file(cf, target_globals)

    override_from_args(remaining, target_globals)


# Run the magic when this module is imported directly (not via ``from``).
# Guard with __name__ check so it does not fire when the file is executed
# directly as a script.
if __name__ != "__main__":
    _apply_to_caller_globals()


# ---------------------------------------------------------------------------
# CLI self-test / demonstration
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick smoke-test: print resolved config from CLI
    cfg = get_config()
    print("Resolved configuration:")
    for k, v in sorted(cfg.items()):
        print(f"  {k} = {v!r}")
