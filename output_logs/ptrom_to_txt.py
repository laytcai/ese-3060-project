#!/usr/bin/env python3
"""
Convert .ptrom torch logs inside output_logs/ into readable .txt files.

Each .ptrom file is expected to be a torch.save()'d dict containing:
  - code: the python source that produced the log
  - accs: a 1D tensor of accuracies

The script tries to use torch if it is installed. If not, it falls back to a
small stdlib-only loader that understands the storage format used here.
"""

from __future__ import annotations

import argparse
import array
import io
import pickle
import sys
import types
import zipfile
from math import prod
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_with_torch(path: Path) -> Dict:
    import torch

    return torch.load(path, map_location="cpu")


def _load_without_torch(path: Path) -> Dict:
    """Minimal loader for these specific .ptrom files without torch installed."""

    class FloatStorage:
        def __init__(self, key: str, location: str, data: bytes):
            self.key = key
            self.location = location
            self.data = data

        def tolist(self) -> List[float]:
            arr = array.array("f")
            arr.frombytes(self.data)
            return arr.tolist()

    class FakeTensor:
        def __init__(
            self,
            storage: FloatStorage,
            offset: int,
            size: Tuple[int, ...],
            stride: Tuple[int, ...],
            requires_grad: bool,
            backward_hooks=None,
        ):
            self.storage = storage
            self.offset = offset
            self.size = size
            self.stride = stride
            self.requires_grad = requires_grad
            self.backward_hooks = backward_hooks

        def tolist(self) -> List[float]:
            values = self.storage.tolist()
            total = prod(self.size) if self.size else 0
            start = self.offset
            end = start + total
            return values[start:end]

    def rebuild_tensor_v2(storage, offset, size, stride, requires_grad, backward_hooks):
        return FakeTensor(storage, offset, size, stride, requires_grad, backward_hooks)

    # Temporarily register fake modules so pickle can resolve torch symbols.
    original_modules = {name: sys.modules.get(name) for name in ("torch", "torch._utils")}
    sys.modules["torch"] = types.SimpleNamespace(FloatStorage=FloatStorage)
    sys.modules["torch._utils"] = types.SimpleNamespace(_rebuild_tensor_v2=rebuild_tensor_v2)

    try:
        with zipfile.ZipFile(path) as zf:
            data_bytes = zf.read("log/data.pkl")
            storage_files = {
                name.rsplit("/", 1)[-1]: zf.read(name)
                for name in zf.namelist()
                if name.startswith("log/data/") and name != "log/data.pkl"
            }

            def persistent_load(pid):
                if isinstance(pid, tuple) and pid[0] == "storage":
                    _, storage_type, key, location, *rest = pid
                    data = storage_files.get(str(key), b"")
                    return FloatStorage(key, location, data)
                raise RuntimeError(f"unsupported persistent id: {pid}")

            unpickler = pickle.Unpickler(io.BytesIO(data_bytes))
            unpickler.persistent_load = persistent_load
            return unpickler.load()
    finally:
        # Restore module table to avoid side effects.
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_log(path: Path) -> Dict:
    """Load a .ptrom log with torch when available, otherwise use the fallback."""
    try:
        return load_with_torch(path)
    except ModuleNotFoundError:
        return _load_without_torch(path)


def summarize_accs(accs: Iterable[float]) -> Dict[str, float]:
    values = list(accs)
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5
    return {"count": len(values), "mean": mean, "std": std, "min": min(values), "max": max(values)}


def render_text(source: Path, code: str, accs: List[float]) -> str:
    stats = summarize_accs(accs)
    lines = [
        f"source: {source.name}",
        f"entries: {stats['count']}",
        f"mean: {stats['mean']:.4f}    std: {stats['std']:.4f}    min: {stats['min']:.4f}    max: {stats['max']:.4f}",
        "",
    ]
    if accs:
        lines.append("accuracies:")
        for idx, val in enumerate(accs, 1):
            lines.append(f"  {idx:3d}: {val:.6f}")
        lines.append("")
    lines.append("code:")
    lines.append(code.rstrip())
    lines.append("")  # final newline
    return "\n".join(lines)


def iter_targets(paths: List[str]) -> List[Path]:
    targets: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            targets.extend(sorted(path.glob("*.ptrom")))
        elif path.suffix == ".ptrom" and path.exists():
            targets.append(path)
    return targets


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert .ptrom logs to .txt files.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["output_logs"],
        help="Directories or .ptrom files to convert (default: output_logs).",
    )
    args = parser.parse_args(argv)

    targets = iter_targets(args.paths)
    if not targets:
        print("No .ptrom files found.", file=sys.stderr)
        return 1

    for src in targets:
        try:
            raw = load_log(src)
            code = raw.get("code", "")
            accs_obj = raw.get("accs")
            if accs_obj is None:
                accs: List[float] = []
            elif hasattr(accs_obj, "tolist"):
                accs = list(accs_obj.tolist())
            else:
                accs = list(accs_obj)

            text = render_text(src, code, accs)
            dest = src.with_suffix(".txt")
            dest.write_text(text, encoding="utf-8")
            print(f"Wrote {dest}")
        except Exception as exc:  # pragma: no cover - conversion should be best-effort
            print(f"Failed to convert {src}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
