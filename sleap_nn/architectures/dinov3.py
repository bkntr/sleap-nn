"""Utilities to load DINOv3 ConvNeXt weights into our ConvNeXt encoder.

This provides a best-effort key mapping from facebookresearch/dinov3 ConvNeXt
state_dicts to the torchvision-style ConvNeXt used by our ConvNeXtEncoder.

Notes:
- This loader is path-based. Provide a local .pth checkpoint from DINOv3.
- If some keys can't be mapped, they are skipped (strict=False load).
- Input normalization for DINOv3 typically expects ImageNet mean/std.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import os
import torch
from loguru import logger


def _infer_dinov3_ckpt_state(ckpt: object) -> Dict[str, torch.Tensor]:
    """Extract a state_dict from various DINOv3 checkpoint formats.

    Accepts:
    - raw state_dict (dict of tensors)
    - dict with 'model' or 'state_dict' field
    - torch.hub.load(model).state_dict()
    """
    if isinstance(ckpt, dict):
        # Try common keys first
        for key in ("model", "state_dict", "module", "net", "params"):
            if key in ckpt and isinstance(ckpt[key], dict):
                inner = ckpt[key]
                # Some wrappers store tensors under 'model'->'state_dict'
                if isinstance(inner, dict) and any(
                    isinstance(v, torch.Tensor) for v in inner.values()
                ):
                    return inner
        # Might already be a state_dict
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt  # type: ignore[return-value]
    raise ValueError("Unsupported DINOv3 checkpoint format: couldn't find state_dict.")


def _stage_index_to_features_idx(stage_idx: int) -> int:
    """Map DINOv3 stage index (0..3) to our ConvNeXtEncoder features index.

    Our features layout:
        0: stem (Conv2dNormActivation)
        1: stage0
        2: downsample1
        3: stage1
        4: downsample2
        5: stage2
        6: downsample3
        7: stage3
    """
    return 1 + 2 * stage_idx


def map_dinov3_state_dict_to_torchvision(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map DINOv3 ConvNeXt state_dict keys to torchvision ConvNeXtEncoder keys.

    This handles typical key patterns seen in DINOv3 convnext implementation:
    - downsample_layers.0.{conv,norm} -> features.0.{0/1}
    - downsample_layers.{1..3}.{norm,conv} -> features.{2,4,6}.{0/1}
    - stages.{i}.blocks.{j}.{dwconv,norm,pwconv1,pwconv2,gamma} ->
      features.{stage_features_idx}.{j}.block.{0(dwconv),1(norm),2(pw1),4(pw2)} or 'gamma'

    Returns a new dict containing only mapped keys.
    """
    mapped: Dict[str, torch.Tensor] = {}

    for k, v in state_dict.items():
        # Stem mapping: downsample_layers.0.0 = conv, .0.1 = norm
        if k.startswith("downsample_layers.0."):
            parts = k.split(".")  # [downsample_layers, 0, idx, ...]
            if len(parts) >= 3:
                idx = parts[2]
                suffix = ".".join(parts[3:]) if len(parts) > 3 else None
                if idx == "0":  # conv
                    # Our stem Conv2dNormActivation: conv at [0]
                    new_k = "features.0.0." + (suffix or "weight")
                    mapped[new_k] = v
                    continue
                if idx in ("1", "2"):  # norm (some impls use .1) safe-map to norm index 1
                    new_k = "features.0.1." + (suffix or "weight")
                    mapped[new_k] = v
                    continue

        # Downsample layers after stages: downsample_layers.{1..3}.{0:norm,1:conv}
        if k.startswith("downsample_layers.") and not k.startswith("downsample_layers.0."):
            parts = k.split(".")
            if len(parts) >= 4:
                ds_idx = int(parts[1])  # 1..3
                sub_idx = parts[2]  # 0 or 1
                suffix = ".".join(parts[3:])
                # Map to features.{2,4,6}
                features_idx = 2 * ds_idx
                if sub_idx == "0":  # norm
                    new_k = f"features.{features_idx}.0.{suffix}"
                    mapped[new_k] = v
                    continue
                if sub_idx == "1":  # conv
                    new_k = f"features.{features_idx}.1.{suffix}"
                    mapped[new_k] = v
                    continue

        # Stage blocks mapping
        if k.startswith("stages."):
            # stages.{i}.blocks.{j}.X.Y
            parts = k.split(".")
            # Guard for expected length
            if len(parts) >= 5 and parts[2] == "blocks":
                try:
                    stage_i = int(parts[1])
                    block_j = int(parts[3])
                except ValueError:
                    continue
                subpath = ".".join(parts[4:])
                features_idx = _stage_index_to_features_idx(stage_i)

                # Map known submodules
                if subpath.startswith("dwconv."):
                    new_k = f"features.{features_idx}.{block_j}.block.0." + subpath.split(".", 1)[1]
                    mapped[new_k] = v
                    continue
                if subpath.startswith("norm."):
                    new_k = f"features.{features_idx}.{block_j}.block.1." + subpath.split(".", 1)[1]
                    mapped[new_k] = v
                    continue
                if subpath.startswith("pwconv1."):
                    new_k = f"features.{features_idx}.{block_j}.block.2." + subpath.split(".", 1)[1]
                    mapped[new_k] = v
                    continue
                if subpath.startswith("pwconv2."):
                    # GELU at .block.3, so second pointwise conv is at .block.4
                    new_k = f"features.{features_idx}.{block_j}.block.4." + subpath.split(".", 1)[1]
                    mapped[new_k] = v
                    continue
                if subpath == "gamma" or subpath.startswith("gamma"):
                    new_k = f"features.{features_idx}.{block_j}.gamma"
                    mapped[new_k] = v
                    continue

    return mapped


def load_dinov3_convnext_weights(
    *,
    ckpt_path: str,
    map_only: bool = False,
) -> Dict[str, torch.Tensor]:
    """Load a DINOv3 ConvNeXt checkpoint and map it to our encoder state_dict.

    Args:
        ckpt_path: Local filesystem path to a DINOv3 ConvNeXt checkpoint (.pth).
        map_only: If True, don't attempt to load into a module—just return mapped dict.

    Returns:
        A state_dict compatible with ConvNeXtEncoder (subset of keys), to be loaded with strict=False.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"DINOv3 checkpoint not found at: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu")
    try:
        dinov3_sd = _infer_dinov3_ckpt_state(raw)
    except ValueError:
        # Try if raw itself is already the state_dict
        if isinstance(raw, dict) and any(isinstance(v, torch.Tensor) for v in raw.values()):
            dinov3_sd = raw  # type: ignore[assignment]
        else:
            raise

    mapped = map_dinov3_state_dict_to_torchvision(dinov3_sd)
    if not mapped:
        logger.warning("No DINOv3 keys could be mapped to our ConvNeXt encoder. Skipping load.")
    return mapped
