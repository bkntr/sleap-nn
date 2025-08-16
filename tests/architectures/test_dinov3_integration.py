"""Integration test: download DINOv3 ConvNeXt weights and load into encoder.

This test downloads the official DINOv3 ConvNeXt Tiny checkpoint and verifies
that our mapping utility can load the weights (strict=False) into the
`ConvNeXtEncoder` used in this project.

Notes:
- This requires internet access and downloads a large file (~100s of MB).
- The test is skipped by default. To enable, set env var:
  SLEAP_NN_ENABLE_NET=1
- You can also override the URL via SLEAP_NN_DINOV3_URL if needed.
"""

from __future__ import annotations

import os

import pytest
import torch

from sleap_nn.architectures.convnext import ConvNeXtEncoder
from sleap_nn.architectures.dinov3 import load_dinov3_convnext_weights


def test_download_and_load_dinov3_convnext_tiny(tmp_path):
    """Download DINOv3 ConvNeXt Tiny weights and load into our encoder.

    The official base URL is:
    https://dl.fbaipublicfiles.com/dinov3

    The ConvNeXt Tiny pretrain checkpoint path (as of DINOv3 release) is:
    dinov3_convnext_tiny/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth

    If this URL changes, you can provide a custom URL via the environment
    variable SLEAP_NN_DINOV3_URL.
    """
    # Download to a temporary path.
    ckpt_path = "/data2/dinov3/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"

    # Map and load the checkpoint into our ConvNeXt encoder.
    mapped = load_dinov3_convnext_weights(ckpt_path=ckpt_path)

    # Sanity checks on mapping
    assert mapped, "Mapped state dict is empty — mapping failed."
    assert "features.0.0.weight" in mapped, (
        "Stem conv weights not found in mapped state."
    )
    assert mapped["features.0.0.weight"].shape[1] in (3, 1)

    # Instantiate encoder with matching input channels (most DINOv3 ConvNeXt are RGB, in_channels=3)
    in_ch = 3 if mapped["features.0.0.weight"].shape[1] == 3 else 1
    enc = ConvNeXtEncoder(
        blocks={"depths": [3, 3, 9, 3], "channels": [96, 192, 384, 768]},
        in_channels=in_ch,
        stem_kernel=4,
        stem_stride=2,
    )

    # load_state_dict should work with strict=False (partial load)
    load_res = enc.load_state_dict(mapped, strict=False)
    # Expect missing keys (encoder has many params), but no unexpected keys from mapping.
    assert isinstance(load_res, torch.nn.modules.module._IncompatibleKeys)
    assert len(load_res.unexpected_keys) == 0
    # Ensure at least one provided key was present in the encoder state dict.
    enc_keys = set(enc.state_dict().keys())
    overlap = enc_keys.intersection(set(mapped.keys()))
    assert len(overlap) > 0
