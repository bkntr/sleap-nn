"""Tests for DINOv3 weight mapping for ConvNeXt."""

import torch

from sleap_nn.architectures.dinov3 import (
    map_dinov3_state_dict_to_torchvision,
)
from sleap_nn.architectures.convnext import ConvNeXtEncoder


def test_dinov3_key_mapping_and_load():
    """Map a minimal fake DINOv3-like state dict and load into encoder."""
    dinov3_sd = {}
    # Stem conv and norm
    dinov3_sd["downsample_layers.0.0.weight"] = torch.randn(96, 1, 4, 4)
    dinov3_sd["downsample_layers.0.1.weight"] = torch.randn(96)
    dinov3_sd["downsample_layers.0.1.bias"] = torch.randn(96)
    # First stage first block pieces
    dinov3_sd["stages.0.blocks.0.dwconv.weight"] = torch.randn(96, 1, 7, 7)
    dinov3_sd["stages.0.blocks.0.norm.weight"] = torch.randn(96)
    dinov3_sd["stages.0.blocks.0.norm.bias"] = torch.randn(96)
    dinov3_sd["stages.0.blocks.0.pwconv1.weight"] = torch.randn(384, 96, 1, 1)
    dinov3_sd["stages.0.blocks.0.pwconv1.bias"] = torch.randn(384)
    dinov3_sd["stages.0.blocks.0.pwconv2.weight"] = torch.randn(96, 384, 1, 1)
    dinov3_sd["stages.0.blocks.0.pwconv2.bias"] = torch.randn(96)
    dinov3_sd["stages.0.blocks.0.gamma"] = torch.randn(96)

    mapped = map_dinov3_state_dict_to_torchvision(dinov3_sd)
    # Ensure some expected keys are present
    assert "features.0.0.weight" in mapped
    assert "features.0.1.weight" in mapped
    assert "features.1.0.block.0.weight" in mapped  # dwconv
    assert "features.1.0.block.2.weight" in mapped  # pwconv1
    assert "features.1.0.block.4.weight" in mapped  # pwconv2
    assert "features.1.0.gamma" in mapped

    # Load into our encoder (strict=False) should succeed
    enc = ConvNeXtEncoder(
        blocks={"depths": [3, 3, 9, 3], "channels": [96, 192, 384, 768]},
        in_channels=1,
        stem_kernel=4,
        stem_stride=2,
    )
    enc.load_state_dict(mapped, strict=False)
