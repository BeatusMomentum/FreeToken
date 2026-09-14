"""GLM-5.3-Flash vision tower: shapes and host streaming on a tiny random tower, parity with the reference on a real checkpoint."""

from __future__ import annotations

import os

import pytest
import torch

from freetoken.models.glm5_next import Glm5NextVisionModel, VisionConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CHECKPOINT = os.environ.get("FREETOKEN_GLM53_MODEL", "")
needs_checkpoint = pytest.mark.skipif(not os.path.exists(os.path.join(CHECKPOINT, "config.json")), reason="FREETOKEN_GLM53_MODEL not set")


def _tiny_vc():
    return VisionConfig(
        hidden_size=64, depth=2, num_heads=4, intermediate_size=128, projection_intermediate_size=96, out_hidden_size=48,
        in_channels=3, patch_size=4, temporal_patch_size=2, spatial_merge_size=2, rms_norm_eps=1e-5, swiglu_limit=10.0,
        attention_bias=True,
    )


def _build(vc):
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            tower = Glm5NextVisionModel(vc)
    finally:
        torch.set_default_dtype(torch.float32)
    for p in tower.state_dict().values():
        p.normal_(0, 0.02)
    return tower


def test_tiny_tower_merges_four_patches_per_token():
    tower = _build(_tiny_vc())
    feature = torch.randn(16 + 64, 3 * 2 * 4 * 4, dtype=torch.bfloat16, device="cuda")
    out = tower.forward(feature, [[1, 4, 4], [1, 8, 8]])
    assert out.shape == (4 + 16, 48) and out.dtype == torch.bfloat16
    keys = tower.state_dict()
    assert keys["blocks.0.attn.qkv.bias"].shape == (192,) and keys["merger.post_projection_norm.bias"].shape == (48,)
    assert "merger.gate_proj.bias" not in keys and keys["downsample.weight"].shape == (48, 64, 2, 2)


def test_host_streamed_weights_compute_the_same_output():
    tower = _build(_tiny_vc())
    feature = torch.randn(64, 3 * 2 * 4 * 4, dtype=torch.bfloat16, device="cuda")
    resident = tower.forward(feature, [[1, 8, 8]])
    keys = tower.state_dict()
    tower.place_weights("host")
    assert tower.state_dict().keys() == keys.keys()
    assert tower.state_dict()["blocks.0.attn.qkv.weight"].device.type == "cpu"
    assert tower.state_dict()["merger.proj.weight"].device.type == "cuda"
    for _ in range(2):  # a second forward reuses the staging buffers behind the release events
        assert torch.equal(tower.forward(feature, [[1, 8, 8]]), resident)
    tower.place_weights("gpu")
    assert tower.state_dict()["blocks.0.attn.qkv.weight"].device.type == "cuda"
    assert torch.equal(tower.forward(feature, [[1, 8, 8]]), resident)


def _textured_image(width, height):
    """Gradient, shapes and mild noise: on flat colour fields the bf16 reference itself drifts far from its fp32 result."""
    from PIL import Image, ImageDraw

    xs = torch.linspace(0, 255, width).expand(height, width)
    ys = torch.linspace(0, 255, height)[:, None].expand(height, width)
    rgb = torch.stack([xs, ys, 127 + (ys - xs) / 2], dim=-1)
    rgb = rgb + torch.randn(height, width, 3, generator=torch.Generator().manual_seed(0)) * 12
    img = Image.fromarray(rgb.clamp(0, 255).to(torch.uint8).numpy())
    draw = ImageDraw.Draw(img)
    draw.ellipse((100, 60, 340, 300), fill=(255, 215, 0))
    draw.rectangle((40, 340, 400, 380), fill=(220, 20, 60))
    return img


@pytest.mark.needs_weights
@needs_checkpoint
def test_tower_matches_the_reference_on_a_checkpoint():
    """The bf16 tower stays within the reference's own bf16 noise around its fp32 result."""
    import json

    from safetensors import safe_open
    from transformers import AutoConfig, AutoImageProcessor
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel as HFVision

    from freetoken.mm.config import MultimodalConfig
    from freetoken.mm.processors.glm5_next import Glm5NextMMProcessor
    from freetoken.models.glm5_next import parse_config

    torch.set_float32_matmul_precision("highest")
    hf_config = AutoConfig.from_pretrained(CHECKPOINT)
    hf_config.vision_config._attn_implementation = "sdpa"
    index = json.load(open(os.path.join(CHECKPOINT, "model.safetensors.index.json")))["weight_map"]
    weights = {}
    for name, file in index.items():
        if name.startswith("model.visual."):
            with safe_open(os.path.join(CHECKPOINT, file), "pt") as f:
                weights[name[len("model.visual.") :]] = f.get_tensor(name)

    def reference(dtype):
        # the default dtype sets the parameters; the rotary table is built fp32 regardless, as from_pretrained keeps it
        torch.set_default_dtype(dtype)
        try:
            vision = HFVision(hf_config.vision_config).to("cuda").eval()
        finally:
            torch.set_default_dtype(torch.float32)
        vision.load_state_dict({k: v.to(dtype) for k, v in weights.items()}, strict=True)
        return lambda pixels, grid: vision(pixels.to(dtype), grid_thw=grid).pooler_output

    config = parse_config(hf_config)
    tower = _build(config.vision_config)
    tower.load_state_dict({k: v.cuda() for k, v in weights.items()})

    img = _textured_image(448, 448)
    out = AutoImageProcessor.from_pretrained(CHECKPOINT)(images=img, return_tensors="pt")
    pixels, grid = out["pixel_values"].cuda(), out["image_grid_thw"].cuda()
    with torch.inference_mode():
        ref32 = reference(torch.float32)(pixels, grid)
        ref16 = reference(torch.bfloat16)(pixels, grid).float()
        (item,) = Glm5NextMMProcessor(hf_config, CHECKPOINT, MultimodalConfig()).process([img])
        ours = tower.forward(item.feature, [item.grid_thw]).float()
    assert ours.shape == ref32.shape == (item.grid_thw[1] * item.grid_thw[2] // 4, config.hidden_size)
    cos = torch.nn.functional.cosine_similarity
    noise, error = (ref32 - ref16).norm(), (ref32 - ours).norm()
    print(f"glm5_next tower: ours vs fp32 mean cos {cos(ref32, ours, dim=-1).mean():.5f}, error/noise {error / noise:.3f}")
    assert cos(ref32, ours, dim=-1).mean() > 0.99
    assert error <= 1.2 * noise
