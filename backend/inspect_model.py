"""
inspect_model.py
----------------
Run this ONCE to print the real F5-TTS DiT layer structure.
This tells us exactly which linear layers are safe for LoRA.

Usage:
    python inspect_model.py

Output: prints every nn.Linear with its full path and parent module type.
"""
import torch
import torch.nn as nn

def inspect_dit():
    from f5_tts.model import DiT
    from f5_tts.infer.utils_infer import load_model
    from cached_path import cached_path

    BASE_MODEL_HF = "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    VOCAB_HF      = "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt"

    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    model = load_model(
        model_cls=DiT, model_cfg=model_cfg,
        ckpt_path=str(cached_path(BASE_MODEL_HF)),
        mel_spec_type="vocos",
        vocab_file=str(cached_path(VOCAB_HF)),
        ode_method="euler", use_ema=True, device="cpu",
    )

    print("\n=== ALL nn.Linear layers in F5-TTS DiT ===\n")
    print(f"{'Full path':<60} {'Parent type':<30} {'Shape'}")
    print("-" * 110)

    # Build parent lookup
    parent_map = {}
    for name, module in model.named_modules():
        for child_name, _ in module.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_map[full] = type(module).__name__

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent = parent_map.get(name, "unknown")
            shape  = f"{module.in_features}→{module.out_features}"
            print(f"{name:<60} {parent:<30} {shape}")

    print("\n=== SAFE targets (copy these into trainer.py) ===\n")
    # A layer is safe if no ancestor is a ConvNeXt-type block
    safe = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        path_parts = name.split('.')
        ancestors = ['.'.join(path_parts[:i]) for i in range(1, len(path_parts))]
        ancestor_types = []
        for a in ancestors:
            for n2, m2 in model.named_modules():
                if n2 == a:
                    ancestor_types.append(type(m2).__name__)
        unsafe = any('conv' in t.lower() or 'convnext' in t.lower() for t in ancestor_types)
        if not unsafe:
            leaf = path_parts[-1]
            if leaf not in safe:
                safe.append(leaf)
    print(f"SAFE_LORA_TARGETS = {safe}")

if __name__ == "__main__":
    inspect_dit()