#!/usr/bin/env python3
"""
Test inventory conditioning of a trained Diamond BC checkpoint.

Creates synthetic scenarios with specific inventory states and checks
if the model predicts the expected crafting/equip actions.

python scripts/test_inventory_conditioning.py --checkpoint checkpoints/bc_diamond_action_inven1/bc_diamond_epoch028.ckpt
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from algorithms.foundation_dagger.diamond_policy import MineRLPolicy, MineRLPolicyConfig


# Inventory item order (from MineRLActionVocab.INVENTORY_ITEMS)
INVENTORY_ITEMS = [
    "coal", "cobblestone", "crafting_table", "dirt", "furnace",
    "iron_axe", "iron_ingot", "iron_ore", "iron_pickaxe", "log",
    "planks", "stick", "stone", "stone_axe", "stone_pickaxe",
    "torch", "wooden_axe", "wooden_pickaxe"
]
ITEM_TO_IDX = {name: i for i, name in enumerate(INVENTORY_ITEMS)}

# Action vocabularies
CRAFT_VOCAB = ["none", "crafting_table", "planks", "stick", "torch"]
NEARBY_CRAFT_VOCAB = ["none", "furnace", "iron_axe", "iron_pickaxe", "stone_axe", "stone_pickaxe", "wooden_pickaxe"]
EQUIP_VOCAB = ["none", "iron_axe", "iron_pickaxe", "stone_axe", "stone_pickaxe", "wooden_pickaxe"]
PLACE_VOCAB = ["none", "cobblestone", "crafting_table", "dirt", "furnace", "stone", "torch"]


def make_inventory(**items) -> np.ndarray:
    """Create inventory array from item counts."""
    inv = np.zeros(len(INVENTORY_ITEMS), dtype=np.int32)
    for name, count in items.items():
        if name in ITEM_TO_IDX:
            inv[ITEM_TO_IDX[name]] = count
    return inv


def make_dummy_frame(h=64, w=64) -> np.ndarray:
    """Create a dummy frame (black with some noise)."""
    return np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)


def test_scenario(
    model: MineRLPolicy,
    device: torch.device,
    name: str,
    inventory: np.ndarray,
    equipped_type: int = 0,
    context_frames: int = 1,
):
    """Test model predictions for a given inventory scenario."""
    print(f"\n{'='*60}")
    print(f"Scenario: {name}")
    print(f"Inventory: ", end="")
    nonzero = [(INVENTORY_ITEMS[i], inventory[i]) for i in range(len(inventory)) if inventory[i] > 0]
    print(nonzero if nonzero else "(empty)")
    print(f"{'='*60}")
    
    # Create dummy frames [1, T, H, W, C]
    frames = np.stack([make_dummy_frame() for _ in range(context_frames)], axis=0)
    frames = torch.from_numpy(frames).unsqueeze(0).to(device)  # [1, T, H, W, C]
    
    # Inventory [1, T, 18]
    inv_tensor = torch.from_numpy(inventory).unsqueeze(0).unsqueeze(0).expand(1, context_frames, -1).to(device)
    
    # Equipped type [1, T]
    equipped = torch.full((1, context_frames), equipped_type, dtype=torch.int64, device=device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(frames, inv_tensor, equipped)
    
    # Analyze categorical predictions
    print("\nCategorical action probabilities:")
    
    for action_name, vocab in [
        ("craft", CRAFT_VOCAB),
        ("nearby_craft", NEARBY_CRAFT_VOCAB),
        ("equip", EQUIP_VOCAB),
        ("place", PLACE_VOCAB),
    ]:
        probs = F.softmax(logits[action_name][:, -1], dim=-1).cpu().numpy()[0]
        print(f"\n  {action_name}:")
        # Show all classes with prob > 1%
        for i, (v, p) in enumerate(zip(vocab, probs)):
            if p > 0.01 or i == 0:
                marker = " <--" if p == probs.max() and i > 0 else ""
                print(f"    {i}: {v:20s} {p*100:5.1f}%{marker}")
    
    # Also show button predictions
    print("\n  Buttons (prob of pressing):")
    for btn_name in ["fwd", "jump", "attack"]:
        key = f"button_{btn_name}"
        if key in logits:
            probs = F.softmax(logits[key][:, -1], dim=-1).cpu().numpy()[0]
            print(f"    {btn_name}: {probs[1]*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Test inventory conditioning")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    cfg = checkpoint["config"]
    if isinstance(cfg, dict):
        cfg = MineRLPolicyConfig(**cfg)
    
    model = MineRLPolicy(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Model loaded (epoch {checkpoint.get('epoch', '?')})")
    print(f"Config: n_camera_bins={cfg.n_camera_bins}, minimal_actions={cfg.minimal_actions}")
    
    # Test scenarios
    
    # 1. Has logs -> should craft planks
    test_scenario(
        model, device,
        name="Has logs (should craft planks)",
        inventory=make_inventory(log=4),
    )
    
    # 2. Has planks -> should craft sticks or crafting_table
    test_scenario(
        model, device,
        name="Has planks (should craft sticks or crafting_table)",
        inventory=make_inventory(planks=8),
    )
    
    # 3. Has sticks + planks -> should craft wooden_pickaxe (nearby_craft)
    test_scenario(
        model, device,
        name="Has sticks + planks + crafting_table (should nearby_craft wooden_pickaxe)",
        inventory=make_inventory(stick=4, planks=6, crafting_table=1),
    )
    
    # 4. Has cobblestone + sticks -> should craft stone_pickaxe
    test_scenario(
        model, device,
        name="Has cobblestone + sticks + crafting_table (should nearby_craft stone_pickaxe)",
        inventory=make_inventory(cobblestone=10, stick=4, crafting_table=1),
    )
    
    # 5. Has iron_ingot + sticks -> should craft iron_pickaxe
    test_scenario(
        model, device,
        name="Has iron_ingot + sticks + crafting_table (should nearby_craft iron_pickaxe)",
        inventory=make_inventory(iron_ingot=3, stick=4, crafting_table=1),
    )
    
    # 6. Has wooden_pickaxe -> should equip it
    test_scenario(
        model, device,
        name="Has wooden_pickaxe (should equip it)",
        inventory=make_inventory(wooden_pickaxe=1),
    )
    
    # 7. Has stone_pickaxe -> should equip it
    test_scenario(
        model, device,
        name="Has stone_pickaxe (should equip it)",
        inventory=make_inventory(stone_pickaxe=1, wooden_pickaxe=1),
    )
    
    # 8. Has coal + iron_ore + furnace -> should nearby_smelt iron_ingot
    test_scenario(
        model, device,
        name="Has coal + iron_ore + furnace (should smelt iron_ingot)",
        inventory=make_inventory(coal=4, iron_ore=3, furnace=1),
    )
    
    # 9. Empty inventory -> should predict "none" for all
    test_scenario(
        model, device,
        name="Empty inventory (should predict 'none' for all)",
        inventory=make_inventory(),
    )
    
    # 10. Has coal + sticks -> should craft torch
    test_scenario(
        model, device,
        name="Has coal + sticks (should craft torch)",
        inventory=make_inventory(coal=4, stick=4),
    )


if __name__ == "__main__":
    main()

