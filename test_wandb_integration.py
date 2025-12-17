#!/usr/bin/env python
"""
Test script to verify wandb integration works correctly.
This script tests the logger without actually running training.
"""

import sys
import torch
from unittest.mock import Mock

# Test imports
print("=" * 60)
print("Testing wandb integration for WanVideo pipeline")
print("=" * 60)

# Test 1: Import ModelLogger
print("\n[Test 1] Importing ModelLogger...")
try:
    from diffsynth.diffusion import ModelLogger
    print("✓ ModelLogger imported successfully")
except Exception as e:
    print(f"✗ Failed to import ModelLogger: {e}")
    sys.exit(1)

# Test 2: Initialize logger without wandb
print("\n[Test 2] Initializing ModelLogger without wandb...")
try:
    logger = ModelLogger(
        output_path="./test_output",
        remove_prefix_in_ckpt="pipe.dit.",
        use_wandb=False
    )
    assert logger.use_wandb == False
    assert logger.wandb_initialized == False
    print("✓ ModelLogger initialized without wandb")
except Exception as e:
    print(f"✗ Failed to initialize ModelLogger: {e}")
    sys.exit(1)

# Test 3: Initialize logger with wandb (may not have wandb installed)
print("\n[Test 3] Initializing ModelLogger with wandb flag...")
try:
    logger_wandb = ModelLogger(
        output_path="./test_output",
        use_wandb=True,
        wandb_project="test-project",
        wandb_run_name="test-run",
        wandb_config={"learning_rate": 1e-4, "epochs": 5}
    )
    print(f"✓ ModelLogger initialized with wandb (enabled: {logger_wandb.use_wandb})")
    if not logger_wandb.use_wandb:
        print("  Note: wandb is not installed, which is expected")
except Exception as e:
    print(f"✗ Failed to initialize ModelLogger with wandb: {e}")
    sys.exit(1)

# Test 4: Test init_wandb method
print("\n[Test 4] Testing init_wandb method...")
try:
    mock_accelerator = Mock()
    mock_accelerator.is_main_process = True
    
    # This should not crash even if wandb is not installed
    logger_wandb.init_wandb(mock_accelerator)
    print("✓ init_wandb method executed without errors")
except Exception as e:
    print(f"✗ init_wandb method failed: {e}")
    sys.exit(1)

# Test 5: Test log_metrics method
print("\n[Test 5] Testing log_metrics method...")
try:
    metrics = {"train/loss": 0.5, "train/accuracy": 0.95}
    logger_wandb.log_metrics(metrics)
    print("✓ log_metrics method executed without errors")
except Exception as e:
    print(f"✗ log_metrics method failed: {e}")
    sys.exit(1)

# Test 6: Test on_step_end with loss
print("\n[Test 6] Testing on_step_end with loss parameter...")
try:
    mock_model = Mock()
    loss_tensor = torch.tensor(0.5)
    logger.on_step_end(mock_accelerator, mock_model, save_steps=None, loss=loss_tensor)
    assert logger.num_steps == 1
    print("✓ on_step_end with loss parameter works correctly")
except Exception as e:
    print(f"✗ on_step_end failed: {e}")
    sys.exit(1)

# Test 7: Check parsers
print("\n[Test 7] Checking parser functions...")
try:
    from diffsynth.diffusion import add_wandb_config
    import argparse
    parser = argparse.ArgumentParser()
    parser = add_wandb_config(parser)
    
    # Check if arguments are added
    args = parser.parse_args(['--use_wandb', '--wandb_project', 'test', '--wandb_run_name', 'run1'])
    assert args.use_wandb == True
    assert args.wandb_project == 'test'
    assert args.wandb_run_name == 'run1'
    print("✓ Parser functions work correctly")
except Exception as e:
    print(f"✗ Parser test failed: {e}")
    sys.exit(1)

# Test 8: Verify runner imports
print("\n[Test 8] Verifying runner imports...")
try:
    from diffsynth.diffusion import launch_training_task
    print("✓ launch_training_task imported successfully")
except Exception as e:
    print(f"✗ Failed to import launch_training_task: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nThe wandb integration is ready to use.")
print("\nTo use wandb logging in training:")
print("  1. Install wandb: pip install wandb")
print("  2. Login: wandb login")
print("  3. Add --use_wandb to your training command")
print("\nExample:")
print("  accelerate launch train.py [args...] --use_wandb \\")
print("    --wandb_project 'my-project' --wandb_run_name 'exp001'")
print("=" * 60)
