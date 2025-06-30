#!/usr/bin/env python3
"""
Test script to verify backward compatibility of multi-task DP3 implementation.
This script tests that existing single-task configurations still work correctly.
"""

import torch
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import_compatibility():
    """Test that imports work correctly."""
    print("Testing import compatibility...")

    try:
        from diffusion_policy_3d.policy.dp3 import DP3
        print("✓ DP3 import successful")
    except Exception as e:
        print(f"✗ DP3 import failed: {e}")
        return False

    try:
        # Test that DP3 class has the expected signature
        import inspect
        sig = inspect.signature(DP3.__init__)
        params = list(sig.parameters.keys())

        # Check that multi_task_config is optional
        assert 'multi_task_config' in params, "multi_task_config parameter missing"
        assert sig.parameters['multi_task_config'].default is None, "multi_task_config should default to None"
        print("✓ DP3 signature compatibility verified")

    except Exception as e:
        print(f"✗ DP3 signature test failed: {e}")
        return False

    return True

def test_configuration_compatibility():
    """Test that configuration files are compatible."""
    print("Testing configuration compatibility...")

    # Test that existing config files don't have multi_task_config
    import yaml

    try:
        with open('diffusion_policy_3d/config/dp3.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Check that policy section exists but doesn't have multi_task_config
        assert 'policy' in config, "Policy section missing from config"
        policy_config = config['policy']

        # multi_task_config should not be present in existing configs
        assert 'multi_task_config' not in policy_config, "Existing config should not have multi_task_config"
        print("✓ Existing configuration compatibility verified")

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

    return True

def test_multi_task_config_structure():
    """Test that multi-task configurations have the correct structure."""
    print("Testing multi-task configuration structure...")

    import yaml

    try:
        # Test multi-task config file
        with open('diffusion_policy_3d/config/dp3_multi_task.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Check that multi_task_config is present and correctly structured
        assert 'policy' in config, "Policy section missing from multi-task config"
        policy_config = config['policy']

        assert 'multi_task_config' in policy_config, "multi_task_config missing from multi-task policy"
        mt_config = policy_config['multi_task_config']

        # Check required fields
        assert 'enabled' in mt_config, "enabled field missing from multi_task_config"
        assert mt_config['enabled'] is True, "multi_task should be enabled in multi-task config"
        assert 'language_encoder' in mt_config, "language_encoder missing from multi_task_config"
        assert 'language_mlp_dim' in mt_config, "language_mlp_dim missing from multi_task_config"

        print("✓ Multi-task configuration structure verified")

    except Exception as e:
        print(f"✗ Multi-task configuration test failed: {e}")
        return False

    return True

def test_dataset_compatibility():
    """Test that dataset configurations are compatible."""
    print("Testing dataset compatibility...")

    try:
        # Test that MultiTaskRobotwinDataset can be imported
        from diffusion_policy_3d.dataset.multi_task_robotwin_dataset import MultiTaskRobotwinDataset
        print("✓ MultiTaskRobotwinDataset import successful")

        # Test that the class has the expected interface
        import inspect
        sig = inspect.signature(MultiTaskRobotwinDataset.__init__)
        params = list(sig.parameters.keys())

        # Check backward compatibility parameters
        assert 'zarr_path' in params, "zarr_path parameter missing (needed for backward compatibility)"
        assert 'task_name' in params, "task_name parameter missing (needed for backward compatibility)"
        assert 'multi_task_config' in params, "multi_task_config parameter missing"

        # Check that backward compatibility parameters are optional
        assert sig.parameters['zarr_path'].default is None, "zarr_path should be optional"
        assert sig.parameters['task_name'].default is None, "task_name should be optional"
        assert sig.parameters['multi_task_config'].default is None, "multi_task_config should be optional"

        print("✓ Dataset backward compatibility verified")

    except Exception as e:
        print(f"✗ Dataset compatibility test failed: {e}")
        return False

    return True

def main():
    """Run all backward compatibility tests."""
    print("Running backward compatibility tests for DP3 multi-task implementation...\n")

    all_passed = True

    # Run all tests
    tests = [
        test_import_compatibility,
        test_configuration_compatibility,
        test_multi_task_config_structure,
        test_dataset_compatibility
    ]

    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            all_passed = False

    if all_passed:
        print("\n✅ All backward compatibility tests passed!")
        print("Existing single-task configurations should work without modification.")
        print("Multi-task configurations are properly structured.")
    else:
        print("\n❌ Some tests failed!")
        return False

    return True

if __name__ == "__main__":
    main()
