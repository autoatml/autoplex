import pytest
import yaml
from pathlib import Path
from autoplex import PacemakerSettings

def test_generate_input_yaml_structure(test_dir):
    """
    Task 1: Verify that we can generate a Pydantic model that matches 
    the structure of the provided Si_input.yaml.
    """
    # 1. Load the reference YAML provided by user
    ref_yaml_path = test_dir / "fitting/PACE/Si_input.yaml"
    if not ref_yaml_path.exists():
        pytest.skip("Si_input.yaml not found in test_data")
        
    with open(ref_yaml_path, 'r') as f:
        ref_dict = yaml.safe_load(f)

    # 2. Extract sections from reference to simulate passing them as kwargs
    #    (mimicking how MLIPFitMaker passes arguments)
    fit_config = ref_dict.get("fit", {})
    potential_config = ref_dict.get("potential", {})
    backend_config = ref_dict.get("backend", {})
    cutoff = ref_dict.get("cutoff", 8.0)
    
    # 3. Initialize the Settings Object (Autoplex Logic)
    settings = PacemakerSettings()
    
    # Simulate the update logic used in utils.py/flows.py
    settings.cutoff = cutoff
    settings.fit = fit_config
    settings.potential = potential_config
    settings.backend = backend_config
    
    # Set dummy data file just to make validation pass
    settings.data.filename = "dummy.extxyz" 
    
    # 4. Serialize to dict
    generated_dict = settings.model_dump(by_alias=True, exclude_none=True)

    # 5. Assertions - Compare Critical Sections
    assert generated_dict['potential']['elements'] == ref_dict['potential']['elements']
    assert 'embeddings' in generated_dict['potential']
    assert 'ALL' in generated_dict['potential']['embeddings']
    assert generated_dict['potential']['embeddings']['ALL']['npot'] == "FinnisSinclairShiftedScaled"
    assert generated_dict['fit']['optimizer'] == ref_dict['fit']['optimizer']
    assert generated_dict['fit']['loss']['kappa'] == ref_dict['fit']['loss']['kappa']
    assert generated_dict['backend']['evaluator'] == ref_dict['backend']['evaluator']
    
    print("\n[SUCCESS] Autoplex PacemakerSettings model successfully replicated the structure of Si_input.yaml")

def assert_subset_recursive(reference, generated, path="root"):
    """
    Recursively check that all keys and values in 'reference' exist in 'generated'.
    This ensures that Autoplex does not drop any configuration parameters provided by the user.
    """
    for key, val in reference.items():
        current_path = f"{path}.{key}"
        
        # 1. Skip validation if input value is explicitly None (handled by schema defaults)
        if val is None:
            continue 

        # 2. Check Key Existence
        assert key in generated, f"Missing Parameter: Key '{current_path}' found in input but lost in Autoplex output."
        
        gen_val = generated[key]

        # 3. Check Value Matching
        if isinstance(val, dict):
            assert isinstance(gen_val, dict), f"Type Mismatch: '{current_path}' should be dict, got {type(gen_val)}"
            assert_subset_recursive(val, gen_val, current_path)
        
        elif isinstance(val, list):
            if val and isinstance(val[0], float):
                assert gen_val == pytest.approx(val), f"List Mismatch at '{current_path}'"
            else:
                assert gen_val == val, f"List Mismatch at '{current_path}'"
        
        elif isinstance(val, (int, float)) and isinstance(gen_val, (int, float)):
             assert gen_val == pytest.approx(val), f"Numeric Mismatch at '{current_path}': {val} vs {gen_val}"
            
        else:
            assert str(gen_val) == str(val), f"Value Mismatch at '{current_path}': {val} vs {gen_val}"


@pytest.mark.parametrize("yaml_filename", [
    "Si_input.yaml", 
    "GST_input.yaml", 
    "C_input.yaml"
])
def test_generate_complex_inputs(test_dir, yaml_filename):
    """
    Verify that Autoplex's PacemakerSettings class can NATIVELY parse 
    complex input dictionaries provided by the user.
    """
    # 1. Load Reference
    ref_yaml_path = test_dir / "fitting/PACE" / yaml_filename
    if not ref_yaml_path.exists():
        pytest.skip(f"{yaml_filename} not found in test data")
        
    print(f"\n--- Testing generation for: {yaml_filename} ---")
        
    with open(ref_yaml_path, 'r') as f:
        ref_dict = yaml.safe_load(f)

    # 2. Initialize Settings - THE "REAL" API TEST
    # Instead of manually looping and setting attributes (simulation), 
    # we pass the whole dictionary directly.
    # If Autoplex's internal schema (settings.py) is correct, Pydantic 
    # will automatically parse, validate, and convert nested types here.
    try:
        settings = PacemakerSettings(**ref_dict)
    except Exception as e:
        pytest.fail(f"Autoplex failed to parse {yaml_filename} natively: {e}")
    
    # 3. Generate Output (Serialize back to dict)
    # This tests the 'dump' logic of Autoplex
    generated_dict = settings.model_dump(by_alias=True, exclude_none=True)

    # 4. Verification
    # Ensure the output matches the input structure
    assert_subset_recursive(ref_dict, generated_dict)
    
    # Print for inspection
    print(f"\n[OUTPUT] Autoplex Generated YAML for {yaml_filename}:\n{'-'*50}")
    print(yaml.dump(generated_dict, sort_keys=False))
    print(f"{'-'*50}")
    
    print(f"[SUCCESS] {yaml_filename} reproduced correctly via native API injection.")

# Keep the simple structure test for sanity checking specifics
def test_generate_input_yaml_structure(test_dir):
    """
    Task 1: Basic structural check using direct instantiation.
    """
    ref_yaml_path = test_dir / "fitting/PACE/Si_input.yaml"
    if not ref_yaml_path.exists():
        pytest.skip("Si_input.yaml not found")
        
    with open(ref_yaml_path, 'r') as f:
        ref_dict = yaml.safe_load(f)

    # Test Native Instantiation
    settings = PacemakerSettings(**ref_dict)
    
    generated_dict = settings.model_dump(by_alias=True, exclude_none=True)

    # Assertions
    assert generated_dict['potential']['embeddings']['ALL']['npot'] == "FinnisSinclairShiftedScaled"
    assert generated_dict['backend']['evaluator'] == ref_dict['backend'].get('evaluator', 'tensorpot')