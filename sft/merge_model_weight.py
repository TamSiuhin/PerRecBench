from safetensors.torch import load_file, save_file
import torch
from typing import List, Dict
from collections import defaultdict

def average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Average multiple state dictionaries element-wise.
    
    Args:
        state_dicts (List[Dict[str, torch.Tensor]]): List of state dictionaries to average
        
    Returns:
        Dict[str, torch.Tensor]: Averaged state dictionary
        
    Raises:
        ValueError: If state_dicts is empty or if dictionaries have different keys
    """
    if not state_dicts:
        raise ValueError("List of state dictionaries cannot be empty")
    
    # Verify all state dicts have the same keys
    reference_keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        if set(sd.keys()) != reference_keys:
            raise ValueError("All state dictionaries must have the same keys")
    
    # Initialize the averaged state dict
    averaged_state = {}
    
    # Iterate through all keys
    for key in reference_keys:
        # Stack all tensors for the current key
        stacked_tensors = torch.stack([sd[key] for sd in state_dicts])
        # Calculate mean along the first dimension (across models)
        averaged_state[key] = torch.mean(stacked_tensors, dim=0)
    
    return averaged_state

def average_state_dicts_weighted(state_dicts: List[Dict[str, torch.Tensor]], 
                               weights: List[float]) -> Dict[str, torch.Tensor]:
    """
    Average multiple state dictionaries with custom weights.
    
    Args:
        state_dicts (List[Dict[str, torch.Tensor]]): List of state dictionaries to average
        weights (List[float]): List of weights for each state dict (must sum to 1)
        
    Returns:
        Dict[str, torch.Tensor]: Weighted averaged state dictionary
    """
    if not state_dicts or not weights:
        raise ValueError("Lists cannot be empty")
    
    if len(state_dicts) != len(weights):
        raise ValueError("Number of weights must match number of state dicts")
    
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1")
    
    # Convert weights to tensor for broadcasting
    weights = torch.tensor(weights, dtype=torch.float32)
    
    # Initialize the averaged state dict
    averaged_state = {}
    
    # Get reference keys from first state dict
    reference_keys = set(state_dicts[0].keys())
    
    # Iterate through all keys
    for key in reference_keys:
        # Stack tensors and apply weights
        stacked_tensors = torch.stack([sd[key] for sd in state_dicts])
        # Apply weights along first dimension and sum
        # Reshape weights for proper broadcasting
        weights_shaped = weights.view(-1, *([1] * len(stacked_tensors.shape[1:])))
        averaged_state[key] = torch.sum(stacked_tensors * weights_shaped, dim=0)
    
    return averaged_state

# Example usage
if __name__ == "__main__":
    # Create some dummy state dicts for testing
    state_dict_point = load_file("/afs/crc.nd.edu/user/z/ztan3/Private/PerRecLLM/sft/model/PerRecBench-Mistral-Nemo-sft-lora-point/adapter_model.safetensors")
    state_dict_pair = load_file("/afs/crc.nd.edu/user/z/ztan3/Private/PerRecLLM/sft/model/PerRecBench-Mistral-Nemo-sft-lora-pair/adapter_model.safetensors")
    state_dict_group = load_file("/afs/crc.nd.edu/user/z/ztan3/Private/PerRecLLM/sft/model/PerRecBench-Mistral-Nemo-sft-lora-group/adapter_model.safetensors")

    
    # Test equal weights averaging
    state_dicts = [state_dict_point, state_dict_pair, state_dict_group]
    averaged = average_state_dicts(state_dicts)
    print(averaged)
    save_file(averaged, "/afs/crc.nd.edu/user/z/ztan3/Private/PerRecLLM/sft/model/PerRecBench-Mistral-Nemo-sft-lora-avg/adapter_model.safetensors")

    # # Test weighted averaging
    # weights = [0.5, 0.3, 0.2]
    # weighted_averaged = average_state_dicts_weighted(state_dicts, weights)