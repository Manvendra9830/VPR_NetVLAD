import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# Define which layers to prune for each architecture
# We target convolutional and linear layers as they contain the most parameters.
LAYERS_TO_PRUNE = {
    'vgg16': [nn.Conv2d, nn.Linear],
    'alexnet': [nn.Conv2d, nn.Linear]
}

def apply_unstructured_pruning(model, arch, amount):
    """Applies unstructured L1 magnitude pruning to the specified model."""
    print(f'Applying {amount*100:.1f}% unstructured pruning to {arch}...')
    layers_to_prune = LAYERS_TO_PRUNE.get(arch, [])
    
    for module in model.modules():
        if isinstance(module, tuple(layers_to_prune)):
            prune.l1_unstructured(module, name='weight', amount=amount)
    print('Unstructured pruning applied.')
    return model


def apply_structured_pruning(model, arch, amount):
    """Applies structured L2 magnitude pruning to the specified model."""
    print(f'Applying {amount*100:.1f}% structured pruning to {arch}...')
    layers_to_prune = LAYERS_TO_PRUNE.get(arch, [])

    for module in model.modules():
        if isinstance(module, tuple(layers_to_prune)):
            # For Conv2d, prune entire filters (dim=0)
            # For Linear, prune entire output neurons (dim=0)
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
    print('Structured pruning applied.')
    return model

def make_pruning_permanent(model):
    """Removes the pruning re-parameterization from the model."""
    print('Making pruning permanent...')
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if prune.is_pruned(module):
                prune.remove(module, 'weight')
    print('Pruning made permanent.')
    return model


def calculate_sparsity(model):
    """Calculates the global sparsity of the model."""
    total_zeros = 0
    total_params = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total_zeros += torch.sum(module.weight == 0)
            total_params += module.weight.nelement()
    
    if total_params == 0:
        return 0
        
    sparsity = 100. * float(total_zeros) / float(total_params)
    print(f'Global Sparsity: {sparsity:.2f}%')
    return sparsity
