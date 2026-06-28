import sys
import torch

# Patch PyTorch's load function to default to weights_only=False.
# This ensures compatibility with older pre-trained weights files in newer PyTorch versions.
_original_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

import torch.nn as nn

# Register 'ssn_model' as 'model' in sys.modules to handle serialization and
# import path compatibility during PyTorch model unpickling.
import ssn_model
sys.modules['model'] = ssn_model

from ultralytics import RTDETR
from ssn_handler import FrozenSSN
from modules import SuperpixelGAT

# Global buffer to temporarily store superpixel features computed during the backbone pre-hook
ssn_feature_buffer = {} 

def get_ssn_hook(ssn_model):
    """
    Creates a pre-forward hook to compute superpixel features before the backbone execution.
    """
    def pre_hook(module, args):
        imgs = args[0]  # Extract the input image batch
        with torch.no_grad():
            # Forward pass through the frozen SSN model to get superpixel features
            ssn_out = ssn_model(imgs)

        # Store the computed superpixel features in the global buffer
        ssn_feature_buffer['feat'] = ssn_out

        return args
    return pre_hook

def get_gat_hook(gat_model):
    """
    Creates a forward hook to enhance the deepest backbone features (C5) 
    using the buffered superpixel features via Graph Attention.
    """
    def forward_hook(module, inputs, output):
        ssn_feat = ssn_feature_buffer.get('feat')
        
        if ssn_feat is not None:
            try:
                # Enhance the C5 feature maps (typically the last element of the backbone outputs)
                enhanced_c5 = gat_model(output[-1], ssn_feat)
                output[-1] = enhanced_c5 
            except Exception as e:
                print(f"GAT Hook Error: {e}")
                pass
            
        return output
    return forward_hook

if __name__ == '__main__':
    # Initialize the base RT-DETR model using pre-trained weights
    model = RTDETR('rtdetr-l.pt')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Instantiate the Simple Superpixel Network (SSN) and load its pre-trained weights
    ssn = FrozenSSN('best_model.pth', nspix=100).to(device)
    
    # Instantiate the Superpixel-Guided Graph Attention Network (GAT)
    gat = SuperpixelGAT(in_channels=2048, n_spix=100).to(device)
    
    # Extract the backbone module (the first component of the RT-DETR architecture)
    backbone = model.model.model[0]
    
    # Register the pre-forward hook to extract superpixel features on the fly
    print("Registering SSN Pre-forward Hook...")
    backbone.register_forward_pre_hook(get_ssn_hook(ssn))
    
    # Register the forward hook to inject the GAT enhancement to C5 features
    print("Registering GAT Forward Hook...")
    backbone.register_forward_hook(get_gat_hook(gat))
    
    # Add the custom GAT module to the model hierarchy so PyTorch 
    # tracks and updates its parameters during backward optimization.
    model.model.add_module('custom_gat', gat)
    
    # Begin training the customized model on the specified dataset with custom hyperparameters
    model.train(
        data=r'D:\Lab\UAVVasteDataset\FloW_IMG\yolo_dataset\dataset.yaml', 
        epochs=100, 
        imgsz=640,
        batch=4, 
        lr0=0.001,
        device=0,
        workers=0
    )
