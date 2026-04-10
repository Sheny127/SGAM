import sys
import torch

_original_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

import torch.nn as nn

import ssn_model
sys.modules['model'] = ssn_model

from ultralytics import RTDETR
from ssn_handler import FrozenSSN
from modules import SuperpixelGAT

ssn_feature_buffer = {} 

def get_ssn_hook(ssn_model):

    def pre_hook(module, args):

        imgs = args[0]
        with torch.no_grad():

            ssn_out = ssn_model(imgs)

        ssn_feature_buffer['feat'] = ssn_out

        return args
    return pre_hook

def get_gat_hook(gat_model):

    def forward_hook(module, inputs, output):

        ssn_feat = ssn_feature_buffer.get('feat')
        
        if ssn_feat is not None:

            try:
                enhanced_c5 = gat_model(output[-1], ssn_feat)
                output[-1] = enhanced_c5 
            except Exception as e:
                print(f"GAT Hook Error: {e}")

                pass
            
        return output
    return forward_hook

if __name__ == '__main__':

    model = RTDETR('rtdetr-l.pt')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    ssn = FrozenSSN('best_model.pth', nspix=100).to(device)
    
    gat = SuperpixelGAT(in_channels=2048, n_spix=100).to(device)
    
    backbone = model.model.model[0]
    
    print("SSN Hook...")
    backbone.register_forward_pre_hook(get_ssn_hook(ssn))
    
    print("GAT Hook...")
    backbone.register_forward_hook(get_gat_hook(gat))
    
    model.model.add_module('custom_gat', gat)
    
    model.train(
        data=r'D:\Lab\UAVVasteDataset\FloW_IMG\yolo_dataset\dataset.yaml', 
        epochs=100, 
        imgsz=640,
        batch=4, 
        lr0=0.001,
        device=0,
        workers=0
    )