import sys
import os

import torch

from transformer2 import resizeImage, SinPosEncoding, HWRTransformer

def main():
    # input height, input width, nr of chars, and longest label
    #   hardcoded here as it is dependent on original dataset
    transformer = HWRTransformer(128, 2260, 82, 56)

    transformer.load_state_dict(torch.load("model_test1.pth")).to("cpu")
    
if __name__ == '__main__':
    main()