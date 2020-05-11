import torch
import os

from torch.onnx import export

from config import cfg
from modeling import build_model

if __name__ == '__main__':
    checkpoint_path = './logs/resnet50_model_120.pth'
    cfg_filename = './configs/softmax_triplet_with_center.yml'
    num_classes = 2494

    cfg.merge_from_file(cfg_filename)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = torch.load(checkpoint_path)

    with torch.no_grad():
        export(model, torch.zeros((32, 3, 192, 192), dtype=torch.float32).cuda(),
               './resnest50_model.onnx',
               opset_version=9,
               do_constant_folding=True)
