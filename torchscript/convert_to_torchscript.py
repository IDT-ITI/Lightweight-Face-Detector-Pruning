import torch
from data.config import cfg
from models.eresfd import build_model_for_torchscript
from torch.utils.mobile_optimizer import optimize_for_mobile

if __name__ == '__main__':
    """ Convert model to torchscript and save to disk """

    # Load model
    model_path = './weights/ERES.pth'

    net = build_model_for_torchscript('test', cfg.NUM_CLASSES, width_mult=0.0625)
    net.load_state_dict(torch.load(model_path), strict=True)
    net.eval()
    print('Model loaded')

    # Script and save model
    print('Scripting...')
    net = net.to('cpu')
    scripted_model = torch.jit.script(net)
    print('Scripting ended. Saving...')
    optimized_scripted_module = optimize_for_mobile(scripted_model)
    optimized_scripted_module._save_for_lite_interpreter("torchscript/lite_scripted_model.ptl")
    print('Model saved!')
