import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
# import intel_extension_for_pytorch as ipex

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
# device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = 'LR/*'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to("xpu")
# model = ipex.optimize(model)

print('Model path {:s}. \nTesting...'.format(model_path))

# idx = 0
# for path in glob.glob(test_img_folder):
#     idx += 1
#     base = osp.splitext(osp.basename(path))[0]
#     print(idx, base)
# read images
path = "LR/99257.jpg"
img = cv2.imread(path, cv2.IMREAD_COLOR)
img = img * 1.0 / 255
img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
img_LR = img.unsqueeze(0)
img_LR = img_LR.to("xpu")

with torch.no_grad():
    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
output = (output * 255.0).round()
cv2.imwrite('results/{:s}_rlt2.jpg'.format("output"), output)
