from utils import BiSeNet, SpecificNorm, cosin_metric
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
import numpy as np

class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()
        self.device="cuda"
        self.spNorm = SpecificNorm()
        self.netSeg = BiSeNet(n_classes=19).to(self.device)
        self.netSeg.load_state_dict(torch.load('./checkpoints/FaceParser.pth'))
        self.netSeg.eval()



    def makeMask(self, origin_mask):
        numpy = origin_mask.squeeze(0).detach().cpu().numpy().argmax(0)
        numpy = numpy.copy().astype(np.uint8)

        # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        ids = [1, 2, 3, 4, 5, 10, 11, 12, 13]

        mask     = np.zeros([512, 512])
        for id in ids:
            index = np.where(numpy == id)
            mask[index] = 1
        mask = (mask * 255).astype(np.uint8)
        return mask



    def forward(self, targ_image):

        targ_mask = targ_image.detach().clone()
        targ_mask = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(targ_mask)
        targ_mask = self.netSeg(self.spNorm(targ_mask))[0]
        targ_mask = transforms.Resize((512,512))(targ_mask)

        mask  = self.makeMask(targ_mask)
        mask = Image.fromarray(mask)
        return mask