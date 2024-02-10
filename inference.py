import argparse

import cv2
import numpy as np
import torch

from backbones import get_model


@torch.no_grad()
def inference(net, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        feat = net(img)
        return feat[0].view(-1).numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--img', type=str, default=None)
    args = parser.parse_args()
    args.network = 'vit_s'
    args.weight = './checkpoints/glint360k_model_TransFace_S.pt'

    net = get_model(args.network, fp16=False)
    net.load_state_dict(torch.load(args.weight))
    net.eval()

    args.img = './imgs/dsw1.jpg'
    imgs = ['./imgs/dsw0.jpg', './imgs/dsw1.jpg', './imgs/dsw2.jpg','./imgs/dsw3.jpg']
    mj = ['./imgs/MJ0.jpg', './imgs/MJ1.jpg']
    f1 = inference(net, mj[0])
    # f2 = inference(net, imgs[3])
    f2 = inference(net, mj[1])
    # f2 = inference(net, mj[0])
    sim = np.dot(f1, f2)
    print(sim)