import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import argparse
from model import VAEForFaces
from pathlib import Path
from datetime import datetime
from data import ImageList, make_rgb, to_byte_tensor, to_float_tensor, ResizeFixed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    return parser.parse_args()

args = parse_args()

output_path = Path(args.output)/datetime.now().strftime('%Y%m%d')
output_path.mkdir(parents=True, exist_ok=True)

img_celebs = Path.home()/'GDL_code/data/celeb/img_align_celeba'
tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor]
il = ImageList.from_files(img_celebs, tfms=tfms)

dl = torch.utils.data.DataLoader(il, batch_size=args.batch_size, shuffle=True)

from torch.autograd import Variable

model = VAEForFaces()
opt = Adam(model.parameters(), lr=args.learning_rate)

r_loss_func = nn.MSELoss(reduction='sum')
def loss_func(pred, x, mu, logvar):
    r_loss = r_loss_func(pred, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return r_loss + kl_divergence

for epoch in range(10):
    running_loss = 0.0
    for x in dl:
        pred, mu, logvar = model(x)
        loss = loss_func(pred, x, mu, logvar)
        opt.zero_grad()
        loss.backward()
        opt.step()
        running_loss += loss.item()
    print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(dl.dataset)))
    with torch.no_grad():
        sample = torch.randn(64, 200)
        sample = model.decode(sample)
        save_image(sample.view(args.batch_size, 3, 128, 128), output_path/'image_{}.png'.format(epoch+1))
