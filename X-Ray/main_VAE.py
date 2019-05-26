from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import Normal, Laplace, Independent, Bernoulli, Gamma, Uniform, Beta
from torch.distributions.kl import kl_divergence
import pickle

import model_VAE
import evaluation_VAE
import data_loader


parser = argparse.ArgumentParser(description='VAE XRAY')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--tsne', action='store_true',
                    help='t-sne plot added')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

path = 'chest-xray-pneumonia/chest_xray/'

'''
train_loader = torch.utils.data.DataLoader(
    data_loader.Xray_Dataset(path, train=True),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    data_loader.Xray_Dataset(path, train=False),
    batch_size=args.batch_size, shuffle=True, **kwargs)
'''

train_data = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

test_data = datasets.MNIST('../data', train=False, transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.batch_size, shuffle=True, **kwargs)

x_dim = 28
z_dim = 2

model = model_VAE.VAE(x_dim**2, z_dim).to(device)
#filename = 'finalized_model_laplace.sav'
#model = pickle.load(open(filename, 'rb'))

optimizer = optim.Adam(model.parameters(), lr=1e-3)

reconstruction_function = nn.BCELoss()

def loss_function(x_hat, x, q_z, p_x, z):
    #x = x.view(-1, x_hat.size(1))
    #tensor = torch.ones(1)
    #p_x_dist = Beta(tensor.new_full((z.size(0), z_dim), 0.5), tensor.new_full((z.size(0), z_dim), 0.5))
    p_x_dist = Normal(torch.zeros(z.size(0), z_dim), torch.ones(z.size(0), z_dim))
    BCE = torch.sum(-p_x.log_prob(x.view(x.size(0), x_dim**2)))
    KLD = torch.sum(q_z.log_prob(z) - p_x_dist.log_prob(z))

    print(BCE, KLD)

    return BCE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, q_z, p_x, z = model(data)
        loss = loss_function(recon_batch, data, q_z, p_x, z)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, q_z, p_x, z = model(data)
            test_loss += loss_function(recon_batch, data, q_z, p_x, z).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, x_dim, x_dim)[:n]])
                print(label[:n])
                save_image(comparison.cpu(),
                         'results/reconstruction_nnlaplace_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, z_dim).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, x_dim, x_dim),
'results/sample_betabetalaplace_' + str(epoch) + '.png')
        filename = 'finalized_model_laplace25.sav'
        pickle.dump(model, open(filename, 'wb'))
    #if args.tsne:
        #evaluation_VAE.tsne_plot(model, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
