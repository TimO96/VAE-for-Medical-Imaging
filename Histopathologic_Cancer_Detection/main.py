from model import *
from distributions import *
from loader import *
import argparse


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        #data = Variable(data)
        optimizer.zero_grad()
        x_hat, loss, _ = model(data, epoch)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch, batch_size):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data
            x_hat, loss, _ = model(data, epoch)
            test_loss += loss.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      x_hat.view(batch_size, 1, 28, 28)[:n]])

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))



parser = argparse.ArgumentParser(description='Histopathologic VAE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--learning-rate', type=int, default=1e-3, metavar='L',
                                help='Learning rate of the optimizer')
parser.add_argument('--beta', type=int, default=[0, 1, 0], metavar='B',
                                help='set the beta parameter for b-VAE from 0 to 1 in the first 90% if [0, 1, .9]')
parser.add_argument('--final-beta', type=int, default=5, metavar='B',
                                help='set the beta parameter for the last epoch')
parser.add_argument('--base-distribution', type=int, default=normal_dist, metavar='T',
                                help='set the beta parameter for the last epoch')
parser.add_argument('--loss-distribution', type=int, default=bernoulli_loss, metavar='T',
                                help='set the beta parameter for the last epoch')
parser.add_argument('--target-distribution', type=int, default=normal_dist, metavar='T',
                                help='set the beta parameter for the last epoch')
args = parser.parse_args()


if __name__ == "__main__":
    labels = pd.read_csv('data/train_labels.csv')
    img_class_dict = {k:v for k, v in zip(labels.id, labels.label)}
    param = np.arange(0, 0.4, 0.05)

    data_transforms = transforms.Compose([
        #transforms.CenterCrop(300),
        #torchvision.transforms.ColorJitter(saturation=np.random.choice(param), brightness=np.random.choice(param)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
        ])
    data_transforms_test = transforms.Compose([
        #transforms.CenterCrop(300),
        transforms.ToTensor()
        ])


    dataset = CancerDataset(datafolder='data/train/', datatype='train', transform=data_transforms, labels_dict=img_class_dict)
    test_set = CancerDataset(datafolder='data/test/', datatype='test', transform=data_transforms_test)

    batch_size = 64
    num_workers = 0

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    for i, (data, label) in enumerate(train_loader):
        #plt.imshow(data[0].permute(1, 2, 0))
        #plt.show()
        break


    x_dim = 784
    z_dim = 20

    lr = 1e-3 # Learning rate
    batch_size = 64
    epochs = 10
    max_beta = 3
    one_third_epochs = round(epochs/3)


    # Prior distribution
    tensor = torch.ones(1)
    p_x_dist = Beta(tensor.new_full((1, 20), 0.5), tensor.new_full((1, 20), 0.5))
    p_x_dist = Independent(p_x_dist, 1)

    # Target distribution
    q_z_dist = beta_dist

    # Loss distribution
    loss_dist = bernoulli_loss

    conv_encoder_layers = [
        # nn.Conv2d(3, 16, 3, padding=1),
        # nn.ReLU(True),
        # nn.MaxPool2d(2, 2),
        # nn.Conv2d(16, 32, 3, padding=1),
        # nn.ReLU(True),
        # nn.MaxPool2d(2, 2),
        # nn.Conv2d(32, 64, 3, padding=1),
        # nn.ReLU(True),
        # nn.MaxPool2d(2,2)
        ]

    linear_encoder_layers = [
        nn.Linear(96*96*3, 392),
        nn.ReLU(True),
        nn.Linear(392, 196),
        nn.ReLU(True),
        nn.Linear(196, 49),
        nn.ReLU(True),
        nn.Linear(49, z_dim*2),
        nn.Softplus()
        ]

    linear_decoder_layers = [
        nn.Linear(z_dim, 49),
        nn.ReLU(True),
        nn.Linear(49, 196),
        nn.ReLU(True),
        nn.Linear(196, 392),
        nn.ReLU(True),
        nn.Linear(392, 96*96*3),
        nn.Sigmoid()
        ]

    conv_decoder_layers = [
        # nn.ConvTranspose2d(64, 32, 4, padding=1),
        # nn.ReLU(True),
        # nn.Upsample(scale_factor=3),
        # nn.ConvTranspose2d(32, 16, 4, padding=1),
        # nn.ReLU(True),
        # nn.Upsample(scale_factor=3),
        # nn.ConvTranspose2d(16, 3, 5, padding=1),
        # nn.ReLU(True),
        ]

    # Load the MNIST data
    #train_data, train_loader, test_data, test_loader = load_data(batch_size)

    # Create the model
    model = VAE(conv_encoder=conv_encoder_layers, linear_encoder=linear_encoder_layers,
                linear_decoder=linear_decoder_layers, conv_decoder=conv_decoder_layers,
                p_x_dist=p_x_dist, q_z_dist=q_z_dist, loss_dist=loss_dist, max_beta=max_beta, one_third_epochs=one_third_epochs)

    optimizer = optim.Adam(model.parameters(), lr=lr)


    for epoch in range(1, epochs + 1):
        train(epoch)
        #images = model.decode(torch.tensor(sample, dtype=torch.float)).detach().numpy()
