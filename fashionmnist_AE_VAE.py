import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Fashion_MNIST Dataset
# Load the Fashion MNIST dataset
transform = transforms.ToTensor()
Fashion_mnist_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(Fashion_mnist_dataset, batch_size=100, shuffle=False)

dataiter = iter(data_loader)
images, labels = next(dataiter)
print(torch.min(images),torch.max(images))

#  AUTOENCODER  :

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, x_dim, h_dim1,h_dim2):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, h_dim2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(h_dim2, h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, x_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Build model
ae = Autoencoder(x_dim=784, h_dim1=512, h_dim2=256)
optimizer = optim.Adam(ae.parameters())

# Training Autoencoder :
num_epoches = 50
outputs_AE = []
loss_auto=[]
print('AUTOENCODER : ')
for epoch in range(num_epoches):
    total_loss = 0
    for img, _ in data_loader:
        img = img.view(-1, 784)
        recon = ae(img)
        loss = F.mse_loss(recon, img, reduction='sum')
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    print(f'Epoch: {epoch+1}, Average Loss: {total_loss / len(data_loader.dataset):.4f}')
    outputs_AE.append((img, recon))
    loss_auto.append(float(f'{total_loss / len(data_loader.dataset):.4f}'))

# Visualization of results (Autoencoder)
# Create a grid of subplots
fig, axs = plt.subplots(2 * 4, 9, figsize=(9, 2 * 4))
plt.suptitle('Autoencoder Result', fontsize=16)  # Add title at the top

indices = [0,12,25,49]  # Specify the indices of the epochs to display

for i, k in enumerate(indices):
    # Get the images and reconstructions for the current epoch
    imgs = outputs_AE[k][0].detach().numpy().reshape(-1, 28, 28)
    recon = outputs_AE[k][1].detach().numpy().reshape(-1, 28, 28)

    # Plot the images from imgs
    for j, item in enumerate(imgs):
        if j >= 9:
            break
        axs[2 * i, j].imshow(item, cmap='gray')
        axs[2 * i, j].axis('off')

    # Plot the images from recon
    for j, item in enumerate(recon):
        if j >= 9:
            break
        axs[2 * i + 1, j].imshow(item, cmap='gray')
        axs[2 * i + 1, j].axis('off')

    # Set the title of each row as the epoch number
    axs[2 * i, 0].set_title(f'(AE) Epoch {k + 1}')

# Remove empty subplots
for k in range(2 * len(indices), axs.shape[0]):
    for j in range(axs.shape[1]):
        axs[k, j].axis('off')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# Autoencoder Loss curve :
x1 = [i for i in range(1, num_epoches+1)]
plt.plot(x1, loss_auto, color='r', label='Autoencoder loss curve')
plt.suptitle('Autoencoder loss curve', fontsize=16)  # Add title at the top
plt.grid()
plt.legend()
plt.show()


#  VARIATIONAL AUTOENCODER :

# Define the Variational Autoencoder (VAE) architecture :
class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
    
# build model
vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=2)
optimizer_VAE = optim.Adam(vae.parameters())

# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    MSE = F.mse_loss(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE + KLD

# Training Variational Autoencoder :
num_epochs_VAE = 50
outputs_VAE = []
loss_VAE=[]
print('VARIATIONAL AUTOENCODER : ')
for epoch in range(num_epochs_VAE):
    total_loss = 0
    for img, _ in data_loader:
        recon, mu, logvar = vae(img)
        loss = loss_function(recon, img, mu, logvar)
        optimizer_VAE.zero_grad()
        loss.backward()
        total_loss += loss.item()
        optimizer_VAE.step()

    print(f'Epoch: {epoch + 1}, Average Loss: {total_loss / len(data_loader.dataset):.4f}')
    outputs_VAE.append((img, recon))
    loss_VAE.append(float(f'{total_loss / len(data_loader.dataset):.4f}'))

# Visualization of results (Variational Autoencoder)
# Create a grid of subplots
fig, axs = plt.subplots(2 * 4, 9, figsize=(9, 2 * 4))
plt.suptitle('Variational Autoencoder output', fontsize=16)  # Add title at the top

indices = [0,12,25,49]   # Specify the indices of the epochs to display

for i, k in enumerate(indices):
    # Get the images and reconstructions for the current epoch
    imgs = outputs_VAE[k][0].detach().numpy().reshape(-1, 28, 28)
    recon = outputs_VAE[k][1].detach().numpy().reshape(-1, 28, 28)

    # Plot the images from imgs
    for j, item in enumerate(imgs):
        if j >= 9:
            break
        axs[2 * i, j].imshow(item, cmap='gray')
        axs[2 * i, j].axis('off')

    # Plot the images from recon
    for j, item in enumerate(recon):
        if j >= 9:
            break
        axs[2 * i + 1, j].imshow(item, cmap='gray')
        axs[2 * i + 1, j].axis('off')

    # Set the title of each row as the epoch number
    axs[2 * i, 0].set_title(f'(VAE) Epoch {k + 1}')

# Remove empty subplots
for k in range(2 * len(indices), axs.shape[0]):
    for j in range(axs.shape[1]):
        axs[k, j].axis('off')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# Variational Autoencoder Loss curve :
x1 = [i for i in range(1, num_epochs_VAE+1)]
plt.plot(x1, loss_VAE, color='g', label='VAE loss curve')
plt.suptitle('VAE loss curve', fontsize=16)  # Add title at the top
plt.grid()
plt.legend()
plt.show()

# Comparision :
x1=[i for i in range(1,num_epoches+1)]
lines=plt.plot(x1,loss_auto,x1,loss_VAE)
plt.setp(lines[0],color='r',label='Autoencoder loss curve')
plt.setp(lines[1],c='g',linewidth=2.0,label='VAE loss curve')
plt.grid()
plt.suptitle('Comparison', fontsize=16)  # Add title at the top
plt.legend()
plt.show()
