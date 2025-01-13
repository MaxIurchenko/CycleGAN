# CycleGAN Implementation

import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.utils import save_image

class PairedDataset(Dataset):
    def __init__(self, dataset_X, dataset_Y):
        self.dataset_X = dataset_X
        self.dataset_Y = dataset_Y

    def __len__(self):
        return min(len(self.dataset_X), len(self.dataset_Y))  # Ensure the dataset length is the same

    def __getitem__(self, idx):
        # Get one sample from each dataset
        image_X, _ = self.dataset_X[idx]
        image_Y, _ = self.dataset_Y[idx]
        return image_X, image_Y

# ----------------------
# Generator Network
# ----------------------
class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, num_residual_blocks=9):
        super(Generator, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling layers
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResNetBlock(in_features)]

        # Upsampling layers
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ----------------------
# Discriminator Network
# ----------------------
class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(3):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        model += [
            nn.Conv2d(in_features, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ----------------------
# Loss Functions
# ----------------------
class CycleGANLoss:
    def __init__(self, device, lambda_cycle=10, lambda_identity=0.5):
        self.adversarial_loss = nn.MSELoss()  # Can also use LSGAN
        self.cycle_loss = nn.L1Loss()
        self.identity_loss = nn.L1Loss()
        self.device = device
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compute_adversarial_loss(self, predictions, target_is_real):
        target = torch.ones_like(predictions) if target_is_real else torch.zeros_like(predictions)
        return self.adversarial_loss(predictions, target)

    def compute_cycle_loss(self, real_image, cycled_image):
        return self.cycle_loss(real_image, cycled_image)

    def compute_identity_loss(self, real_image, same_image):
        return self.identity_loss(real_image, same_image)


# ----------------------
# Dataset Loader
# ----------------------
def load_data(domain_X_dir, domain_Y_dir, image_size=256, batch_size=1):
    # Define the transformation for the images
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load images from the 'horses' domain (X)
    dataset_X = datasets.ImageFolder(domain_X_dir, transform=transform)

    # Load images from the 'zebras' domain (Y)
    dataset_Y = datasets.ImageFolder(domain_Y_dir, transform=transform)

    # Create a paired dataset
    paired_dataset = PairedDataset(dataset_X, dataset_Y)

    # Create a DataLoader for the paired dataset
    return DataLoader(paired_dataset, batch_size=batch_size, shuffle=True)


# ----------------------
# Save and Plot Images
# ----------------------
def save_and_plot_images(G, F, test_input, epoch, output_dir='output_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        fake_image_G = G(test_input)
        fake_image_F = F(fake_image_G)

    save_image(fake_image_G, os.path.join(output_dir, f'fake_image_G_epoch_{epoch}.png'))
    save_image(fake_image_F, os.path.join(output_dir, f'fake_image_F_epoch_{epoch}.png'))
    save_image(test_input, os.path.join(output_dir, f'test_image_epoch_{epoch}.png'))

def save_training_statistics(stats, output_file='training_stats.txt'):
    with open(output_file, 'a') as f:
        f.write(f"Epoch {stats['epoch']}:\n")
        f.write(f"Generator Loss: {stats['G_loss']}\n")
        f.write(f"Discriminator X Loss: {stats['D_X_loss']}\n")
        f.write(f"Discriminator Y Loss: {stats['D_Y_loss']}\n\n")


# ----------------------
# Save Model Function
# ----------------------
def save_model(epoch, G, F, D_X, D_Y, optimizer_G, optimizer_D_X, optimizer_D_Y, loss, path='model_checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_G_state_dict': G.state_dict(),
        'model_F_state_dict': F.state_dict(),
        'model_D_X_state_dict': D_X.state_dict(),
        'model_D_Y_state_dict': D_Y.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_X_state_dict': optimizer_D_X.state_dict(),
        'optimizer_D_Y_state_dict': optimizer_D_Y.state_dict(),
        'loss': loss
    }, path)


# ----------------------
# Load Model Function
# ----------------------
def load_model(path, G, F, D_X, D_Y, optimizer_G, optimizer_D_X, optimizer_D_Y):
    checkpoint = torch.load(path)
    G.load_state_dict(checkpoint['model_G_state_dict'])
    F.load_state_dict(checkpoint['model_F_state_dict'])
    D_X.load_state_dict(checkpoint['model_D_X_state_dict'])
    D_Y.load_state_dict(checkpoint['model_D_Y_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D_X.load_state_dict(checkpoint['optimizer_D_X_state_dict'])
    optimizer_D_Y.load_state_dict(checkpoint['optimizer_D_Y_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss



# ----------------------
# Training Loop
# ----------------------
def train(cycle_gan, dataloader, num_epochs, device):
    G_X = cycle_gan['G_X']  # Generator for domain X
    G_Y = cycle_gan['G_Y']  # Generator for domain Y
    D_X = cycle_gan['D_X']  # Discriminator for domain X
    D_Y = cycle_gan['D_Y']  # Discriminator for domain Y
    optimizer_G = cycle_gan['optimizer_G']
    optimizer_D_X = cycle_gan['optimizer_D_X']
    optimizer_D_Y = cycle_gan['optimizer_D_Y']

    cycle_gan_loss = CycleGANLoss(device)

    for epoch in range(num_epochs):
        G_loss_total, D_X_loss_total, D_Y_loss_total = 0.0, 0.0, 0.0

        for i, (real_images_X, real_images_Y) in enumerate(dataloader):
            real_images_X = real_images_X.to(device)
            real_images_Y = real_images_Y.to(device)

            # ---------------------
            # Update Generators
            # ---------------------
            optimizer_G.zero_grad()

            # G_X: X -> Y
            fake_Y = G_X(real_images_X)
            # G_Y: Y -> X
            fake_X = G_Y(real_images_Y)

            # Cycle consistency
            cycled_X = G_Y(fake_Y)
            cycled_Y = G_X(fake_X)

            # Identity loss (optional)
            identity_X = G_Y(real_images_X)
            identity_Y = G_X(real_images_Y)

            # Generator losses
            G_X_loss = cycle_gan_loss.compute_adversarial_loss(D_Y(fake_Y), True)
            G_Y_loss = cycle_gan_loss.compute_adversarial_loss(D_X(fake_X), True)

            # Cycle consistency losses
            cycle_loss_X = cycle_gan_loss.compute_cycle_loss(real_images_X, cycled_X)
            cycle_loss_Y = cycle_gan_loss.compute_cycle_loss(real_images_Y, cycled_Y)

            # Identity losses
            identity_loss_X = cycle_gan_loss.compute_identity_loss(real_images_X, identity_X)
            identity_loss_Y = cycle_gan_loss.compute_identity_loss(real_images_Y, identity_Y)

            G_loss = G_X_loss + G_Y_loss + cycle_loss_X * cycle_gan_loss.lambda_cycle + cycle_loss_Y * cycle_gan_loss.lambda_cycle + identity_loss_X * cycle_gan_loss.lambda_identity + identity_loss_Y * cycle_gan_loss.lambda_identity

            G_loss.backward()
            optimizer_G.step()

            # ---------------------
            # Update Discriminators
            # ---------------------
            optimizer_D_X.zero_grad()
            optimizer_D_Y.zero_grad()

            # Discriminator losses
            D_X_loss = cycle_gan_loss.compute_adversarial_loss(D_X(real_images_X), True) + cycle_gan_loss.compute_adversarial_loss(D_X(fake_X.detach()), False)
            D_Y_loss = cycle_gan_loss.compute_adversarial_loss(D_Y(real_images_Y), True) + cycle_gan_loss.compute_adversarial_loss(D_Y(fake_Y.detach()), False)

            D_X_loss.backward()
            D_Y_loss.backward()
            optimizer_D_X.step()
            optimizer_D_Y.step()

            G_loss_total += G_loss.item()
            D_X_loss_total += D_X_loss.item()
            D_Y_loss_total += D_Y_loss.item()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Step [{i}/{len(dataloader)}] "
                      f"G Loss: {G_loss.item():.4f} "
                      f"D_X Loss: {D_X_loss.item():.4f} "
                      f"D_Y Loss: {D_Y_loss.item():.4f}")

        # Save images and statistics periodically
        save_and_plot_images(cycle_gan['G_X'], cycle_gan['G_Y'], real_images_X[0].unsqueeze(0), epoch, output_dir="output")
        save_training_statistics({
            'epoch': epoch,
            'G_loss': G_loss_total / len(dataloader),
            'D_X_loss': D_X_loss_total / len(dataloader),
            'D_Y_loss': D_Y_loss_total / len(dataloader),
        }, output_file="training_stats.txt")

        # Save the model checkpoint periodically
        save_model(epoch, G_X, G_Y, D_X, D_Y, optimizer_G, optimizer_D_X, optimizer_D_Y, G_loss_total / len(dataloader))



# ----------------------
# Main Function
# ----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize generators and discriminators
    G = Generator().to(device)  # Generator for domain X
    F = Generator().to(device)  # Generator for domain Y
    D_X = Discriminator().to(device)  # Discriminator for domain X
    D_Y = Discriminator().to(device)  # Discriminator for domain Y

    # Initialize optimizers
    optimizer_G = optim.Adam(list(G.parameters()) + list(F.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_X = optim.Adam(D_X.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_Y = optim.Adam(D_Y.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Load data
    data_dir_x = "./data/A"  # Path to domain X
    data_dir_y = "./data/B"  # Path to domain Y
    dataloader = load_data(data_dir_x, data_dir_y)

    # Create the cycle_gan dictionary
    cycle_gan = {
        'G_X': G,
        'G_Y': F,
        'D_X': D_X,
        'D_Y': D_Y,
        'optimizer_G': optimizer_G,
        'optimizer_D_X': optimizer_D_X,
        'optimizer_D_Y': optimizer_D_Y
    }

    # Train CycleGAN
    train(cycle_gan, dataloader, num_epochs=200, device=device)

def test_image(image_path, output_dir='output_images', model_path='model_checkpoint.pth', device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained models
    G_X = Generator().to(device)
    F = Generator().to(device)
    checkpoint = torch.load(model_path)
    G_X.load_state_dict(checkpoint['model_G_state_dict'])
    F.load_state_dict(checkpoint['model_F_state_dict'])

    # Set the model to evaluation mode
    G_X.eval()
    F.eval()

    # Preprocess the input image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to the size used during training
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image to [-1, 1] range
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')  # Open the image
    image = transform(image).unsqueeze(0).to(device)  # Apply transformation and add batch dimension

    # Generate the transformed image using the generator
    with torch.no_grad():
        fake_image = G_X(image)  # Generate an image using the G_X generator (X -> Y)

    # Save and plot the generated image
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_image(fake_image, os.path.join(output_dir, 'generated_image.png'))

    print("Test image saved to:", os.path.join(output_dir, 'generated_image.png'))


if __name__ == "__main__":
    main()

# # Example usage for test:
# if __name__ == "__main__":
#     test_image('path_to_your_input_image.jpg')  # Provide your test image path here
