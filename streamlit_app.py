from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.utils as utils
from torchvision.utils import save_image


class DeblurCNN(nn.Module):
    def __init__(self):
        super(DeblurCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


device = 'cpu'


def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)


def predict(model, image_tensor):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        sharp_tensor = model(image_tensor)
    return sharp_tensor


def deblurimage(image):
    # import torch

    # Load the saved model state dictionary
    model_state_dict = torch.load('/content/drive/MyDrive/idb/srup/model21.pth')

    # Create a new instance of the model with the same architecture as the original model
    model = DeblurCNN()
    model.load_state_dict(model_state_dict)

    # Set the model to evaluation mode
    model.eval()
    blur_image = Image.open(image)
    # blur_image.show()
    # Convert grayscale to RGB
    if blur_image.mode == 'L':
        blur_image = blur_image.convert('RGB')

    # Apply transformations to the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    tensor_img = transform(blur_image)

    # create a batch of size 1
    blur_tensor = tensor_img.unsqueeze(0)

    # predict the sharp image
    sharp_tensor = predict(model, blur_tensor)

    # save the output image
    img = sharp_tensor.view(sharp_tensor.size(0), 3, 224, 224)
    save_decoded_image(img, "/content/drive/MyDrive/idb/savedimg.jpg")
