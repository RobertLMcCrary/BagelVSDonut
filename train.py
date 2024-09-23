import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
import torch.nn.functional as f
from PIL import Image


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load the dataset
train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)


#define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
for epoch in range(10):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')


# Save the model
torch.save(model.state_dict(), 'bagel_donut_classifier.pth')


# Function to make predictions
def predict(image_path):
    model.load_state_dict(torch.load('bagel_donut_classifier.pth'))
    model.eval()

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    return 'Bagel' if predicted.item() == 0 else 'Donut'


# first prediction should be a bagel
result1 = predict('./Data/train/bagel/bagel.jpeg')
print('1st Prediction: ', result1)

#second prediction should be a donut
result2 = predict('Data/train/donut/donut.jpeg')
print('2nd Prediction: ', result2)