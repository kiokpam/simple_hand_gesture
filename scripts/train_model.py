import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
from pathlib import Path

# Append the root path of the project to sys.path
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))

from utils.data_utils import NPYDataset
from models.linear_model import LinearModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

features_path = r'data\npy\train_hand_keypoints.npy'
labels_path = r'data\npy\train_hand_labels.npy'
test_features_path = r'data\npy\test_hand_keypoints.npy'
test_labels_path = r'data\npy\test_hand_labels.npy'

features = np.load(features_path)
labels = np.load(labels_path)
test_features = np.load(test_features_path)
test_labels = np.load(test_labels_path)

print(f'Features shape: {features.shape}, Labels shape: {labels.shape}, Test Features shape: {test_features.shape}, Test Labels shape: {test_labels.shape}')

dataset = NPYDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
model = LinearModel(num_classes=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = torch.nn.CrossEntropyLoss()
num_epochs = 10
for epoch in range(num_epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    train_loss = 0
    accuracy = 0
    best_accuracy = 0
    for i, (batch_features, batch_labels) in progress_bar:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        outputs = model(batch_features)
        loss = loss_function(outputs, batch_labels)
        accuracy += (torch.argmax(outputs, 1) == batch_labels).float().mean()
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.set_description(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy/(i+1):.4f}')
    checkpoint = {'model': model,
                  'state_dict': model.state_dict(),
                  'optimizer' : optimizer.state_dict()}
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(checkpoint, 'checkpoints/best.pth')
    torch.save(checkpoint, 'checkpoints/last.pth')
    model.eval()
    with torch.no_grad():
        test_features = torch.tensor(test_features, dtype=torch.float32).to(device)
        test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)
        outputs = model(test_features)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == test_labels).sum().item()
        test_accuracy = correct / len(test_labels)
        print(f'Test Accuracy: {test_accuracy:.4f}')
    