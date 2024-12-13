
## Import Necessary Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Utility Functions
def preprocess_data(data):
    """
    Preprocess the input data by normalizing and encoding as needed.
    Args:
        data (DataFrame): The input data.
    Returns:
        tuple: Preprocessed features and labels.
    """
    # Example preprocessing logic
    features = data.drop(columns=['label'])
    labels = data['label']
    features = (features - features.mean()) / features.std()
    return features.values, labels.values

## Dataset Preparation
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        """
        Custom PyTorch Dataset.
        Args:
            data (ndarray): Feature matrix.
            labels (ndarray): Target labels.
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Load and preprocess data
data = pd.read_csv('data.csv')  # Example data loading
X, y = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DataLoader objects
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

## Model Definition
class TransformerRecommender(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, num_classes):
        """
        Transformer-based recommender model.
        Args:
            input_dim (int): Input feature dimension.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            hidden_dim (int): Hidden layer dimension.
            num_classes (int): Number of output classes.
        """
        super(TransformerRecommender, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x

# Initialize model
input_dim = X_train.shape[1]
num_classes = len(np.unique(y))
model = TransformerRecommender(input_dim=input_dim, num_heads=4, num_layers=2, hidden_dim=256, num_classes=num_classes)

## Training Loop
def train_model(model, dataloader, criterion, optimizer, epochs):
    """
    Train the model.
    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        epochs (int): Number of epochs.
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

## Evaluation
def evaluate_model(model, dataloader):
    """
    Evaluate the model.
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for test data.
    Returns:
        float: Accuracy of the model.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in dataloader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

# Training and Evaluation
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
train_model(model, train_loader, criterion, optimizer, epochs)
evaluate_model(model, test_loader)
