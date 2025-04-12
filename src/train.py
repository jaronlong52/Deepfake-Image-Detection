import torch
import torch.optim as optim
import torch.nn as nn
from model.xception_model import XceptionNet
from utils.data_utils import get_data_loaders
from sklearn.metrics import accuracy_score

# Define the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up paths and hyperparameters
data_dir = "data/"  # Make sure this points to the correct data directory
batch_size = 32
num_epochs = 10
learning_rate = 1e-4

# Load the data
train_loader, val_loader, _ = get_data_loaders(data_dir, batch_size)

# Initialize the model, loss function, and optimizer
model = XceptionNet(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:  # No need for the batch length check here
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Validate the model
    model.eval()
    val_labels = []
    val_preds = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(val_labels, val_preds)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
