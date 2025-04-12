import torch
from model.xception_model import XceptionNet
from utils.data_utils import get_data_loaders
from sklearn.metrics import accuracy_score

# Define the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = XceptionNet(num_classes=2).to(device)
model.load_state_dict(torch.load("xception_model.pth"))  # Load the trained model

# Load the test data
_, _, test_loader = get_data_loaders("data/dataset", batch_size=32)

# Test the model
model.eval()
test_labels = []
test_preds = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_labels.extend(labels.cpu().numpy())
        test_preds.extend(preds.cpu().numpy())

# Calculate the accuracy
accuracy = accuracy_score(test_labels, test_preds)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
