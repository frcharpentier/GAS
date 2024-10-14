import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, 
                                    test_size=0.2, random_state=42)

# Define logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

# Initialize the model
input_size = X_train.shape[1]
num_classes = len(torch.unique(y_train))
model = LogisticRegression(input_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 200
for epoch in tqdm(range(num_epochs)):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Predict on test data
with torch.no_grad():
    outputs = model(X_test)
    _, y_pred = torch.max(outputs, 1)

# Convert predictions and labels to numpy arrays
y_pred_np = y_pred.numpy()
y_test_np = y_test.numpy()

# Calculate accuracy
accuracy = accuracy_score(y_test_np, y_pred_np)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test_np, y_pred_np))