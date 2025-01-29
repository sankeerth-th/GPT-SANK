import torch
import torch.nn as nn

# 1) Check for MPS availability
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 2) Define a tiny neural net
#    (just a single linear layer for demonstration)
class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.linear = nn.Linear(10, 1)  # 10 inputs -> 1 output

    def forward(self, x):
        return self.linear(x)

# 3) Instantiate the network and move it to the device
model = TinyNet().to(device)

# 4) Create dummy data
#    We'll feed the network a batch of 5 samples, each with 10 features
x = torch.randn(5, 10).to(device)
y = torch.randn(5, 1).to(device)

# 5) Set up a simple loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 6) Training loop (just a few steps to illustrate)
for step in range(5):
    # Forward pass
    preds = model(x)
    loss = criterion(preds, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Step {step+1} - Loss: {loss.item():.4f}")

print("Done training on device:", device)
