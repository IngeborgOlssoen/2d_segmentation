import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from data_loader import train_loader, val_loader
from UNet import unet_model
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = unet_model.to(device)

# Loss and optimizer
loss_function = DiceLoss(to_onehot_y=False, softmax=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Early stopping parameters
patience = 500  # Number of epochs to wait before stopping if no improvement is seen
best_val_loss = float('inf')
counter = 0  # Counter to keep track of epochs with no improvement

train_losses = []
val_losses = []
# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        images, masks = batch['img'].to(device), batch['seg'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss) 
    
    # Validation
    model.eval()
    total_val_loss = 0
    for batch in val_loader:
        images, masks = batch['img'].to(device), batch['seg'].to(device)
        with torch.no_grad():
            outputs = model(images)
            val_loss = loss_function(outputs, masks)
        total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss) 
    
    print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')

    # Learning rate scheduler step
    scheduler.step(avg_val_loss)

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("Saved best model!")
        counter = 0  # Reset counter
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

print("Training finished!")

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
