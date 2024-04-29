import torch
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.losses import DiceCELoss #focal og dice
from monai.metrics import DiceMetric
from data_loader import train_loader, val_loader
from UNet import unet_model
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import sigmoid
from torch.optim.lr_scheduler import CyclicLR


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = unet_model.to(device)



# Loss and optimizer
loss_function = DiceCELoss(to_onehot_y=False, softmax=False).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,weight_decay=0.01)

base_lr = 1e-5  # Set this to a lower bound of learning rate
max_lr = 1e-3   # Set this to an upper bound of learning rate

# Learning rate scheduler
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Early stopping parameters
patience = 20
best_val_loss = float('inf')
counter = 0

train_losses = []
val_losses = []



# Training loop
num_epochs = 20
scheduler = CosineAnnealingLR(optimizer=optimizer,T_max=num_epochs,eta_min=1e-6)



for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        images, masks = batch['img'].to(device), batch['seg'].to(device)
        # Global list to store feature maps

        
        optimizer.zero_grad()
        outputs = model(images)
        outputs = sigmoid(outputs)  # Applying sigmoid activation
        loss = loss_function(outputs, masks)
        loss.backward()
        #if epoch%5==0:
        #    plot_gradients(model)
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    #Validation
    model.eval()
    total_val_loss = 0
    for batch in val_loader:
        images, masks = batch['img'].to(device), batch['seg'].to(device)
        with torch.no_grad():
            outputs = model(images)
            outputs = sigmoid(outputs)  # Applying sigmoid activation
            val_loss = loss_function(outputs, masks)
        total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}, LR: {scheduler.get_last_lr()[0]}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

    # Learning rate scheduler step
    scheduler.step()

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("Saved best model!")
        counter = 0
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


