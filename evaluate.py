import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import EnsureType
from data_loader import  val_loader # make sure your validation loader is suitable for PyTorch
from UNet import unet_model

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = unet_model.to(device)

# Metrics setup using MONAI
dice_metric = DiceMetric(include_background=False, reduction="mean")  # Adjust based on your model's output
hd_metric = HausdorffDistanceMetric(percentile=95)

# Ensure correct data type for MONAI metrics
ensure_type = EnsureType(device=device)

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    dice_scores = []
    hausdorff_distances = []

    # Randomly select 5 batches for visualization
    visualize_indices = np.random.choice(len(data_loader), size=5, replace=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            images, true_masks = batch['img'].to(device), batch['seg'].to(device)
            images, true_masks = ensure_type(images), ensure_type(true_masks)
            
            predicted_masks = model(images)
            # Binarizing the outputs for metric calculation
            predicted_masks = (predicted_masks > 0.8).float()
            true_masks = true_masks.float()

            # Update metrics
            dice_metric(y_pred=predicted_masks, y=true_masks)
            hd_metric(y_pred=predicted_masks, y=true_masks)

            # Visualize randomly selected batches
            if batch_idx in visualize_indices:
                # Plot the images, true masks, and predicted masks
                plot_batch(images, true_masks, predicted_masks, batch_idx)
            
            # Evaluate metrics every batch
            dice_scores.append(dice_metric.aggregate().item())
            hausdorff_distances.append(hd_metric.aggregate().item())
            dice_metric.reset()
            hd_metric.reset()

    return np.mean(dice_scores), np.mean(hausdorff_distances)

def plot_batch(images, true_masks, predicted_masks, batch_idx):
    # Move tensors from GPU to CPU and convert to numpy arrays
    images = images.cpu().numpy()
    true_masks = true_masks.cpu().numpy()
    predicted_masks = predicted_masks.cpu().numpy()

    # Plot the first image, true mask, and predicted mask
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(images[0, 0], cmap='gray')
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(true_masks[0, 0], cmap='gray')
    plt.title('True Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(predicted_masks[0, 0], cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()  # Display the visualization momentarily
    plt.close()

# Running the evaluation
dice_score, hausdorff_distance = evaluate_model(model, val_loader)
print(f"Average Dice Score: {dice_score}")
print(f"Average Hausdorff Distance 95th Percentile: {hausdorff_distance}")
