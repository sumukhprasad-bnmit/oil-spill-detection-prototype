import torch
from torch.optim import Adam
import time

from dataset import OilSpillDataset
from losses import DiceBCELoss
from model import UNet
from training import train_one_epoch, validate

# device
if torch.backends.mps.is_available():
	device = torch.device("mps")
elif torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

print(f"device: {device}")


# datasets
train_dataset = OilSpillDataset(
	"../data/images/train",
	"../data/masks/train",
)

val_dataset = OilSpillDataset(
	"../data/images/val",
	"../data/masks/val",
)


# loaders
train_loader = torch.utils.data.DataLoader(
	train_dataset,
	batch_size=4,
	shuffle=True,
	num_workers=0,
)

val_loader = torch.utils.data.DataLoader(
	val_dataset,
	batch_size=4,
	shuffle=False,
	num_workers=0,
)


# model
model = UNet().to(device)


# loss + optimizer
criterion = DiceBCELoss()

optimizer = Adam(
	model.parameters(),
	lr=1e-3,
)


# training
epochs = 1

best_dice = 0.0


start_g = time.time()
print(f'Start time: {start_g}')
print("---")

for epoch in range(epochs):
	start_e = time.time()
	print(f'Start time for epoch {epoch + 1:02d}: {start_e}')
	
	train_loss = train_one_epoch(
		model,
		train_loader,
		optimizer,
		criterion,
		device,
	)

	val_metrics = validate(
		model,
		val_loader,
		criterion,
		device,
	)

	print(
		f"Epoch {epoch + 1:02d}/{epochs} | "
		f"train loss: {train_loss:.4f} | "
		f"val loss: {val_metrics['loss']:.4f} | "
		f"dice: {val_metrics['dice']:.4f} | "
		f"iou: {val_metrics['iou']:.4f} | "
		f"precision: {val_metrics['precision']:.4f} | "
		f"recall: {val_metrics['recall']:.4f}"
	)

	# save best model
	if val_metrics["dice"] > best_dice:
		best_dice = val_metrics["dice"]

		torch.save(
			model.state_dict(),
			"best_model.pt",
		)

		print(
			f"--> saved best model "
			f"(dice={best_dice:.4f})"
		)
	
	end_e = time.time()
	print(f'End time for epoch {epoch + 1:02d}: {end_e}')

print("---")
end_g = time.time()
print(f'End time: {end_g}')