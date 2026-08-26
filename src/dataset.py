from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset



class OilSpillDataset(Dataset):
	def __init__(self, image_dir, mask_dir):
		self.image_dir = Path(image_dir)
		self.mask_dir = Path(mask_dir)

		self.image_paths = sorted(self.image_dir.glob("*.png"))
	

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		image_path = self.image_paths[idx]
		mask_path = self.mask_dir / image_path.name

		# Load grayscale image.
		image = np.array(
			Image.open(image_path).convert("L"),
			dtype=np.float32
		)

		# Load grayscale mask.
		mask = np.array(
			Image.open(mask_path).convert("L"),
			dtype=np.float32
		)

		# Normalize SAR image from [0, 255] --> [0, 1].
		image /= 255.0

		# Convert antialiased mask into binary mask.
		mask = (mask >= 128).astype(np.float32)

		# Add channel dimension:
		# [H, W] → [1, H, W]
		image = torch.from_numpy(image).unsqueeze(0)
		mask = torch.from_numpy(mask).unsqueeze(0)

		return image, mask