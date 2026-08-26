import torch
import torch.nn as nn


class DiceLoss(nn.Module):
	def __init__(self, smooth=1.0):
		super().__init__()
		self.smooth = smooth

	def forward(self, logits, targets):

		probabilities = torch.sigmoid(logits)

		probabilities = probabilities.contiguous().view(
			probabilities.size(0), -1
		)

		targets = targets.contiguous().view(
			targets.size(0), -1
		)

		intersection = (
			probabilities * targets
		).sum(dim=1)

		dice = (
			2.0 * intersection + self.smooth
		) / (
			probabilities.sum(dim=1)
			+ targets.sum(dim=1)
			+ self.smooth
		)

		return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
	def __init__(self, dice_weight=1.0, bce_weight=1.0):
		super().__init__()

		self.dice = DiceLoss()
		self.bce = nn.BCEWithLogitsLoss()

		self.dice_weight = dice_weight
		self.bce_weight = bce_weight

	def forward(self, logits, targets):

		dice_loss = self.dice(logits, targets)
		bce_loss = self.bce(logits, targets)

		return (
			self.dice_weight * dice_loss
			+ self.bce_weight * bce_loss
		)