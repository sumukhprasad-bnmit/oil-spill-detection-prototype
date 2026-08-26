import torch


def segmentation_metrics(logits, targets, threshold=0.5):
	probabilities = torch.sigmoid(logits)
	predictions = probabilities >= threshold

	targets = targets >= 0.5

	predictions = predictions.flatten()
	targets = targets.flatten()

	tp = (predictions & targets).sum().float()
	fp = (predictions & ~targets).sum().float()
	fn = (~predictions & targets).sum().float()

	epsilon = 1e-7

	dice = (
		2 * tp
		/ (2 * tp + fp + fn + epsilon)
	)

	iou = (
		tp
		/ (tp + fp + fn + epsilon)
	)

	precision = (
		tp
		/ (tp + fp + epsilon)
	)

	recall = (
		tp
		/ (tp + fn + epsilon)
	)

	return {
		"dice": dice.item(),
		"iou": iou.item(),
		"precision": precision.item(),
		"recall": recall.item(),
	}