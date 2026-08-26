import torch


def train_one_epoch(
	model,
	loader,
	optimizer,
	criterion,
	device,
):
	model.train()

	running_loss = 0.0

	for images, masks in loader:

		images = images.to(device)
		masks = masks.to(device)

		optimizer.zero_grad()

		logits = model(images)

		loss = criterion(logits, masks)

		loss.backward()
		optimizer.step()

		running_loss += loss.item()

	return running_loss / len(loader)


@torch.no_grad()
def validate(
	model,
	loader,
	criterion,
	device,
):
	model.eval()

	running_loss = 0.0

	total_tp = 0.0
	total_fp = 0.0
	total_fn = 0.0

	for images, masks in loader:

		images = images.to(device)
		masks = masks.to(device)

		logits = model(images)

		loss = criterion(logits, masks)

		running_loss += loss.item()

		predictions = torch.sigmoid(logits) >= 0.5
		targets = masks >= 0.5

		total_tp += (
			(predictions & targets)
			.sum()
			.item()
		)

		total_fp += (
			(predictions & ~targets)
			.sum()
			.item()
		)

		total_fn += (
			(~predictions & targets)
			.sum()
			.item()
		)

	epsilon = 1e-7

	dice = (
		2 * total_tp
		/ (
			2 * total_tp
			+ total_fp
			+ total_fn
			+ epsilon
		)
	)

	iou = (
		total_tp
		/ (
			total_tp
			+ total_fp
			+ total_fn
			+ epsilon
		)
	)

	precision = (
		total_tp
		/ (
			total_tp
			+ total_fp
			+ epsilon
		)
	)

	recall = (
		total_tp
		/ (
			total_tp
			+ total_fn
			+ epsilon
		)
	)

	return {
		"loss": running_loss / len(loader),
		"dice": dice,
		"iou": iou,
		"precision": precision,
		"recall": recall,
	}