import torch
import torch.nn.functional as F
from utils.segmentation_statistics import SegmentationStatistics
from utils.utils import compute_average


def evaluate(net, dataloader, device, threshold, show_stat=False):
    """Evaluate the model using test data using different evaluation metrics (check utils/segmentation_statistics.py)
    Args:
        net (torch.nn.Module): Trained model
        dataloader (DataLoader): Test data loader
        device (torch.device): Device (cpu or cuda)
        show stat (bool): Show the statistical result based on dataset cohort
    Returns:
        stats (dict): the average evaluation metrics
    """

    stats = []
    net.eval()
    num_val_batches = len(dataloader)

	# Disable grad
    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.float().to(device), target.float().to(device)
            output = net(data)

            preds = (output > threshold).float()

			# Convert to numpy boolean
            preds = preds.cpu().numpy()
            target = target.cpu().numpy()	
            preds = preds.astype(bool)
            target = target.astype(bool)

            batch, channel, depth, width, height = preds.shape

            for idx in range(batch):  
                stat = SegmentationStatistics(preds[idx,0,:,:,:], target[idx,0,:,:,:], (3,2,1))
                stats.append(stat.to_dict())

    if show_stat:
		# Average
        print("All:")
        print(compute_average(stats, dataframe=True))

		# HC
        print("\nHealthy Control:")
        print(compute_average(stats,0,25,dataframe=True))

		# CKD
        print("\nChronic Kidney Disease:")
        print(compute_average(stats,25,None,dataframe=True))

        
    return compute_average(stats)
