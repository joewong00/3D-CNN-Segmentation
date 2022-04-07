import torch
import torch.nn.functional as F
from utils.segmentation_statistics import SegmentationStatistics
from utils.utils import compute_average


def evaluate(net, dataloader, device, show_stat=False):

    stats = []
    net.eval()
    num_val_batches = len(dataloader)

	# Disable grad
    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.float().to(device), target.float().to(device)
            output = net(data)

            preds = (F.sigmoid(output) > 0.5).float()

			# Convert to numpy boolean
            preds = preds.cpu().numpy()
            target = target.cpu().numpy()	
            preds = preds.astype(bool)
            target = target.astype(bool)

            batch, channel, depth, width, height = preds.shape

            for idx in len(batch):  
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
