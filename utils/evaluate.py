import torch
import numpy as np
from utils.segmentation_statistics import SegmentationStatistics
from utils.utils import compute_average, bland_altman_plot


def evaluate(net, dataloader, device, threshold, show_stat=False, plot=False):
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
    truth_vol = np.array([])
    pred_vol = np.array([])
    net.eval()

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
                stat = SegmentationStatistics(preds[idx,0,:,:,:], target[idx,0,:,:,:], (5,1.5,1.5))
                pred_vol = np.append(pred_vol, stat.prediction_volume())
                truth_vol = np.append(truth_vol, stat.truth_volume())
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


    if plot:
        # Average
        bland_altman_plot(pred_vol,truth_vol,'Mean Volume', 'Absolute Volume Difference (mL)', 'Average TKV Between Prediction and Ground Truth', savefig=True, filename='average')

        # HC
        bland_altman_plot(pred_vol[0:25],truth_vol[0:25],'Mean Volume', 'Absolute Volume Difference (mL)', 'TKV Between Prediction and Ground Truth in Healthy Control Set', savefig=True, filename='hc')

        # CKD
        bland_altman_plot(pred_vol[25:],truth_vol[25:],'Mean Volume', 'Absolute Volume Difference (mL)', 'TKV Between Prediction and Ground Truth in Chronic Kidney Disease Set', savefig=True, filename='ckd')

        
    return compute_average(stats)
