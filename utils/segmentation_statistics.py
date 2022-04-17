import numpy as np
import surface_distance as sd # https://github.com/deepmind/surface-distance
import pandas as pd

from tabulate import tabulate


class SegmentationStatistics:

    def __init__(self, prediction, truth, zoom):
        self.prediction = prediction
        self.truth = truth
        self.zoom = zoom
        self.dice = self._dice()
        self.jaccard = self._jaccard()
        self.sensitivity = self._sensitivity()
        self.specificity = self._specificity()
        self.precision = self._precision()
        self.accuracy = self._accuracy()
        self._surface_dist = sd.compute_surface_distances(self.prediction, self.truth, self.zoom)
        self.mean_surface_distance = self._av_dist()
        self.hausdorff_distance = self._hausdorff_dist(95)
        self.volume_difference = self._volume_difference()
        self.dict = self.to_dict()
        self.df = self.to_df()

    def truth_volume(self):
        return np.sum(self.truth) * np.prod(self.zoom) / 1000

    def prediction_volume(self):
        return np.sum(self.prediction) * np.prod(self.zoom) / 1000

    def to_dict(self):
        return {'Dice': self.dice,
                'Jaccard': self.jaccard, 
                'Sensitivity': self.sensitivity,
                'Specificity': self.specificity,
                'Precision': self.precision, 
                'Accuracy': self.accuracy, 
                'Mean_Surface_Distance': self.mean_surface_distance, 
                'Hausdorff_Distance': self.hausdorff_distance, 
                'Volume_Difference': self.volume_difference}

    def to_df(self):
        df = pd.DataFrame(self.dict.items(), columns=['Metric','Score'])
        return df

    def print_table(self):
        print(tabulate(self.df[['Metric', 'Score']], headers=['Metric', 'Score'], tablefmt='github', showindex=False))

    def _dice(self):
        return np.sum(self.prediction[self.truth==1]) * 2.0 / (np.sum(self.prediction) + np.sum(self.truth))

    def _jaccard(self):
        return np.sum(self.prediction[self.truth==1]) / (np.sum(self.prediction[self.truth==1]) + np.sum(self.prediction!=self.truth))

    def _sensitivity(self):
        return np.sum(self.prediction[self.truth==1])/(np.sum(self.prediction[self.truth==1])+np.sum((self.truth==1) & (self.prediction==0)))

    def _specificity(self):
        return np.sum((self.truth==0) & (self.prediction==0))/(np.sum((self.truth==0) & (self.prediction==0))+np.sum((self.truth==0) & (self.prediction==1)))

    def _precision(self):
        return np.sum(self.prediction[self.truth==1])/(np.sum(self.prediction[self.truth==1]) + np.sum((self.truth==0) & (self.prediction==1)))

    def _accuracy(self):
        return (np.sum(self.prediction[self.truth==1]) + np.sum((self.truth==0) & (self.prediction==0))) / self.truth.size

    def _av_dist(self):
        av_surf_dist = sd.compute_average_surface_distance(self._surface_dist)
        return np.mean(av_surf_dist)

    def _hausdorff_dist(self, percentile=95):
        return sd.compute_robust_hausdorff(self._surface_dist, percentile)

    def _volume_difference(self):
        return self.prediction_volume() - self.truth_volume()

