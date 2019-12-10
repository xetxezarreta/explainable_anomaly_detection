import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class PCA_model_classification(object):
    def __init__(self, train_data, anomaly_threshold):
        self.pca = PCA(n_components=0.95).fit(train_data)
        self.anomaly_threshold = anomaly_threshold

    def predict(self, data):
        rec_data = self.__reconstruct_data(data)
        loss = self.get_projection_loss(data, rec_data)

        predicted_list = []
        for row in loss:
            error = row.mean()
            predicted = 0
            if error >= self.anomaly_threshold:
                predicted = 1                   
            predicted_list.append(predicted)      
        return np.asarray(predicted_list)
    
    def get_projection_loss(self, data, rec_data):        
        return ((data - rec_data) ** 2)

    def __reconstruct_data(self, data):
        data_transformed = self.pca.transform(data)  
        data_reconstructed = self.pca.inverse_transform(data_transformed)
        return data_reconstructed
