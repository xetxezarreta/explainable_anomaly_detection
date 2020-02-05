import numpy as np
from sklearn.decomposition import PCA

class ReconstructionErrorModel(object):
    def __init__(self, train_data, model=PCA(n_components=0.95)):
        """
        Constructor

        Parameters
        ----------
        train_data: Data used to train component analisys model.
        model: Any sklearn component analisys algorithm.       
        """
        self.model = model.fit(train_data)

    def predict(self, data):
        """
        Makes the inverse transformation with the component analysis algorithm, 
        returning the reconstruction error of each instance.

        Parameters
        ----------
        data: Instances to make prediction.         

        Returns
        -------
        Prediction of the instances.
        """
        rec_data = self.__reconstruct_data(data)
        loss = self.__get_projection_loss(data, rec_data)
        error_list = []
        for _, row in loss.iterrows():
            error = row.mean()       
            error_list.append(error)      
        return np.asarray(error_list)
    
    def __get_projection_loss(self, in_data, out_data):   
        """
        Calculates the projection loss between input and output data.

        Parameters
        ----------
        in_data: Input/Original data.       
        out_data: Output/Reconstructed data.   

        Returns
        -------
        Projection loss.
        """        
        return ((in_data - out_data) ** 2)

    def __reconstruct_data(self, data):
        """
        Perform the transform and the inverse transform of the data, 
        trying to reconstruct the original data.

        Parameters
        ----------
        data: Input/Original data.       

        Returns
        -------
        Output/Reconstructed data.  
        """   
        data_transformed = self.model.transform(data)  
        data_reconstructed = self.model.inverse_transform(data_transformed)
        return data_reconstructed
