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
        in_data = np.asarray(data)       
        out_data = self.__reconstruct_data(in_data)        
        errors = np.mean((in_data - out_data)**2, axis=1)
        return np.asarray(errors)
    
    def get_model(self):
        """
        Returns the model used for reconstruction error.        

        Returns
        -------
        The model.
        """
        return self.model

    def __reconstruct_data(self, data):
        """
        Performs the transform and the inverse transform of the data, 
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
