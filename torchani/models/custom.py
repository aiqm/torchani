import torch
from .ani_model import ANIModel


class CustomModel(ANIModel):

    def __init__(self, per_species, reducer=torch.sum, padding_fill=0,
                 derivative=False, derivative_graph=False):
        """Custom single model, no ensemble

        Parameters
        ----------
        per_species : dict
            Dictionary with supported species as keys and objects of
            `torch.nn.Model` as values, storing the model for each supported
            species. These models will finally become `model_X` attributes.
        reducer : function
            The desired `reducer` attribute.
        """
        suffixes = ['']
        models = {}
        for i in per_species:
            models['model_' + i] = per_species[i]
        super(CustomModel, self).__init__(list(per_species.keys()), suffixes,
                                          reducer, padding_fill, models)
