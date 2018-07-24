from .ani_model import ANIModel


class CustomModel(ANIModel):

    def __init__(self, aev_computer, per_species, reducer,
                 derivative=False, derivative_graph=False, benchmark=False):
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
        output_length = None
        models = {}
        for i in per_species:
            model_X = per_species[i]
            if not hasattr(model_X, 'output_length'):
                raise ValueError(
                    '''atomic neural network must explicitly specify
                    output length''')
            elif output_length is None:
                output_length = model_X.output_length
            elif output_length != model_X.output_length:
                raise ValueError(
                    '''output length of each atomic neural network must
                    match''')
            setattr(self, 'model_' + i, model_X)
        super(CustomModel, self).__init__(aev_computer, suffixes, reducer,
                                          output_length, models, derivative,
                                          derivative_graph, benchmark)
