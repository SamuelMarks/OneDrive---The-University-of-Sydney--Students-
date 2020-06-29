""" Implementation of ml_params API """

from ml_params.base import BaseTrainer


class TraxTrainer(BaseTrainer):
    """ Implementation of ml_params BaseTrainer for Trax """
    data = None

    def load_data(self, dataset_name, data_loader, data_type='tensorflow_datasets', output_type='numpy'):
        self.data = super(TraxTrainer, self).load_data(dataset_name=dataset_name,
                                                       data_loader=data_loader,
                                                       data_type=data_type,
                                                       output_type=output_type)

    def train(self, epochs, *args, **kwargs):
        super(TraxTrainer, self).train(epochs, *args, **kwargs)
        assert self.data is not None
        raise NotImplementedError()


del BaseTrainer

__all__ = ['TraxTrainer']
