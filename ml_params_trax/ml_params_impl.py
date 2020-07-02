""" Implementation of ml_params API """

# Trax specifics mostly taken from https://github.com/google/trax/blob/dcf806d/trax/supervised/mnist_test.py

import itertools
from os import path

from ml_params.base import BaseTrainer
from ml_prepare.datasets import datasets2classes
from ml_prepare.exectors import build_tfds_dataset
from trax import layers as tl
from trax.optimizers import adafactor
from trax.supervised import inputs
from trax.supervised import tf_inputs
from trax.supervised import training

from ml_params_trax import get_logger

logger = get_logger('.'.join((path.basename(path.dirname(__file__)),
                              path.basename(__file__).rpartition('.')[0])))


class TraxTrainer(BaseTrainer):
    """ Implementation of ml_params BaseTrainer for Trax """

    data = None  # type: (None or Tuple[tf.data.Dataset, tf.data.Dataset] or trax.supervised.Inputs)
    model = None  # contains the model, e.g., a `tl.Serial`

    def __init__(self, model, **model_kwargs):
        super(TraxTrainer, self).__init__()
        self.model = model(**model_kwargs)

    def load_data(self, dataset_name, data_loader=None,
                  data_loader_kwargs=None, data_type='infer',
                  output_type=None, K=None):
        """
        Load the data for your ML pipeline. Will be fed into `train`.

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param data_loader: function that returns the expected data type.
         Defaults to TensorFlow Datasets and ml_prepare combined one.
        :type data_loader: ```None or (*args, **kwargs) -> tf.data.Datasets or Any```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```None or dict```

        :param data_type: incoming data type, defaults to 'infer'
        :type data_type: ```str```

        :param output_type: outgoing data_type, defaults to no conversion
        :type output_type: ```None or 'numpy'```

        :param K: backend engine, e.g., `np` or `tf`
        :type K: ```None or np or tf or Any```

        :return: Dataset splits (by default, your train and test)
        :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
        """
        self.data = super(TraxTrainer, self).load_data(dataset_name=dataset_name,
                                                       data_loader=self.load_data_from_trax_tfds_or_ml_prepare,
                                                       data_type=data_type,
                                                       output_type=output_type)
        # self.data = trax.supervised.Inputs(*self.data)
        # trax.supervised.inputs.dataset_to_stream(self.data, dataset_name)

    @staticmethod
    def load_data_from_trax_tfds_or_ml_prepare(dataset_name, tensorflow_datasets_dir=None, data_loader_kwargs=None):
        """
        Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param tensorflow_datasets_dir: directory to look for models in. Default is ~/tensorflow_datasets.
        :type tensorflow_datasets_dir: ```None or str```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```None or dict```

        :return: Train and tests dataset splits
        :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
        """
        if data_loader_kwargs is None:
            data_loader_kwargs = {}
        data_loader_kwargs.update({
            'dataset_name': dataset_name,
            'tfds_dir': tensorflow_datasets_dir
        })
        ds_builder = (build_tfds_dataset if dataset_name in datasets2classes
                      else TraxTrainer.dataset_from_trax_tfds)(
            **data_loader_kwargs
        )

        if hasattr(ds_builder, 'download_and_prepare_kwargs'):
            download_and_prepare_kwargs = getattr(ds_builder, 'download_and_prepare_kwargs')
            delattr(ds_builder, 'download_and_prepare_kwargs')
        else:
            download_and_prepare_kwargs = None

        return BaseTrainer.common_dataset_handler(
            ds_builder=ds_builder,
            download_and_prepare_kwargs=download_and_prepare_kwargs,
            scale=None, K=None, as_numpy=False
        )

    @staticmethod
    def dataset_from_trax_tfds(dataset_name='mnist', variable_shapes=False,
                               batch_size_per_device=256, eval_batch_size=256, **kwargs):
        if len(kwargs) and any(True for v in kwargs.values() if v is not None):
            logger.warn('dataset_from_trax_tfds: ignoring arguments {}'.format(kwargs))
        streams = tf_inputs.data_streams(dataset_name)
        return inputs.batcher(streams, variable_shapes=variable_shapes,
                              batch_size_per_device=batch_size_per_device,
                              eval_batch_size=eval_batch_size)

    def train(self, epochs, n_eval_batches=10, batch_size_per_device=256,
              eval_batch_size=256, variable_shapes=False, *args, **kwargs):
        super(TraxTrainer, self).train(epochs=epochs, *args, **kwargs)
        assert self.data is not None
        assert self.model is not None

        task = training.TrainTask(
            itertools.cycle(self.data.train_stream(1)),
            tl.CrossEntropyLoss(),
            adafactor.Adafactor(.02))

        eval_task = training.EvalTask(
            itertools.cycle(self.data.eval_stream(1)),
            [tl.CrossEntropyLoss(), tl.Accuracy()],
            n_eval_batches=n_eval_batches)

        training_session = training.Loop(self.model, task, eval_task=eval_task,
                                         eval_at=lambda step_n: step_n % 50 == 0)

        training_session.run(n_steps=epochs)
        return training_session


__all__ = ['TraxTrainer']
