""" Implementation of ml_params API """

# Trax specifics mostly taken from https://github.com/google/trax/blob/dcf806d/trax/supervised/mnist_test.py

import itertools
from os import path
from sys import stdout
from typing import Tuple

import tensorflow as tf
import trax
from ml_params.base import BaseTrainer
from ml_prepare.exectors import build_tfds_dataset
from trax import layers as tl
from trax.supervised import training

from ml_params_trax import get_logger
from ml_params_trax.datasets import load_data_from_trax_tfds_or_ml_prepare

logger = get_logger('.'.join((path.basename(path.dirname(__file__)),
                              path.basename(__file__).rpartition('.')[0])))


class TraxTrainer(BaseTrainer):
    """ Implementation of ml_params BaseTrainer for Trax """

    data = None  # type: (None or Tuple[tf.data.Dataset, tf.data.Dataset] or trax.supervised.Inputs)
    model = None  # type: tl.Serial

    def load_data(self, dataset_name, data_loader=load_data_from_trax_tfds_or_ml_prepare,
                  data_type='infer', output_type=None, K=None,
                  **data_loader_kwargs):
        """
        Load the data for your ML pipeline. Will be fed into `train`.

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param data_loader: function that returns the expected data type.
         Defaults to Trax TensorFlow Datasets and ml_prepare combined one.
        :type data_loader: ```None or (*args, **kwargs) -> tf.data.Datasets or Any```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```**data_loader_kwargs```

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
                                                       data_loader=data_loader,
                                                       data_type=data_type,
                                                       output_type=output_type,
                                                       K=K,
                                                       **data_loader_kwargs)
        # self.data = trax.supervised.Inputs(*self.data)
        # trax.supervised.inputs.dataset_to_stream(self.data, dataset_name)

    def train(self, callbacks, epochs, loss, metrics, metric_emit_freq, optimizer,
              save_directory, output_type='infer', writer=stdout,
              n_eval_batches=10, batch_size_per_device=256,
              eval_batch_size=256, variable_shapes=False,
              *args, **kwargs):
        """
        Run the training loop for your ML pipeline.

        :param callbacks: Collection of callables that are run inside the training loop
        :type callbacks: ```None or List[Callable] or Tuple[Callable]```

        :param epochs: number of epochs (must be greater than 0)
        :type epochs: ```int```

        :param loss: Loss function, can be a string (depending on the framework) or an instance of a class
        :type loss: ```str or Callable or Any```

        :param metrics: Collection of metrics to monitor, e.g., accuracy, f1
        :type metrics: ```None or List[Callable or str] or Tuple[Callable or str]```

        :param metric_emit_freq: Frequency of metric emission, e.g., `lambda: epochs % 10 == 0`, defaults to every epoch
        :type metric_emit_freq: ```None or (*args, **kwargs) -> bool```

        :param optimizer: Optimizer, can be a string (depending on the framework) or an instance of a class
        :type callbacks: ```str or Callable or Any```

        :param save_directory: Directory to save output in, e.g., weights in h5 files. If None, don't save.
        :type save_directory: ```None or str```

        :param output_type: `if save_directory is not None` then save in this format, e.g., 'h5'.
        :type output_type: ```str```

        :param writer: Writer for all output, could be a TensorBoard instance, a file handler like stdout or stderr
        :type writer: ```stdout or Any```

        :param n_eval_batches:
        :type n_eval_batches: ```int```

        :param batch_size_per_device:
        :type batch_size_per_device: ```int```

        :param eval_batch_size:
        :type eval_batch_size: ```int```

        :param variable_shapes:
        :type variable_shapes: ```bool```

        :param args:
        :param kwargs:
        :return:
        """
        super(TraxTrainer, self).train(callbacks=callbacks,
                                       epochs=epochs,
                                       loss=loss,
                                       metrics=metrics,
                                       metric_emit_freq=metric_emit_freq,
                                       optimizer=optimizer,
                                       save_directory=save_directory,
                                       output_type='infer',
                                       writer=writer,
                                       *args, **kwargs)
        assert self.data is not None
        assert self.model is not None

        task = training.TrainTask(
            itertools.cycle(self.data.train_stream(1)),
            loss,
            optimizer
        )

        eval_task = training.EvalTask(
            itertools.cycle(self.data.eval_stream(1)),
            metrics,
            n_eval_batches=n_eval_batches
        )

        training_session = training.Loop(self.model, task, eval_task=eval_task,
                                         eval_at=metric_emit_freq)

        training_session.run(n_steps=epochs)
        return training_session


del path, Tuple, tf, trax, build_tfds_dataset, get_logger

__all__ = ['TraxTrainer']
