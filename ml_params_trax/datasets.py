from os import path

from ml_params.datasets import load_data_from_ml_prepare
from ml_params.utils import common_dataset_handler
from ml_prepare.datasets import datasets2classes
from trax.supervised import inputs
from trax.supervised import tf_inputs

from ml_params_trax import get_logger

logger = get_logger('.'.join((path.basename(path.dirname(__file__)),
                              path.basename(__file__).rpartition('.')[0])))


def load_data_from_trax_tfds_or_ml_prepare(dataset_name, tfds_dir=None,
                                           K=None, as_numpy=False, **data_loader_kwargs):
    """
    Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

    :param dataset_name: name of dataset
    :type dataset_name: ```str```

    :param tfds_dir: directory to look for models in. Default is ~/tensorflow_datasets.
    :type tfds_dir: ```None or str```

    :param K: backend engine, e.g., `np` or `tf`
    :type K: ```None or np or tf or Any```

    :param as_numpy: Convert to numpy ndarrays
    :type as_numpy: ```bool```

    :param data_loader_kwargs: pass this as arguments to data_loader function
    :type data_loader_kwargs: ```**data_loader_kwargs```

    :return: Train and tests dataset splits
    :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
    """
    if dataset_name in datasets2classes:
        return load_data_from_ml_prepare(dataset_name=dataset_name,
                                         tfds_dir=tfds_dir,
                                         **data_loader_kwargs)

    ds_builder = dataset_from_trax_tfds(
        dataset_name=dataset_name, tfds_dir=tfds_dir, **data_loader_kwargs
    )

    if hasattr(ds_builder, 'download_and_prepare_kwargs'):
        download_and_prepare_kwargs = getattr(ds_builder, 'download_and_prepare_kwargs')
        delattr(ds_builder, 'download_and_prepare_kwargs')
    else:
        download_and_prepare_kwargs = None

    return common_dataset_handler(
        ds_builder=ds_builder,
        download_and_prepare_kwargs=download_and_prepare_kwargs,
        scale=data_loader_kwargs.get('scale'),
        K=data_loader_kwargs.get('K', K),
        as_numpy=data_loader_kwargs.get('as_numpy', as_numpy)
    )


def dataset_from_trax_tfds(dataset_name='mnist', variable_shapes=False, tfds_dir=None,
                           batch_size_per_device=256, eval_batch_size=256, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if len(kwargs):
        logger.warn('dataset_from_trax_tfds: ignoring arguments {}'.format(kwargs))
    streams = tf_inputs.data_streams(dataset_name, data_dir=tfds_dir)
    return inputs.batcher(streams, variable_shapes=variable_shapes,
                          batch_size_per_device=batch_size_per_device,
                          eval_batch_size=eval_batch_size)
