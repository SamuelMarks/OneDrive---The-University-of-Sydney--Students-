from os import path
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase, main as unittest_main

from trax import layers as tl
from trax.optimizers import adafactor

from ml_params_trax.example_model import get_model
from ml_params_trax.ml_params_impl import TraxTrainer


class TestMnist(TestCase):
    tfds_dir = None  # type: str or None
    model_dir = None  # type: str or None
    epochs = 3

    @classmethod
    def setUpClass(cls) -> None:
        TestMnist.tfds_dir = path.join(path.expanduser('~'), 'tensorflow_datasets')
        TestMnist.model_dir = mkdtemp('_model_dir')

    @classmethod
    def tearDownClass(cls) -> None:
        # rmtree(TestMnist.tfds_dir)
        rmtree(TestMnist.model_dir)

    def test_mnist(self) -> None:
        trainer = TraxTrainer()
        trainer.load_data('mnist', tfds_dir=TestMnist.tfds_dir)
        trainer.load_model(get_model, False, num_classes=10)
        training_session = trainer.train(epochs=self.epochs, model_dir=TestMnist.model_dir,
                                         metric_emit_freq=lambda step_n: step_n % 50 == 0,
                                         metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
                                         loss=tl.CrossEntropyLoss(),
                                         optimizer=adafactor.Adafactor(.02),
                                         callbacks=None,
                                         save_directory=None)
        self.assertEqual(training_session.current_step, self.epochs)


if __name__ == '__main__':
    unittest_main()
