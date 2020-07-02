from shutil import rmtree
from os import path
from tempfile import mkdtemp
from unittest import TestCase, main as unittest_main

from ml_params_trax.example_model import get_model
from ml_params_trax.ml_params_impl import TraxTrainer


class TestMnist(TestCase):
    tensorflow_datasets_dir = None
    model_dir = None

    @classmethod
    def setUpClass(cls) -> None:
        TestMnist.tensorflow_datasets_dir = path.join(path.expanduser('~'), 'tensorflow_datasets')
        TestMnist.model_dir = mkdtemp('_model_dir')

    @classmethod
    def tearDownClass(cls) -> None:
        # rmtree(TestMnist.tensorflow_datasets_dir)
        rmtree(TestMnist.model_dir)

    def test_mnist(self) -> None:
        trainer = TraxTrainer(get_model, num_classes=10)
        trainer.load_data('mnist', data_loader_kwargs={'tensorflow_datasets_dir': TestMnist.tensorflow_datasets_dir})

        epochs = 3
        training_session = trainer.train(epochs=epochs, model_dir=TestMnist.model_dir)
        self.assertEqual(training_session.current_step, epochs)


if __name__ == '__main__':
    unittest_main()
