import numpy as np

from pnp_mace import utils


class TestSet:

    def setup_method(self, method):
        np.random.seed(12345)


    def test_stack_init_image(self):
        x = np.random.randn(8, 8)
        y = utils.stack_init_image(x, 2)
        assert np.linalg.norm(x - y[0]) == 0
        assert np.linalg.norm(x - y[1]) == 0
