import unittest

import torch

from daics.config import DATA, WINDOW
from daics.gru_config import GRU_CONFIG
from daics.models.gru import GRUPredictor, parameter_count
from daics.training_gru import sectionwise_mse_sum


class GRUPredictorTests(unittest.TestCase):
    def test_output_shapes_and_parameter_count(self):
        model = GRUPredictor()
        outputs = model(torch.randn(3, DATA.n_features, WINDOW.Win))
        self.assertEqual([tuple(x.shape) for x in outputs], [(3, 5, 4), (3, 7, 4), (3, 12, 4)])
        self.assertEqual(parameter_count(model), 27360)
        self.assertEqual(GRU_CONFIG.hidden_size, 64)

    def test_gradients_and_optimizer_update(self):
        model = GRUPredictor()
        optimizer = torch.optim.Adam(model.parameters(), lr=GRU_CONFIG.learning_rate)
        before = [p.detach().clone() for p in model.parameters()]
        x = torch.randn(2, DATA.n_features, WINDOW.Win)
        targets = [torch.randn(2, n, WINDOW.Wout) for n in DATA.mgse_per_section]
        loss = sectionwise_mse_sum(model(x), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        optimizer.step()
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))


if __name__ == "__main__":
    unittest.main()
