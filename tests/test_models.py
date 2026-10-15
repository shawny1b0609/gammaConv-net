import unittest

import torch

from gammaconv.models import MODELS, make_model
from gammaconv.loss import FlowLoss
from physics_informed.simplified_physics_model import SimplifiedGammaKernel
from physics_informed.dual_kernel_model import DualGammaKernel
from physics_informed.improved_water_balanced_kernel import ImprovedWaterBalancedKernel
from physics_informed.hydrological_model import HydrologicalPhysicsModel
from physics_informed.peak_loss_functions import get_peak_loss_function


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_kernels_are_nonnegative_and_normalized(self):
        single, dual, triple = (
            SimplifiedGammaKernel(),
            DualGammaKernel(),
            ImprovedWaterBalancedKernel(),
        )
        kernels = [single.get_gamma_weights()] + [
            dual.get_gamma_weights(k) for k in ("fast", "slow")
        ]
        kernels += [triple.get_gamma_weights(k) for k in range(3)]
        for weights in kernels:
            self.assertTrue(torch.isfinite(weights).all())
            self.assertTrue((weights >= 0).all())
            self.assertAlmostEqual(weights.sum().item(), 1.0, places=6)

    def test_single_kernel_impulse_response_and_causality(self):
        kernel = SimplifiedGammaKernel(max_lag=8)
        rain = torch.zeros(1, 16, 1)
        rain[0, 4, 0] = 1
        response = kernel(rain)
        torch.testing.assert_close(response[0, :4, 0], torch.zeros(4))
        torch.testing.assert_close(response[0, 4:12, 0], kernel.get_gamma_weights())
        changed = rain.clone()
        changed[:, 12:, :] = 10
        torch.testing.assert_close(kernel(changed)[:, :12], response[:, :12])

    def test_allocation_is_a_probability_vector(self):
        rain = torch.cat(
            [torch.zeros(1, 30, 1), torch.ones(1, 30, 1), torch.rand(4, 30, 1)]
        )
        for kernel in (DualGammaKernel(), ImprovedWaterBalancedKernel()):
            weights = kernel.get_allocation_weights(rain)
            self.assertTrue((weights > 0).all())
            torch.testing.assert_close(weights.sum(1), torch.ones(6))
            self.assertFalse(torch.allclose(weights[0], weights[1]))

    def test_all_three_models_have_finite_forward_and_backward_passes(self):
        for name in MODELS:
            with self.subTest(model=name):
                torch.manual_seed(12)
                model = make_model(name, hidden=16, seq=12, lag=8)
                rain, flow = torch.rand(4, 12, 1), torch.rand(4, 12, 1)
                out = model(rain, flow)
                self.assertEqual(tuple(out.shape), (4, 1))
                self.assertTrue((out >= 0).all())
                out.square().mean().backward()
                for param in model.parameters():
                    self.assertIsNotNone(param.grad)
                    self.assertTrue(torch.isfinite(param.grad).all())

    def test_retained_model_and_losses_run(self):
        torch.manual_seed(4)
        model = HydrologicalPhysicsModel(
            rainfall_seq_len=8, flow_seq_len=8, hidden_dim=8, max_lag=8
        )
        pred = model(torch.rand(3, 8, 1), torch.rand(3, 8, 1))
        self.assertEqual(tuple(pred.shape), (3, 1))
        self.assertTrue(torch.isfinite(pred).all())
        for kind in ["peak_weighted", "adaptive", "sample_aware", "comprehensive"]:
            kwargs = (
                {"sample_size": 100}
                if kind in ("sample_aware", "comprehensive")
                else {}
            )
            values = torch.tensor([[0.1], [0.3], [0.8]], requires_grad=True)
            loss = get_peak_loss_function(kind, **kwargs)(
                values, torch.tensor([[0.2], [0.4], [0.7]])
            )
            loss.backward()
            self.assertTrue(torch.isfinite(values.grad).all())
        self.assertTrue(torch.isfinite(FlowLoss()(torch.ones(2, 1), torch.ones(2, 1))))


if __name__ == "__main__":
    unittest.main()
