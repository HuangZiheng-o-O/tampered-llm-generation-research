import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is optional for local preflight")
class LossTest(unittest.TestCase):
    def test_graddiff_pushes_forget_up_and_retain_down(self):
        import torch
        from unlearning_pilot.losses import graddiff

        forget = torch.tensor(2.0, requires_grad=True)
        retain = torch.tensor(1.0, requires_grad=True)
        graddiff(forget, retain, alpha=2.0).backward()
        self.assertEqual(float(forget.grad), -1.0)
        self.assertEqual(float(retain.grad), 2.0)

    def test_npo_prefers_higher_forget_nll(self):
        import torch
        from unlearning_pilot.losses import npo

        ref = torch.tensor([5.0])
        retain = torch.tensor(1.0)
        low = npo(torch.tensor([4.0]), ref, retain, beta=0.1, alpha=1)
        high = npo(torch.tensor([6.0]), ref, retain, beta=0.1, alpha=1)
        self.assertLess(float(high), float(low))

    def test_rmu_uses_only_answer_tokens(self):
        import torch
        from unlearning_pilot.losses import masked_activation_mse

        current = torch.tensor([[[100., 100.], [2., 2.]]])
        target = torch.zeros_like(current)
        mask = torch.tensor([[False, True]])
        self.assertEqual(float(masked_activation_mse(current, target, mask)), 4.0)


if __name__ == "__main__":
    unittest.main()
