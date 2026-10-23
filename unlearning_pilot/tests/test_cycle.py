import random
import unittest

from unlearning_pilot.train import ShuffledCycle


class ShuffledCycleTest(unittest.TestCase):
    def test_uses_each_row_before_repeating(self):
        cycle = ShuffledCycle([{"id": i} for i in range(8)], random.Random(42))
        first = [cycle.next()["id"] for _ in range(8)]
        second = [cycle.next()["id"] for _ in range(8)]
        self.assertEqual(set(first), set(range(8)))
        self.assertEqual(set(second), set(range(8)))
        self.assertEqual(cycle.visits, 16)


if __name__ == "__main__":
    unittest.main()
