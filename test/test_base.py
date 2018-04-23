import unittest
import numpy as np
try:
    from binningpy.base import BinningBase
except:
    import sys
    from pathlib import Path
    path = str(
        Path(__file__).absolute().parent.parent
    )
    if path not in sys.path:
        sys.path.append(path)
    from binningpy.base import BinningBase


def setUpModule():
    print("setUpModule")


def tearDownModule():
    print("tearUpModule")


class TestBinningBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target_x = np.array([1, 1, 3, 3, 2, 1, 3, 5, 6, 7, 7, 2]).reshape(-1, 1)
        cls.target_result = np.array([[0], [0], [1], [1], [1], [0], [1], [2], [2], [3], [3], [1]])
        cls.bins = [[0, 2, 4, 6, 8]]
        print("setUpClass")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass")

    def setUp(self):
        print("instance setUp")

    def tearDown(self):
        print("instance tearDown")

    def test_BinningBase_init(self):
        with self.assertRaisesRegex(AttributeError, r"bin number must be int") as a:
            bb = BinningBase(bin_nbr=1.2)

        with self.assertRaisesRegex(AttributeError, r"bin number must > 0 when confined is True") as a:
            bb = BinningBase(bin_nbr=-1)

        with self.assertRaisesRegex(AttributeError, r"bin number must > 2 when confined is False") as a:
            bb = BinningBase(bin_nbr=-1, confined=False)
        bb = BinningBase(bin_nbr=3)
        self.assertIsInstance(bb, BinningBase)
        self.assertEqual(bb.bin_nbr, 3)
        self.assertEqual(bb.confined, True)
        self.assertEqual(bb.copy, True)

    def test_confined_transform(self):
        bb = BinningBase(4)
        bb._bins = self.bins
        result = bb.transform(self.target_x)
        print(result)
        #self.assertSequenceEqual(result, self.target_result)
        assert all(lambda x: x[0] == x[1], zip(result, self.target_result))


def BinningBase_suite():
    suite = unittest.TestSuite()
    suite.addTest(TestBinningBase("test_BinningBase_init"))
    suite.addTest(TestBinningBase("test_confined_transform"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = BinningBase_suite()
    runner.run(test_suite)
