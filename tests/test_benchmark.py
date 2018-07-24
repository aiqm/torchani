import torch
import torchani
import unittest
import copy


class TestBenchmark(unittest.TestCase):

    def setUp(self, dtype=torchani.default_dtype,
              device=torchani.default_device):
        self.dtype = dtype
        self.device = device
        self.conformations = 100
        self.species = list('HHCCNNOO')
        self.coordinates = torch.randn(
            self.conformations, 8, 3, dtype=dtype, device=device)
        self.count = 100

    def _testModule(self, module, asserts):
        keys = []
        for i in asserts:
            if '>=' in i:
                i = i.split('>=')
                keys += [i[0].strip(), i[1].strip()]
            elif '<=' in i:
                i = i.split('<=')
                keys += [i[0].strip(), i[1].strip()]
            elif '>' in i:
                i = i.split('>')
                keys += [i[0].strip(), i[1].strip()]
            elif '<' in i:
                i = i.split('<')
                keys += [i[0].strip(), i[1].strip()]
            elif '=' in i:
                i = i.split('=')
                keys += [i[0].strip(), i[1].strip()]
            else:
                keys.append(i.strip())
        self.assertEqual(set(module.timers.keys()), set(keys))
        for i in keys:
            self.assertEqual(module.timers[i], 0)
        old_timers = copy.copy(module.timers)
        for _ in range(self.count):
            module(self.coordinates, self.species)
            for i in keys:
                self.assertLess(old_timers[i], module.timers[i])
            for i in asserts:
                if '>=' in i:
                    i = i.split('>=')
                    key0 = i[0].strip()
                    key1 = i[1].strip()
                    self.assertGreaterEqual(
                        module.timers[key0], module.timers[key1])
                elif '<=' in i:
                    i = i.split('<=')
                    key0 = i[0].strip()
                    key1 = i[1].strip()
                    self.assertLessEqual(
                        module.timers[key0], module.timers[key1])
                elif '>' in i:
                    i = i.split('>')
                    key0 = i[0].strip()
                    key1 = i[1].strip()
                    self.assertGreater(
                        module.timers[key0], module.timers[key1])
                elif '<' in i:
                    i = i.split('<')
                    key0 = i[0].strip()
                    key1 = i[1].strip()
                    self.assertLess(module.timers[key0], module.timers[key1])
                elif '=' in i:
                    i = i.split('=')
                    key0 = i[0].strip()
                    key1 = i[1].strip()
                    self.assertEqual(module.timers[key0], module.timers[key1])
            old_timers = copy.copy(module.timers)
        module.reset_timers()
        self.assertEqual(set(module.timers.keys()), set(keys))
        for i in keys:
            self.assertEqual(module.timers[i], 0)

    def testAEV(self):
        aev_computer = torchani.SortedAEV(
            benchmark=True, dtype=self.dtype, device=self.device)
        self._testModule(aev_computer, [
                         'terms and indices>radial terms',
                         'terms and indices>angular terms',
                         'total>terms and indices',
                         'total>combinations', 'total>assemble',
                         'total>mask_r', 'total>mask_a'
                         ])

    def testModelOnAEV(self):
        aev_computer = torchani.SortedAEV(
            dtype=self.dtype, device=self.device)
        model = torchani.models.NeuroChemNNP(
            aev_computer, benchmark=True)
        self._testModule(model, ['forward>aev', 'forward>nn'])
        model = torchani.models.NeuroChemNNP(
            aev_computer, benchmark=True, derivative=True)
        self._testModule(
            model, ['forward>aev', 'forward>nn', 'forward>derivative'])


if __name__ == '__main__':
    unittest.main()
