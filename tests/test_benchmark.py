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

    def _testModule(self, run_module, result_module, asserts):
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
        self.assertEqual(set(result_module.timers.keys()), set(keys))
        for i in keys:
            self.assertEqual(result_module.timers[i], 0)
        old_timers = copy.copy(result_module.timers)
        for _ in range(self.count):
            run_module((self.species, self.coordinates))
            for i in keys:
                self.assertLess(old_timers[i], result_module.timers[i])
            for i in asserts:
                if '>=' in i:
                    i = i.split('>=')
                    key0 = i[0].strip()
                    key1 = i[1].strip()
                    self.assertGreaterEqual(
                        result_module.timers[key0], result_module.timers[key1])
                elif '<=' in i:
                    i = i.split('<=')
                    key0 = i[0].strip()
                    key1 = i[1].strip()
                    self.assertLessEqual(
                        result_module.timers[key0], result_module.timers[key1])
                elif '>' in i:
                    i = i.split('>')
                    key0 = i[0].strip()
                    key1 = i[1].strip()
                    self.assertGreater(
                        result_module.timers[key0], result_module.timers[key1])
                elif '<' in i:
                    i = i.split('<')
                    key0 = i[0].strip()
                    key1 = i[1].strip()
                    self.assertLess(result_module.timers[key0],
                                    result_module.timers[key1])
                elif '=' in i:
                    i = i.split('=')
                    key0 = i[0].strip()
                    key1 = i[1].strip()
                    self.assertEqual(result_module.timers[key0],
                                     result_module.timers[key1])
            old_timers = copy.copy(result_module.timers)
        result_module.reset_timers()
        self.assertEqual(set(result_module.timers.keys()), set(keys))
        for i in keys:
            self.assertEqual(result_module.timers[i], 0)

    def testAEV(self):
        aev_computer = torchani.SortedAEV(
            benchmark=True, dtype=self.dtype, device=self.device)
        prepare = torchani.PrepareInput(aev_computer.species, self.device)
        run_module = torch.nn.Sequential(prepare, aev_computer)
        self._testModule(run_module, aev_computer, [
                         'terms and indices>radial terms',
                         'terms and indices>angular terms',
                         'total>terms and indices',
                         'total>combinations', 'total>assemble',
                         'total>mask_r', 'total>mask_a'
                         ])

    def testANIModel(self):
        aev_computer = torchani.SortedAEV(
            dtype=self.dtype, device=self.device)
        prepare = torchani.PrepareInput(aev_computer.species, self.device)
        model = torchani.models.NeuroChemNNP(
            aev_computer.species, benchmark=True).to(self.device)
        run_module = torch.nn.Sequential(prepare, aev_computer, model)
        self._testModule(run_module, model, ['forward'])


if __name__ == '__main__':
    unittest.main()
