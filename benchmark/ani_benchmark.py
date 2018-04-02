from benchmark import Benchmark
import torchani


class ANIBenchmark(Benchmark):

    def __init__(self, device):
        super(ANIBenchmark, self).__init__(device)
        self.aev_computer = torchani.SortedAEV(device=device)
        self.model = torchani.ModelOnAEV(
            self.aev_computer, benchmark=True, derivative=True, from_nc=None)

    def oneByOne(self, coordinates, species):
        conformations = coordinates.shape[0]
        coordinates = coordinates.to(self.device)
        for i in range(conformations):
            c = coordinates[i:i+1, :, :]
            self.model(c, species)
        ret = {
            'aev': self.model.timers['aev'],
            'energy': self.model.timers['nn'],
            'force': self.model.timers['derivative']
        }
        self.model.reset_timers()
        return ret

    def inBatch(self, coordinates, species):
        coordinates = coordinates.to(self.device)
        self.model(coordinates, species)
        ret = {
            'aev': self.model.timers['aev'],
            'energy': self.model.timers['nn'],
            'force': self.model.timers['derivative']
        }
        self.model.reset_timers()
        return ret
