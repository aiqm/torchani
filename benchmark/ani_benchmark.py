from benchmark import Benchmark
import torchani


class NeighborBenchmark(Benchmark):

    def __init__(self, device):
        super(NeighborBenchmark, self).__init__(device)
        self.aev_computer = torchani.AEV(device=device)
        self.model = torchani.ModelOnAEV(
            self.aev_computer, benchmark=True, derivative=True, from_pync=None)

    def oneByOne(self, coordinates, species):
        conformations = coordinates.shape[0]
        coordinates = coordinates.to(self.device)
        for i in range(conformations):
            c = coordinates[i:i+1, :, :]
            self.model(c, species)
        ret = {'aev': self.model.timers['aev'], 'energy': self.model.timers['nn'],
               'force': self.model.timers['derivative']}
        self.model.reset_timers()
        return ret

    def inBatch(self, coordinates, species):
        coordinates = coordinates.to(self.device)
        self.model(coordinates, species)
        ret = {'aev': self.model.timers['aev'], 'energy': self.model.timers['nn'],
               'force': self.model.timers['derivative']}
        self.model.reset_timers()
        return ret


class FreeNeighborBenchmark(Benchmark):

    def __init__(self, device):
        super(FreeNeighborBenchmark, self).__init__(device)
        self.aev_computer = torchani.AEV(benchmark=True, device=device)
        self.model = torchani.ModelOnAEV(
            self.aev_computer, benchmark=True, derivative=True, from_pync=None)

    def oneByOne(self, coordinates, species):
        conformations = coordinates.shape[0]
        coordinates = coordinates.to(self.device)
        for i in range(conformations):
            c = coordinates[i:i+1, :, :]
            self.model(c, species)
        ret = {'aev': self.aev_computer.timers['aev'],
               'energy': self.model.timers['nn'], 'force': self.model.timers['derivative']}
        self.aev_computer.reset_timers()
        self.model.reset_timers()
        return ret

    def inBatch(self, coordinates, species):
        coordinates = coordinates.to(self.device)
        self.model(coordinates, species)
        ret = {'aev': self.aev_computer.timers['aev'],
               'energy': self.model.timers['nn'], 'force': self.model.timers['derivative']}
        self.aev_computer.reset_timers()
        self.model.reset_timers()
        return ret
