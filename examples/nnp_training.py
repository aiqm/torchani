import sys
import torch
import ignite
import torchani
import model

chunk_size = 256
batch_chunks = 4
dataset_path = sys.argv[1]
dataset_checkpoint = 'dataset-checkpoint.dat'
model_checkpoint = 'checkpoint.pt'

shift_energy = torchani.EnergyShifter()
training, validation, testing = torchani.data.load_or_create(
    dataset_checkpoint, dataset_path, chunk_size,
    transform=[shift_energy.dataset_subtract_sae])
training = torchani.data.dataloader(training, batch_chunks)
validation = torchani.data.dataloader(validation, batch_chunks)

nnp = model.get_or_create_model(model_checkpoint)


class Flatten(torch.nn.Module):

    def __init__(self, model):
        super(Flatten, self).__init__()
        self.model = model

    def forward(self, *input):
        return self.model(*input).flatten()


batch_nnp = torchani.models.BatchModel(Flatten(nnp))
container = torchani.ignite.Container({'energies': batch_nnp})
optimizer = torch.optim.Adam(nnp.parameters())
trainer = ignite.engine.create_supervised_trainer(
    container, optimizer, torchani.ignite.energy_mse_loss)
evaluator = ignite.engine.create_supervised_evaluator(container, metrics={
        'RMSE': torchani.ignite.energy_rmse_metric
    })


@trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
def log_training_loss(trainer):
    print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch,
                                          trainer.state.output))


@trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(training)
    metrics = evaluator.state.metrics
    print("Training Results - Epoch: {}  RMSE: {:.2f}"
          .format(trainer.state.epoch, metrics['RMSE']))


@trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(validation)
    metrics = evaluator.state.metrics
    print("Validation Results - Epoch: {}  RMSE: {:.2f}"
          .format(trainer.state.epoch, metrics['RMSE']))


trainer.run(training, max_epochs=10)
