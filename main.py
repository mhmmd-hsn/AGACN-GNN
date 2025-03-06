from  data_loader import EEGDataLoader
from connections import GMatrixCalculator
from visualization import Visualization
from AGACN import AGACN
# from trainer import ModelTrainer
from torch.utils.data import Dataset, DataLoader
from trainer import Trainer
import torch

if __name__ == '__main__':

    dataset = EEGDataLoader("Processed_Data", "AK-SREP", "reading")

    # vis = Visualization()
    # vis.plot_dataset_trials

    model = AGACN(num_timepoints=2000, num_classes=9)
    # model.count_parameters()
    trainer = Trainer(model, dataset, lr=0.0001, epochs=10, batch_size=80, num_folds=5)

    trainer.train()
    # torch.save(model, 'model_full.pth')
    # trainer.validate()

