from  data_loader import EEGDataLoader
from AGACN import AGACN
from trainer import Trainer
import numpy as np

if __name__ == '__main__':
    # /content/decoding-lilingual-EEG-signals/Processed_Data_
    dataset = EEGDataLoader("D:\Work\MachineLearning\Projects\DRAFT\Processed_Data_", "AK-SREP", "reading")

    model = AGACN(num_timepoints=2000, num_classes=9)
    model.count_parameters()
    trainer = Trainer(model, dataset, lr=0.0001, epochs=10, batch_size=64, num_folds=5)
    trainer.train()
