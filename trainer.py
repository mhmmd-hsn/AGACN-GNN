import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from visualization import Visualization
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd


class Trainer:
    def __init__(self, model, dataset, lr=0.0001, epochs=500, batch_size=80, num_folds=5, save_best=True, save_path="best_model.pth"):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_folds = num_folds
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        self.save_best = save_best
        self.save_path = save_path
        self.best_acc = 0  # Track best accuracy for model saving
        self.visualizer = Visualization()
        self.all_preds = []
        self.all_labels = []
        self.best_fold_metrics = None  # Store the best fold's metrics
        self.best_results = pd.DataFrame(columns=["fold Number", "Train Accuracy", "Valid Accuracy", "Epoch Number"])

    def train(self):
        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(skf.split(self.dataset.data, self.dataset.labels)):
            print(f"\nFold {fold + 1}/{self.num_folds}")
            
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)

            self.model = self.model.__class__(num_timepoints=self.model.agacn1.weight.shape[0], num_classes=self.model.fc.out_features)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0005)
            
            fold_train_losses, fold_train_accs = [], []
            fold_val_losses, fold_val_accs = [], []
            best_fold_epoch_acc = 0
            
            for epoch in range(self.epochs):
                train_loss = 0
                all_train_preds, all_train_labels = [], []
                self.model.train()
                
                for feature_matrix, adjacency_matrix, labels in train_loader:
                    self.optimizer.zero_grad()
                    output = self.model(feature_matrix, adjacency_matrix)
                    loss = self.criterion(output, labels)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    
                    pred = torch.argmax(output, dim=-1)
                    all_train_preds.extend(pred.cpu().numpy())
                    all_train_labels.extend(labels.cpu().numpy())
                    
                train_acc = accuracy_score(all_train_labels, all_train_preds)    
                val_loss, val_acc, preds, labels = self.validate(val_loader)

                fold_train_losses.append(train_loss / len(train_loader))
                fold_train_accs.append(train_acc)
                fold_val_losses.append(val_loss)
                fold_val_accs.append(val_acc)
                
                print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, '
                      f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                
                if self.save_best and val_acc > best_fold_epoch_acc:
                    best_fold_epoch_acc = val_acc
                    self.best_metrics = [fold+1, train_acc, val_acc, epoch]
                
                if self.save_best and val_acc > self.best_acc:
                    self.best_acc = val_acc
                    self.best_fold_metrics = (fold_train_losses, fold_train_accs, fold_val_losses, fold_val_accs)

            if self.best_metrics:
                self.best_results = self.add_row(self.best_metrics)
        
        column_means = self.best_results.drop(columns=["fold Number"]).mean()
        average_row = pd.DataFrame([["Average"] + column_means.tolist()], columns=self.best_results.columns)
        self.best_results = pd.concat([self.best_results, average_row], ignore_index=True)
        print(self.best_results)

        if self.best_fold_metrics:
            self.visualizer.plot_loss(self.best_fold_metrics[0], self.best_fold_metrics[2])
            self.visualizer.plot_accuracy(self.best_fold_metrics[1], self.best_fold_metrics[3])

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for feature_matrix, adjacency_matrix, labels in val_loader:
                output = self.model(feature_matrix, adjacency_matrix)  # Keep raw logits
                loss = self.criterion(output, labels)
                val_loss += loss.item()
                
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        self.all_preds.extend(all_preds)
        self.all_labels.extend(all_labels)
        
        return val_loss / len(val_loader), accuracy_score(all_labels, all_preds), all_preds, all_labels

    def add_row(self, data):
        new_row = pd.DataFrame([data], columns=self.best_results.columns)  # Create a new DataFrame row
        df = pd.concat([self.best_results, new_row], ignore_index=True)  # Append to the DataFrame
        return df
