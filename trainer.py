import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset



class Trainer:
    def __init__(self, model, dataset, lr=0.0001, epochs=500, batch_size=80, num_folds=5):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_folds = num_folds
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)

    def train(self):
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        all_train_losses, all_train_accs = [], []
        all_val_losses, all_val_accs = [], []

        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(self.dataset)))):

            print(f"\nFold {fold + 1}/{self.num_folds}")

            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)

            fold_train_losses, fold_train_accs = [], []
            fold_val_losses, fold_val_accs = [], []

            for epoch in range(self.epochs):
                train_loss = 0
                correct, total = 0, 0

                for feature_matrix, adjacency_matrix, labels in train_loader:
                    self.optimizer.zero_grad()
                    output = self.model.forward(feature_matrix, adjacency_matrix)
                    # print(output.shape)
                    # print(labels.shape) 
                    loss = self.criterion(output, labels)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    
                    pred = output.argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total += labels.size(0)

                train_acc = correct / total
                val_loss, val_acc = self.validate(val_loader)

                fold_train_losses.append(train_loss / len(train_loader))
                fold_train_accs.append(train_acc)
                fold_val_losses.append(val_loss)
                fold_val_accs.append(val_acc)

                print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, '
                      f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            # Store mean metrics for this fold
            all_train_losses.append(sum(fold_train_losses) / len(fold_train_losses))
            all_train_accs.append(sum(fold_train_accs) / len(fold_train_accs))
            all_val_losses.append(sum(fold_val_losses) / len(fold_val_losses))
            all_val_accs.append(sum(fold_val_accs) / len(fold_val_accs))

        # Compute overall mean across all folds
        mean_train_loss = sum(all_train_losses) / self.num_folds
        mean_train_acc = sum(all_train_accs) / self.num_folds
        mean_val_loss = sum(all_val_losses) / self.num_folds
        mean_val_acc = sum(all_val_accs) / self.num_folds

        print("\n===== Overall Training Results =====")
        print(f"Mean Train Loss: {mean_train_loss:.4f}")
        print(f"Mean Train Accuracy: {mean_train_acc:.4f}")
        print(f"Mean Validation Loss: {mean_val_loss:.4f}")
        print(f"Mean Validation Accuracy: {mean_val_acc:.4f}")

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        correct, total = 0, 0

        with torch.no_grad():
            for feature_matrix, adjacency_matrix, labels in val_loader:
                output = self.model(feature_matrix, adjacency_matrix)
                loss = self.criterion(output, labels)
                val_loss += loss.item()
                
                pred = output.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        return val_loss / len(val_loader), correct / total

