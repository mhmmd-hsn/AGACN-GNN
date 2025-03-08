def train(self):
    kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(self.dataset)))):
        print(f"\nFold {fold + 1}/{self.num_folds}")
        
        train_subset = Subset(self.dataset, train_idx)
        val_subset = Subset(self.dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)

        # Reinitialize model for each fold
        self.model = self.model.__class__()  # Create a fresh instance
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Move to GPU if available

        # Reinitialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0005)

        fold_train_losses, fold_train_accs = [], []
        fold_val_losses, fold_val_accs = [], []
        best_fold_epoch_acc = 0
        
        for epoch in range(self.epochs):
            train_loss, correct, total = 0, 0, 0
            self.model.train()
            
            for feature_matrix, adjacency_matrix, labels in train_loader:
                feature_matrix, adjacency_matrix, labels = (
                    feature_matrix.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                    adjacency_matrix.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                    labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                )
                self.optimizer.zero_grad()
                output = self.model(feature_matrix, adjacency_matrix)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
                pred = output.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                
            train_acc = correct / total
            val_loss, val_acc, preds, labels, probs = self.validate(val_loader)
            
            fold_train_losses.append(train_loss / len(train_loader))
            fold_train_accs.append(train_acc)
            fold_val_losses.append(val_loss)
            fold_val_accs.append(val_acc)
            
            print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            if self.save_best and val_acc > best_fold_epoch_acc:
                best_fold_epoch_acc = val_acc
                self.best_metrics = [fold+1, train_acc, val_acc, epoch]
                torch.save(self.model.state_dict(), f'best_model_fold{fold+1}.pth')
            
            if self.save_best and val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), self.save_path)
                self.best_fold_metrics = (fold_train_losses, fold_train_accs, fold_val_losses, fold_val_accs)

        if self.best_metrics:
            self.best_results = self.add_row(self.best_metrics)

    column_means = self.best_results.drop(columns=["fold Number"]).mean()
    average_row = pd.DataFrame([["Average"] + column_means.tolist()], columns=self.best_results.columns)
    self.best_results = pd.concat([self.best_results, average_row], ignore_index=True)
    print(self.best_results)
