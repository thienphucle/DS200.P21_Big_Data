import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score

from dataset import TikTokDataset
from classic_models import ModernBaselineModels
from deep_models import TikTokGrowthPredictor
from visualizer import Visualizer

class Trainer:    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.baseline_models = ModernBaselineModels()
        self.training_history = {
            'train_losses': {'regression': [], 'classification': []},
            'val_losses': {'regression': [], 'classification': []}
        }
        
    def prepare_data(self, training_data, user_trend_features, text_features, test_size=0.2, batch_size=16):
        stratify_col = training_data['engagement_growth_class'] if 'engagement_growth_class' in training_data.columns else None
        indices = np.arange(len(training_data))
        train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=42, stratify=stratify_col)
        
        train_df = training_data.iloc[train_idx].reset_index(drop=True)
        val_df = training_data.iloc[val_idx].reset_index(drop=True)
        train_text = text_features[train_idx]
        val_text = text_features[val_idx]

        train_ds = TikTokDataset(train_df, user_trend_features, train_text, mode='train')
        val_ds = TikTokDataset(val_df, user_trend_features, val_text, mode='train')
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        X_train, feature_names = self.baseline_models.prepare_features(train_df, train_text)
        X_val, _ = self.baseline_models.prepare_features(val_df, val_text)

        y_train_reg = train_df.filter(like='target_').values
        y_val_reg = val_df.filter(like='target_').values

        growth_map = {'increase': 2, 'stable': 1, 'decrease': 0}
        y_train_cls = train_df.filter(like='_class').replace(growth_map).values.astype(int)
        y_val_cls = val_df.filter(like='_class').replace(growth_map).values.astype(int)

        return train_loader, val_loader, X_train, X_val, y_train_reg, y_val_reg, y_train_cls, y_val_cls, feature_names
    
    def train(self, train_loader, val_loader, X_train, X_val, y_train_reg, y_val_reg, y_train_cls, y_val_cls,
              feature_names, epochs=15, lr=2e-5, save_path='best_tiktok_model.pth'):

        self.model = TikTokGrowthPredictor(text_dim=train_loader.dataset.text_features.shape[1])
        self.model.to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        reg_loss_fn = nn.SmoothL1Loss()
        cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader)
        )
        
        best_val_loss = float('inf')
        patience = 5
        counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_reg, total_cls = 0, 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                text, struct, time = batch['text_features'].to(self.device), batch['structured_features'].to(self.device), batch['time_features'].to(self.device)
                y_reg, y_cls = batch['targets'].to(self.device), batch['class_targets'].to(self.device)

                out_reg, out_cls, _ = self.model(text, struct, time)
                loss_reg = reg_loss_fn(out_reg, y_reg)
                loss_cls = sum(cls_loss_fn(out_cls[:, i], y_cls[:, i]) for i in range(out_cls.shape[1])) / out_cls.shape[1]
                loss = loss_reg + 0.3 * loss_cls
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_reg += loss_reg.item()
                total_cls += loss_cls.item()
            
            self.model.eval()
            val_reg, val_cls = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    text, struct, time = batch['text_features'].to(self.device), batch['structured_features'].to(self.device), batch['time_features'].to(self.device)
                    y_reg, y_cls = batch['targets'].to(self.device), batch['class_targets'].to(self.device)
                    out_reg, out_cls, _ = self.model(text, struct, time)
                    loss_reg = reg_loss_fn(out_reg, y_reg)
                    loss_cls = sum(cls_loss_fn(out_cls[:, i], y_cls[:, i]) for i in range(out_cls.shape[1])) / out_cls.shape[1]
                    val_reg += loss_reg.item()
                    val_cls += loss_cls.item()

            avg_val_loss = val_reg + 0.3 * val_cls
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'model_config': {'text_dim': train_loader.dataset.text_features.shape[1]}
                }, save_path)
            else:
                counter += 1
                if counter >= patience:
                    break
        
        print("Finished training neural model.")
        baseline_results = self.baseline_models.train_baseline_models(X_train, y_train_reg, y_train_cls, X_val, y_val_reg, y_val_cls, feature_names)
        return baseline_results
    
    def evaluate_and_visualize(self, val_loader, y_val_reg, y_val_cls, baseline_results, model_path='best_tiktok_model.pth'):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = TikTokGrowthPredictor(text_dim=checkpoint['model_config']['text_dim'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        reg_preds, cls_preds = [], []
        with torch.no_grad():
            for batch in val_loader:
                text, struct, time = batch['text_features'].to(self.device), batch['structured_features'].to(self.device), batch['time_features'].to(self.device)
                reg_out, cls_out, _ = self.model(text, struct, time)
                reg_preds.append(reg_out.cpu().numpy())
                cls_preds.append(torch.softmax(cls_out, dim=-1).cpu().numpy())

        reg_preds = np.vstack(reg_preds)
        cls_preds = np.vstack(cls_preds)
        
        reg_avg_true = np.mean(y_val_reg, axis=1)
        reg_avg_pred = np.mean(reg_preds, axis=1)
        cls_true = y_val_cls[:, -1]
        cls_pred = np.argmax(cls_preds[:, -1, :], axis=1)

        results = {
            'mse': mean_squared_error(reg_avg_true, reg_avg_pred),
            'mae': mean_absolute_error(reg_avg_true, reg_avg_pred),
            'r2': r2_score(reg_avg_true, reg_avg_pred),
            'accuracy': accuracy_score(cls_true, cls_pred),
            'f1_macro': f1_score(cls_true, cls_pred, average='macro')
        }

        print("Evaluation results:", results)
        Visualizer.plot_model_performance_comparison(results, baseline_results)
        Visualizer.plot_feature_importance(self.baseline_models.feature_importance)
        return results