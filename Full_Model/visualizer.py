import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    
    @staticmethod
    def plot_model_performance_comparison(neural_results, baseline_results, save_path='model_performance.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Regression comparison (R²)
        reg_models = [k for k in baseline_results if 'reg' in k or 'linear' in k]
        r2_scores = [baseline_results[m].get('r2', 0) for m in reg_models]
        r2_scores.append(neural_results.get('r2', 0))
        model_names = [m.replace('_', ' ').title() for m in reg_models] + ['NeuralNet']
        
        bars = ax1.bar(model_names, r2_scores, color='skyblue')
        ax1.set_title("Regression Performance (R² Score)")
        ax1.set_ylabel("R²")
        ax1.set_ylim(0, max(r2_scores) + 0.05)
        ax1.grid(True, linestyle='--', alpha=0.5)
        for bar, score in zip(bars, r2_scores):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{score:.3f}", ha='center', va='bottom')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # Classification comparison (F1 Macro)
        cls_models = [k for k in baseline_results if 'cls' in k or 'logistic' in k]
        f1_scores = [baseline_results[m].get('f1_macro', 0) for m in cls_models]
        f1_scores.append(neural_results.get('f1_macro', 0))
        model_names_cls = [m.replace('_', ' ').title() for m in cls_models] + ['NeuralNet']

        bars2 = ax2.bar(model_names_cls, f1_scores, color='lightgreen')
        ax2.set_title("Classification Performance (F1 Macro)")
        ax2.set_ylabel("F1 Macro")
        ax2.set_ylim(0, max(f1_scores) + 0.05)
        ax2.grid(True, linestyle='--', alpha=0.5)
        for bar, score in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{score:.3f}", ha='center', va='bottom')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()

    @staticmethod
    def plot_feature_importance(feature_importance_dict, top_n=15, save_path='feature_importance.png'):
        if not feature_importance_dict:
            print("No feature importance found.")
            return

        best_model = max(feature_importance_dict, key=lambda k: len(feature_importance_dict[k]))
        importances = feature_importance_dict[best_model]
        sorted_feat = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]

        features, scores = zip(*sorted_feat)
        plt.figure(figsize=(10, 6))
        bars = plt.barh(features, scores, color=plt.cm.viridis(np.linspace(0.2, 0.8, top_n)))
        plt.gca().invert_yaxis()
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Important Features ({best_model})")
        for bar, val in zip(bars, scores):
            plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va='center')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
