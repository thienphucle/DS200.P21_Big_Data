import argparse
import pandas as pd
from trainer import Trainer
from Feature_Engineering import TikTokFeatureEngineer

def main(args):
    # Load data
    df = pd.read_csv(args.data_path)
    engineer = TikTokFeatureEngineer()
    features = engineer.transform(df)

    if features['training_data'].empty:
        print("❌ No valid training data found.")
        return

    # Initialize trainer
    trainer = Trainer()

    # Prepare data
    train_loader, val_loader, X_train, X_val, y_train_reg, y_val_reg, y_train_cls, y_val_cls, feature_names = \
        trainer.prepare_data(
            features['training_data'],
            features['user_trend_features'],
            features['text_features'],
            batch_size=args.batch_size
        )

    # Train
    if args.model == 'deep':
        baseline_results = trainer.train(
            train_loader, val_loader, X_train, X_val,
            y_train_reg, y_val_reg, y_train_cls, y_val_cls,
            feature_names, epochs=args.epochs, lr=args.lr
        )

        neural_results, reg_preds, cls_preds = trainer.evaluate_and_visualize(
            val_loader, y_val_reg, y_val_cls, baseline_results,
            model_path=args.save_path
        )

        results_df, attn = trainer.predict(
            features['training_data'],
            features['user_trend_features'],
            features['text_features'],
            model_path=args.save_path
        )

        results_df.to_csv(args.output_path, index=False)
        print(f"✅ Predictions saved to {args.output_path}")

    else:
        print("⚠️ Only '--model deep' supported in this script. Use classic or automl runner separately.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deep", help="Model type: deep/classic/automl")
    parser.add_argument("--data_path", type=str, default="training_data.csv", help="CSV path to preprocessed training data")
    parser.add_argument("--save_path", type=str, default="best_model.pth", help="Path to save model")
    parser.add_argument("--output_path", type=str, default="predictions.csv", help="CSV path to save predictions")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")

    args = parser.parse_args()
    main(args)
