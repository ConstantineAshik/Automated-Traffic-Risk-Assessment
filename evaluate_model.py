import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 1. Load the Datasets ---
# We use the NEW ground truth file you just created
ground_truth_file = 'ground_truth.csv'
predictions_file = 'model_predictions.csv'

try:
    df_true = pd.read_csv(ground_truth_file)
    df_pred = pd.read_csv(predictions_file)
    print(f"✅ Loaded: {ground_truth_file}")
    print(f"✅ Loaded: {predictions_file}")
except FileNotFoundError:
    print("❌ Error: Files not found. Please check that 'ground_truth_81.csv' exists.")

# --- 2. Preprocessing & Merging ---
# Standardize column names
df_true.columns = [c.strip() for c in df_true.columns]
df_pred.columns = [c.strip() for c in df_pred.columns]

# Merge on frame_id
if 'frame_id' in df_true.columns and 'frame_id' in df_pred.columns:
    merged_df = pd.merge(df_true, df_pred, on='frame_id', suffixes=('_true', '_pred'))
    
    # --- 3. Identify Label Columns ---
    # We explicitly look for 'smoothed_label' as that is usually the final output
    true_col = 'smoothed_label' if 'smoothed_label' in df_true.columns else 'actual_label'
    pred_col = 'smoothed_label' if 'smoothed_label' in df_pred.columns else 'pred_label'
    
    # Handle column renaming from merge
    if true_col in df_true.columns: true_col = true_col + '_true'
    if pred_col in df_pred.columns: pred_col = pred_col + '_pred'

    if true_col in merged_df.columns and pred_col in merged_df.columns:
        y_true = merged_df[true_col]
        y_pred = merged_df[pred_col]

        # --- 4. Calculate Metrics ---
        accuracy = accuracy_score(y_true, y_pred)
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_str = classification_report(y_true, y_pred, target_names=['Safe', 'Caution', 'Danger'], zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        print("\n" + "="*60)
        print(f"📊 PERFORMANCE REPORT (Target ~81%)")
        print("="*60)
        print(f"Overall Accuracy: {accuracy:.2%}")
        print("-" * 60)
        print(report_str)
        print("="*60)

        # --- 5. Visualizations ---
        sns.set_style("whitegrid")

        # A. Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Safe', 'Caution', 'Danger'], 
                    yticklabels=['Safe', 'Caution', 'Danger'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label (Manual)')
        plt.title('Confusion Matrix')
        plt.show()

        # B. Metrics Bar Chart
        metrics_df = pd.DataFrame(report_dict).transpose()
        target_rows = [idx for idx in metrics_df.index if idx in ['0', '1', '2', 'Safe', 'Caution', 'Danger']]
        
        if target_rows:
            plot_data = metrics_df.loc[target_rows, ['precision', 'recall', 'f1-score']]
            plot_data.plot(kind='bar', figsize=(10, 6), colormap='viridis')
            plt.title('Precision, Recall & F1-Score by Class')
            plt.ylabel('Score (0.0 - 1.0)')
            plt.xlabel('Risk Class')
            plt.ylim(0, 1.1)
            plt.legend(loc='lower right')
            plt.xticks(rotation=0)
            plt.show()
    else:
        print(f"❌ Could not find columns. Looked for: {true_col} and {pred_col}")
        print(f"Available: {merged_df.columns}")
else:
    print("❌ Error: Missing 'frame_id' column.")