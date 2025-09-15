import os
import pandas as pd
import numpy as np
from scipy import stats
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Suppress warnings
warnings.filterwarnings('ignore')


# Main categories we need to analyze
CATEGORIES = [
    "Pred_随访信息来源",
    "Pred_受试者是否死亡",
    "Pred_受试者是否住院",
    "Pred_受试者是否手术",
    "Pred_受试者是否用药"
]

# English names for categories (for output)
CATEGORY_TO_ENGLISH = {
    "Pred_随访信息来源": "follow_up_source",
    "Pred_受试者是否死亡": "is_deceased",
    "Pred_受试者是否住院": "is_hospitalized",
    "Pred_受试者是否手术": "had_surgery",
    "Pred_受试者是否用药": "is_medicated"
}

def bootstrap_ci(data, metric_key, n_bootstraps=1000, confidence_level=0.95):
    """Calculate bootstrap confidence intervals for a given metric.
    
    Args:
        data: Array-like of (gt, pr) tuples
        metric_key: String indicating which metric to calculate ('accuracy', 'sensitivity', etc.)
        n_bootstraps: Number of bootstrap samples
        confidence_level: Confidence level for intervals (default: 0.95)
        
    Returns:
        Tuple of (actual_metric, lower_bound, upper_bound)
    """
    if len(data) == 0:
        return 0, 0, 0
    
    # Convert data to numpy arrays for vectorized operations
    data_array = np.array(data, dtype=object)
    gt = np.array([g for g, _ in data_array])
    pr = np.array([p for _, p in data_array])
    
    # Calculate actual metric on full dataset
    actual_metric = calc_metric(gt, pr, metric_key)
    
    # Try to use CUDA if available
    use_cuda = False
    try:
        import torch
        if torch.cuda.is_available():
            use_cuda = True
            bootstrap_metrics = bootstrap_cuda(gt, pr, metric_key, n_bootstraps)
        else:
            bootstrap_metrics = bootstrap_cpu(gt, pr, metric_key, n_bootstraps)
    except ImportError:
        bootstrap_metrics = bootstrap_cpu(gt, pr, metric_key, n_bootstraps)
    
    # Calculate confidence intervals
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100
    lower_bound = np.percentile(bootstrap_metrics, lower_percentile)
    upper_bound = np.percentile(bootstrap_metrics, upper_percentile)
    
    return actual_metric, lower_bound, upper_bound

def calc_metric(gt, pr, key):
    """Calculate a specific metric from ground truth and predictions."""
    if len(gt) == 0:
        return 0
        
    if key == 'accuracy':
        # Simple equality check
        return np.mean(gt == pr)
        
    elif key == 'sensitivity':
        # True positive rate: TP / (TP + FN)
        positives = np.sum(gt == 1)
        if positives == 0:
            return 0
        true_positives = np.sum((gt == 1) & (pr == 1))
        return true_positives / positives
        
    elif key == 'specificity':
        # True negative rate: TN / (TN + FP)
        negatives = np.sum(gt == 2)
        if negatives == 0:
            return 0
        true_negatives = np.sum((gt == 2) & (pr == 2))
        return true_negatives / negatives
        
    elif key == 'ppv':
        # Positive predictive value: TP / (TP + FP)
        predicted_positives = np.sum(pr == 1)
        if predicted_positives == 0:
            return 0
        true_positives = np.sum((gt == 1) & (pr == 1))
        return true_positives / predicted_positives
        
    elif key == 'npv':
        # Negative predictive value: TN / (TN + FN)
        predicted_negatives = np.sum(pr == 2)
        if predicted_negatives == 0:
            return 0
        true_negatives = np.sum((gt == 2) & (pr == 2))
        return true_negatives / predicted_negatives
        
    return 0

def bootstrap_cuda(gt, pr, metric_key, n_bootstraps):
    """Run bootstrap using CUDA acceleration."""
    import torch
    
    # Convert to CUDA tensors
    gt_tensor = torch.tensor(gt, dtype=torch.float32).cuda()
    pr_tensor = torch.tensor(pr, dtype=torch.float32).cuda()
    
    n = len(gt)
    bootstrap_metrics = torch.zeros(n_bootstraps, device='cuda')
    
    # Generate bootstrap sample indices
    indices = torch.randint(0, n, (n_bootstraps, n), device='cuda')
    
    # Process in batches for memory efficiency
    batch_size = 100  # Process 100 bootstraps at a time
    for b in range(0, n_bootstraps, batch_size):
        end_idx = min(b + batch_size, n_bootstraps)
        
        for i in range(b, end_idx):
            # Get bootstrap sample indices
            idx = indices[i]
            # Sample the data
            sample_gt = torch.index_select(gt_tensor, 0, idx)
            sample_pr = torch.index_select(pr_tensor, 0, idx)
            
            # Calculate metric based on type
            if metric_key == 'accuracy':
                bootstrap_metrics[i] = torch.mean((sample_gt == sample_pr).float())
            
            elif metric_key == 'sensitivity':
                positives = torch.sum(sample_gt == 1)
                if positives == 0:
                    bootstrap_metrics[i] = 0
                else:
                    true_positives = torch.sum((sample_gt == 1) & (sample_pr == 1))
                    bootstrap_metrics[i] = true_positives / positives
            
            elif metric_key == 'specificity':
                negatives = torch.sum(sample_gt == 2)
                if negatives == 0:
                    bootstrap_metrics[i] = 0
                else:
                    true_negatives = torch.sum((sample_gt == 2) & (sample_pr == 2))
                    bootstrap_metrics[i] = true_negatives / negatives
            
            elif metric_key == 'ppv':
                predicted_positives = torch.sum(sample_pr == 1)
                if predicted_positives == 0:
                    bootstrap_metrics[i] = 0
                else:
                    true_positives = torch.sum((sample_gt == 1) & (sample_pr == 1))
                    bootstrap_metrics[i] = true_positives / predicted_positives
            
            elif metric_key == 'npv':
                predicted_negatives = torch.sum(sample_pr == 2)
                if predicted_negatives == 0:
                    bootstrap_metrics[i] = 0
                else:
                    true_negatives = torch.sum((sample_gt == 2) & (sample_pr == 2))
                    bootstrap_metrics[i] = true_negatives / predicted_negatives
    
    # Return as numpy array
    return bootstrap_metrics.cpu().numpy()

def bootstrap_cpu(gt, pr, metric_key, n_bootstraps):
    """Run bootstrap using CPU."""
    n = len(gt)
    bootstrap_metrics = np.zeros(n_bootstraps)
    
    # Generate bootstrap sample indices
    indices = np.random.randint(0, n, size=(n_bootstraps, n))
    
    # Calculate metrics for each bootstrap sample
    for i in range(n_bootstraps):
        idx = indices[i]
        sample_gt = gt[idx]
        sample_pr = pr[idx]
        bootstrap_metrics[i] = calc_metric(sample_gt, sample_pr, metric_key)
    
    return bootstrap_metrics

def calculate_metrics(gt_series, pr_series):
    """Calculate performance metrics between ground truth and prediction series."""
    # Convert to numeric, handling non-numeric values
    gt_series = pd.to_numeric(gt_series, errors='coerce')
    pr_series = pd.to_numeric(pr_series, errors='coerce')
    
    # For all metrics: filter valid indices
    valid_indices = ~gt_series.isna()
    gt_all = gt_series[valid_indices]
    pr_all = pr_series[valid_indices]
    
    # For restricted metrics (sensitivity, specificity, etc.): only use values 1 or 2
    valid_restricted = gt_all.isin([1, 2])
    gt_restricted = gt_all[valid_restricted]
    pr_restricted = pr_all[valid_restricted]
    
    # Calculate components for accuracy
    correct_accuracy_predictions = np.sum(gt_all == pr_all)
    total_accuracy_samples = len(gt_all) # This is 'sample_size'
    
    # If no valid data for restricted metrics, return zeros for those, but accuracy might still be valid
    if len(gt_restricted) == 0:
        # Calculate accuracy if data_all is available
        accuracy_val, acc_low_ci, acc_high_ci = (0,0,0)
        if len(gt_all) > 0:
            accuracy_val, acc_low_ci, acc_high_ci = bootstrap_ci(list(zip(gt_all, pr_all)), 'accuracy')

        return {
            'accuracy': (accuracy_val, acc_low_ci, acc_high_ci),
            'sensitivity': (0, 0, 0),
            'specificity': (0, 0, 0),
            'ppv': (0, 0, 0),
            'npv': (0, 0, 0),
            'sample_size': total_accuracy_samples,
            'positives': 0,
            'negatives': 0,
            'predicted_positives': 0,
            'predicted_negatives': 0,
            'TP': 0,
            'TN': 0,
            'correct_accuracy_predictions': correct_accuracy_predictions if len(gt_all) > 0 else 0
        }
    
    # Map invalid prediction values (not 1 or 2) to the opposite of GT for binary metrics
    pr_mapped = pr_restricted.copy()
    invalid_mask = ~pr_mapped.isin([1, 2])
    
    # For GT=1 (positive), map invalid predictions to 2 (negative)
    pos_invalid = (gt_restricted == 1) & invalid_mask
    pr_mapped.loc[pos_invalid] = 2
    
    # For GT=2 (negative), map invalid predictions to 1 (positive)
    neg_invalid = (gt_restricted == 2) & invalid_mask
    pr_mapped.loc[neg_invalid] = 1
    
    # Calculate TP, TN, predicted_positives, predicted_negatives from the mapped predictions for restricted set
    TP = np.sum((gt_restricted == 1) & (pr_mapped == 1))
    TN = np.sum((gt_restricted == 2) & (pr_mapped == 2))
    predicted_positives_count = (pr_mapped == 1).sum() # TP + FP
    predicted_negatives_count = (pr_mapped == 2).sum() # TN + FN
    
    # Prepare data for bootstrap
    data_all = list(zip(gt_all, pr_all))
    data_restricted = list(zip(gt_restricted, pr_mapped))
    
    # Calculate metrics with bootstrap CI
    accuracy = bootstrap_ci(data_all, 'accuracy')
    sensitivity = bootstrap_ci(data_restricted, 'sensitivity')
    specificity = bootstrap_ci(data_restricted, 'specificity')
    ppv = bootstrap_ci(data_restricted, 'ppv')
    npv = bootstrap_ci(data_restricted, 'npv')
    
    # Count positives and negatives from ground truth of restricted set
    positives = (gt_restricted == 1).sum() # TP + FN
    negatives = (gt_restricted == 2).sum() # TN + FP
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'sample_size': total_accuracy_samples, # For accuracy denominator
        'positives': positives, # For sensitivity denominator
        'negatives': negatives, # For specificity denominator
        'predicted_positives': predicted_positives_count, # For PPV denominator
        'predicted_negatives': predicted_negatives_count, # For NPV denominator
        'TP': TP,
        'TN': TN,
        'correct_accuracy_predictions': correct_accuracy_predictions
    }

def create_confusion_matrix_plot(gt_values, pr_values, output_path, category_name):
    """Create a confusion matrix plot and save it."""
    # Convert to numeric and filter valid values
    gt_series = pd.to_numeric(gt_values, errors='coerce')
    pr_series = pd.to_numeric(pr_values, errors='coerce')
    
    # Filter out NaN values - both GT and PR must be valid
    valid_indices = ~gt_series.isna() & ~pr_series.isna()
    gt_clean = gt_series[valid_indices]
    pr_clean = pr_series[valid_indices]
    
    if len(gt_clean) == 0:
        print(f"      Warning: No valid paired data for {category_name} after filtering NaN values")
        return
    
    # Create confusion matrix
    
    # Create 4x4 confusion matrix - data actually uses labels 1,2,3,4 not 0,1,2,3
    # Swap axes: PR is on y-axis (rows), GT is on x-axis (columns)
    cm = confusion_matrix(pr_clean, gt_clean, labels=[1, 2, 3, 4])
    
    print(f"      Confusion matrix for {category_name}: PR(纵轴) vs GT(横轴)")
    print(f"      Matrix shape: {cm.shape}, Total samples: {np.sum(cm)}")
    print(f"      Valid paired samples: {len(gt_clean)} (filtered from original {len(gt_values)})")
    
    # Create labeled plot
    plt.figure(figsize=(6, 5))
    
    # Use a colormap similar to the example image
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     cbar_kws={'shrink': 0.8}, 
                     square=True, 
                     linewidths=0.5,
                     annot_kws={'size': 12})
    
    # Add labels to make orientation clear
    plt.xlabel('Ground Truth (GT)', fontsize=12)
    plt.ylabel('Predicted (PR)', fontsize=12)
    plt.title(f'{category_name}', fontsize=14)
    
    # Set tick labels - data uses 1,2,3,4
    labels = ['1', '2', '3', '4']
    plt.xticks(np.arange(4) + 0.5, labels)
    plt.yticks(np.arange(4) + 0.5, labels, rotation=0)
    
    # Save labeled plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create clean plot (no labels, no ticks, no title)
    plt.figure(figsize=(6, 5))
    
    # Use a colormap similar to the example image
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     cbar_kws={'shrink': 0.8}, 
                     square=True, 
                     linewidths=0.5,
                     annot_kws={'size': 12})
    
    # Remove all labels, ticks, and title
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')
    plt.xticks([])
    plt.yticks([])
    
    # Create clean filename with prefix "0_clean_" to sort first
    clean_output_path = output_path.parent / f"0_clean_{output_path.name}"
    
    # Save clean plot
    plt.tight_layout()
    plt.savefig(clean_output_path, dpi=300, bbox_inches='tight')
    plt.close()

def find_category_columns(df, sheet_name):
    """Find the PR and GT columns for each category in the dataframe.
    GT column is simply the category name with 'Pred' replaced by 'GT'.
    """
    column_pairs = {}

    col_map = {str(col): i for i, col in enumerate(df.columns)}

    print(f"    Available columns in {sheet_name}: {list(df.columns)}")

    for category in CATEGORIES:
        if category not in col_map:
            print(f"    Warning: Category '{category}' not found in sheet {sheet_name}")
            continue

        pr_idx = col_map[category]
        gt_col_name = category.replace('Pred', 'GT', 1)
        if gt_col_name not in col_map:
            print(f"    Warning: GT column '{gt_col_name}' for '{category}' not found in sheet {sheet_name}")
            continue

        gt_idx = col_map[gt_col_name]
        column_pairs[category] = (pr_idx, gt_idx)
        print(f"    Found pair: {category} (col {pr_idx}) -> {gt_col_name} (col {gt_idx})")

    return column_pairs

def process_file_for_metrics(file_path):
    """Process a single Excel file and calculate metrics for each sheet."""
    print(f"Processing file for metrics: {file_path}")
    
    # Dictionary to store results for each sheet
    results = {}
    model_names = set()
    
    # Dictionary to track unique ground truth data
    gt_data_tracker = {category: {} for category in CATEGORIES}
    
    # Load Excel file
    try:
        with pd.ExcelFile(file_path) as xls:
            # Process each sheet
            for sheet_name in xls.sheet_names:
                try:
                    print(f"  Processing sheet: {sheet_name}")
                    df = pd.read_excel(xls, sheet_name)
                    
                    # Find ID column
                    id_col = None
                    for i, col in enumerate(df.columns):
                        if any(term in str(col).lower() for term in ['dcap', 'id']):
                            id_col = i
                            break
                    
                    # Create sequential IDs if no ID column found
                    if id_col is None:
                        df['_id'] = range(len(df))
                        id_col = df.columns.get_loc('_id')
                    
                    # Find category columns and their GT pairs
                    column_pairs = find_category_columns(df, sheet_name)
                    if not column_pairs:
                        print(f"  Warning: No valid category-GT column pairs found in sheet {sheet_name}. Skipping.")
                        print(f"  Available columns in {sheet_name}: {list(df.columns)}")
                        continue
                    
                    # Process each category and collect metrics
                    experiment_results = {}
                    all_gt_values = []
                    all_pr_values = []
                    
                    for category, (pr_col_idx, gt_col_idx) in column_pairs.items():
                        # Extract GT and PR values
                        gt_values = df.iloc[:, gt_col_idx]
                        pr_values = df.iloc[:, pr_col_idx]
                        
                        # Add to overall metrics collection
                        all_gt_values.extend(gt_values)
                        all_pr_values.extend(pr_values)
                        
                        # Track unique GT data using ID to avoid duplicates
                        for i, (gt_val, id_val) in enumerate(zip(gt_values, df.iloc[:, id_col])):
                            if not pd.isna(gt_val):
                                data_key = f"{id_val}_{category}"
                                if data_key not in gt_data_tracker[category]:
                                    gt_data_tracker[category][data_key] = gt_val
                        
                        # Calculate metrics
                        metrics = calculate_metrics(gt_values, pr_values)
                        
                        # Map category to English for output
                        english_category = CATEGORY_TO_ENGLISH.get(category, category)
                        experiment_results[english_category] = metrics
                        
                        print(f"    Category: {category} - Found {metrics['sample_size']} valid samples " +
                              f"(Pos: {metrics['positives']}, Neg: {metrics['negatives']})")
                    
                    # Calculate overall metrics for this model (all categories combined)
                    if all_gt_values and all_pr_values:
                        combined_metrics = calculate_metrics(pd.Series(all_gt_values), pd.Series(all_pr_values))
                        experiment_results['Overall'] = combined_metrics
                        
                        print(f"    Category: Overall - Found {combined_metrics['sample_size']} valid samples " +
                              f"(Pos: {combined_metrics['positives']}, Neg: {combined_metrics['negatives']})")
                    
                    # Print summary of found categories for this sheet
                    print(f"    Sheet {sheet_name}: Found {len(column_pairs)} categories")
                    print(f"    Found categories: {list(column_pairs.keys())}")
                    
                    # Store results and add to model names
                    results[sheet_name] = experiment_results
                    model_names.add(sheet_name)
                    
                except Exception as e:
                    print(f"  Error processing sheet {sheet_name}: {e}")
            
            # Update metrics with accurate unique counts
            for sheet_name, sheet_results in results.items():
                for category in CATEGORIES:
                    english_category = CATEGORY_TO_ENGLISH.get(category, category)
                    if english_category in sheet_results:
                        # Get unique GT values for this category
                        unique_gt_values = list(gt_data_tracker[category].values())
                        # Update sample size and counts
                        if unique_gt_values:
                            sheet_results[english_category]['sample_size'] = len(unique_gt_values)
                            sheet_results[english_category]['positives'] = sum(1 for val in unique_gt_values if val == 1)
                            sheet_results[english_category]['negatives'] = sum(1 for val in unique_gt_values if val == 2)
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    
    return results, model_names

def plot_file(file_path, plot_dir):
    """Generate confusion matrix plots for a single Excel file."""
    print(f"Generating plots for file: {file_path}")
    
    # Get base filename for plot directory
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    file_plots_dir = plot_dir / base_filename
    file_plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Load Excel file
    try:
        with pd.ExcelFile(file_path) as xls:
            # Process each sheet
            for sheet_name in xls.sheet_names:
                try:
                    print(f"  Generating plots for sheet: {sheet_name}")
                    df = pd.read_excel(xls, sheet_name)
                    
                    # Create sheet plot directory
                    sheet_plots_dir = file_plots_dir / sheet_name
                    sheet_plots_dir.mkdir(exist_ok=True, parents=True)
                    
                    # Find category columns and their GT pairs
                    column_pairs = find_category_columns(df, sheet_name)
                    if not column_pairs:
                        print(f"  Warning: No valid category-GT column pairs found in sheet {sheet_name}. Skipping plots.")
                        continue
                    
                    # Generate plots for each category
                    all_gt_values = []
                    all_pr_values = []
                    
                    for category, (pr_col_idx, gt_col_idx) in column_pairs.items():
                        # Extract GT and PR values
                        gt_values = df.iloc[:, gt_col_idx]
                        pr_values = df.iloc[:, pr_col_idx]
                        
                        # Add to overall collection
                        all_gt_values.extend(gt_values)
                        all_pr_values.extend(pr_values)
                        
                        # Map category to English for output
                        english_category = CATEGORY_TO_ENGLISH.get(category, category)
                        
                        # Generate confusion matrix plot for this category
                        plot_filename = f"{sheet_name}_{english_category}-vs-gt.png"
                        plot_path = sheet_plots_dir / plot_filename
                        create_confusion_matrix_plot(gt_values, pr_values, plot_path, english_category)
                        
                        print(f"    Generated plot: {plot_filename}")
                    
                    # Generate overall confusion matrix plot
                    if all_gt_values and all_pr_values:
                        overall_plot_filename = f"{sheet_name}_Overall-vs-gt.png"
                        overall_plot_path = sheet_plots_dir / overall_plot_filename
                        create_confusion_matrix_plot(pd.Series(all_gt_values), pd.Series(all_pr_values), 
                                                   overall_plot_path, "Overall")
                        print(f"    Generated plot: {overall_plot_filename}")
                    
                    print(f"    Sheet {sheet_name}: Generated {len(column_pairs)} category plots + 1 overall plot = {len(column_pairs) + 1} total plots")
                    
                except Exception as e:
                    print(f"  Error generating plots for sheet {sheet_name}: {e}")
    
    except Exception as e:
        print(f"Error generating plots for file {file_path}: {e}")

def create_summary_excel(all_results, output_path):
    """Create a summary Excel file with metrics."""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Metrics to display
        metrics = [
            ('accuracy', '一致性'),
            ('sensitivity', '敏感性'),
            ('specificity', '特异性'),
            ('ppv', '阳性预测值'),
            ('npv', '阴性预测值')
        ]
        
        # Prepare category names (English for lookup, Chinese for display)
        categories = list(CATEGORY_TO_ENGLISH.values()) + ['Overall']
        category_chinese = {v: k for k, v in CATEGORY_TO_ENGLISH.items()}
        category_chinese['Overall'] = '总体'
        
        # Create a sheet for each file
        for file_name, file_results in all_results.items():
            # Prepare data structure for sheet
            rows = []
            
            # Create header row
            header = ['Model']
            for category in categories:
                header.extend([category_chinese.get(category, category), '95%CI'])
            rows.append(header)
            
            # Add each metric section
            for metric_key, metric_name in metrics:
                # Add metric name row
                rows.append([metric_name])
                
                # Add sample size row
                sample_row = ['样本量']
                
                # Get first model's data for sample sizes
                if file_results:
                    first_model = next(iter(file_results.values()))
                    for category in categories:
                        if category in first_model:
                            item = first_model[category]
                            if metric_key == 'accuracy':
                                sample_row.extend([f"{item['sample_size']}/{item['positives']}", ''])
                            elif metric_key == 'sensitivity':
                                sample_row.extend([f"{item['positives'] + item['negatives']}/{item['positives']}", ''])
                            elif metric_key == 'specificity':
                                sample_row.extend([f"{item['positives'] + item['negatives']}/{item['negatives']}", ''])
                            else:
                                sample_row.extend(['', ''])
                        else:
                            sample_row.extend(['', ''])
                rows.append(sample_row)
                
                # Add each model's metrics
                for model_name, model_results in file_results.items():
                    model_row = [model_name]
                    
                    for category in categories:
                        if category in model_results:
                            item = model_results[category] # This is the metrics dict
                            metric_value, lower_ci, upper_ci = item.get(metric_key, (0, 0, 0))
                            
                            num = 0
                            den = 0

                            if metric_key == 'accuracy':
                                num = item.get('correct_accuracy_predictions', 0)
                                den = item.get('sample_size', 0)
                            elif metric_key == 'sensitivity':
                                num = item.get('TP', 0)
                                den = item.get('positives', 0)
                            elif metric_key == 'specificity':
                                num = item.get('TN', 0)
                                den = item.get('negatives', 0)
                            elif metric_key == 'ppv':
                                num = item.get('TP', 0) # Numerator is TP
                                den = item.get('predicted_positives', 0)
                            elif metric_key == 'npv':
                                num = item.get('TN', 0) # Numerator is TN
                                den = item.get('predicted_negatives', 0)
                            
                            metric_str = f"{metric_value * 100:.2f}% ({int(num)}/{int(den)})"
                            # double check the metric_value is consistent with the num and den, with 0.01 error
                            if abs(metric_value - num / den) > 0.01:
                                print(f"Warning: Metric value inconsistency for {file_name} - {model_name} - {category} - {metric_key}: {metric_value} != {num}/{den}")
                            model_row.extend([metric_str, f"({lower_ci:.3f}-{upper_ci:.3f})"])
                        else:
                            model_row.extend(['', ''])
                    
                    rows.append(model_row)
                
                # Add blank row between metric sections
                rows.append([''])
            
            # Create dataframe and write to Excel
            df = pd.DataFrame(rows)
            
            # Create valid sheet name (max 31 chars, no invalid chars)
            sheet_name = os.path.splitext(os.path.basename(file_name))[0]
            sheet_name = sheet_name[:31].replace('/', '_').replace('\\', '_').replace('?', '_').replace('*', '_').replace('[', '_').replace(']', '_').replace(':', '_')
            
            # Write to Excel without headers (using the data as headers)
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

def main():
    """Main function to process all files and generate output."""
    import argparse

    parser = argparse.ArgumentParser(description="Process evaluation results and generate summary/plots.")
    parser.add_argument('--result_dir_name', type=str, required=True, help='Name of the result directory (under data/eval)')
    parser.add_argument('--files', type=str, nargs='+', required=True, help='List of xlsx files to process (under data/eval)')
    args = parser.parse_args()

    # Define paths
    SOURCE_DIR = Path('data/eval')
    RESULT_DIR = SOURCE_DIR / args.result_dir_name
    PLOTS_DIR = RESULT_DIR / "plots"
    RESULT_DIR.mkdir(exist_ok=True, parents=True)
    PLOTS_DIR.mkdir(exist_ok=True, parents=True)

    # List of files to process
    # 一个xlsx的多个sheet最后会汇总成一个统计表格
    files_to_process = args.files
    all_results = {}
    
    for file_name in files_to_process:
        file_path = SOURCE_DIR / file_name
        if not file_path.exists():
            print(f"Warning: File {file_path} not found. Skipping.")
            continue
        
        # Get base filename without extension for output
        base_name = os.path.splitext(file_name)[0]
        
        # Process file for metrics calculation
        results, model_names = process_file_for_metrics(file_path)
        all_results[base_name] = results
        
        # Generate plots for this file
        plot_file(file_path, PLOTS_DIR)
    
    # Create summary file with all results
    summary_path = RESULT_DIR / "summary_metrics.xlsx"
    create_summary_excel(all_results, summary_path)
    print(f"Summary results saved to {summary_path}")
    print(f"Confusion matrix plots saved to {PLOTS_DIR}")
    
    print("All files processed successfully.")

if __name__ == "__main__":
    main()