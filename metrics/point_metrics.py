"""Trial-level choice metrics: accuracy, AUC, Cohen's kappa (§3)."""

import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def safe_auc(true_labels, pred_probs) -> float:
    """Macro AUC per coherence level; returns np.nan on any failure."""
    try:
        y = np.asarray(true_labels, dtype=int)
        p = np.asarray(pred_probs,  dtype=float)
        if len(np.unique(y)) < 2:
            return float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def safe_kappa(true_labels, pred_choices) -> float:
    """Cohen's kappa; returns np.nan on any failure."""
    try:
        y = np.asarray(true_labels,  dtype=int)
        p = np.asarray(pred_choices, dtype=int)
        if len(np.unique(y)) < 2 or len(np.unique(p)) < 1:
            return float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(cohen_kappa_score(y, p))
    except Exception:
        return float("nan")


def safe_mean(lst: list) -> float:
    v = [x for x in lst if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


class MetricsCalculator:
    """Calculate various performance metrics."""
    
    @staticmethod
    def calculate_accuracy(predictions, targets):
        """Calculate accuracy between predictions and targets."""
        correct = (predictions == targets).sum()
        total = len(predictions)
        return correct / total if total > 0 else 0.0, correct, total
    
    @staticmethod
    def calculate_auc(predicted_probs, true_labels, n_classes):
        """
        Calculate macro-averaged AUC for multi-class classification.
        
        Args:
            predicted_probs: Array of predicted probabilities (n_samples, n_classes)
                           or predicted class indices (n_samples,)
            true_labels: Array of true class labels (n_samples,)
            n_classes: Number of classes
            
        Returns:
            float: Macro-averaged AUC (0 to 1)
        """
        
        # Convert true labels to one-hot
        true_labels_binary = label_binarize(true_labels, classes=range(n_classes))
        
        # Handle both probability and class predictions
        if len(predicted_probs.shape) == 1:
            # If we only have class predictions, create simple probabilities
            # (1.0 for predicted class, 0.0 for others)
            n_samples = len(predicted_probs)
            probs = np.zeros((n_samples, n_classes))
            probs[np.arange(n_samples), predicted_probs.astype(int)] = 1.0
            predicted_probs = probs
        
        try:
            # Macro-averaged AUC
            auc = roc_auc_score(true_labels_binary, predicted_probs, 
                               average='macro', multi_class='ovr')
            return float(auc)
        except Exception as e:
            # If AUC calculation fails (e.g., only one class present)
            return None
    
    
    @staticmethod
    def calculate_rt_metrics(predicted_rts, true_rts):
        """Calculate RT-related metrics."""
        pearson_r, pearson_p = pearsonr(true_rts, predicted_rts)
        spearman_r, spearman_p = spearmanr(true_rts, predicted_rts)
        mae = np.abs(predicted_rts - true_rts).mean()
        
        return {
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'mae': float(mae),
            'mean_predicted': float(predicted_rts.mean()),
            'std_predicted': float(predicted_rts.std()),
            'mean_true': float(true_rts.mean()),
            'std_true': float(true_rts.std()),
            'min_predicted': float(predicted_rts.min()),
            'max_predicted': float(predicted_rts.max()),
            'min_true': float(true_rts.min()),
            'max_true': float(true_rts.max())
        }
    
    @staticmethod
    def calculate_agreement_breakdown(model_choices, human_choices, true_labels):
        """Calculate detailed agreement patterns."""
        n_total = len(model_choices)
        
        all_correct = ((model_choices == true_labels) & 
                      (human_choices == true_labels)).sum()
        model_correct_human_wrong = ((model_choices == true_labels) & 
                                     (human_choices != true_labels)).sum()
        human_correct_model_wrong = ((model_choices != true_labels) & 
                                     (human_choices == true_labels)).sum()
        both_wrong_diff = ((model_choices != true_labels) & 
                          (human_choices != true_labels) & 
                          (model_choices != human_choices)).sum()
        both_wrong_same = ((model_choices != true_labels) & 
                          (human_choices != true_labels) & 
                          (model_choices == human_choices)).sum()
        
        return {
            'all_correct': {'count': int(all_correct), 'pct': float(all_correct / n_total)},
            'model_correct_human_wrong': {'count': int(model_correct_human_wrong), 
                                         'pct': float(model_correct_human_wrong / n_total)},
            'human_correct_model_wrong': {'count': int(human_correct_model_wrong), 
                                         'pct': float(human_correct_model_wrong / n_total)},
            'both_wrong_different': {'count': int(both_wrong_diff), 
                                    'pct': float(both_wrong_diff / n_total)},
            'both_wrong_same_error': {'count': int(both_wrong_same), 
                                     'pct': float(both_wrong_same / n_total)}
        }


class HumanAgreementAnalyzer:
    """Analyse human-human agreement on images with multiple responses."""
    
    @staticmethod
    def analyze_multi_response_images(image_indices, human_choices, true_labels, rts):
        """Analyse images that have multiple human responses."""
        # Group data by image
        image_groups = defaultdict(list)
        for i, img_idx in enumerate(image_indices):
            image_groups[img_idx].append({
                'human_choice': human_choices[i],
                'true_label': true_labels[i],
                'rt': rts[i]
            })
        
        # Filter to images with multiple responses
        multi_response_images = {k: v for k, v in image_groups.items() if len(v) > 1}
        
        if len(multi_response_images) == 0:
            return None
        
        # Calculate pairwise agreement
        pairwise_agreements = []
        choice_variances = []
        rt_variances = []
        human_rt_pairs = []
        
        for img_idx, responses in multi_response_images.items():
            choices = [r['human_choice'] for r in responses]
            rts = [r['rt'] for r in responses]
            
            # Pairwise agreement
            n_responses = len(choices)
            agreements = 0
            total_pairs = 0
            
            for i in range(n_responses):
                for j in range(i+1, n_responses):
                    if choices[i] == choices[j]:
                        agreements += 1
                    total_pairs += 1
                    human_rt_pairs.append((rts[i], rts[j]))
            
            if total_pairs > 0:
                pairwise_agreements.append(agreements / total_pairs)
            
            choice_variances.append(len(set(choices)))
            if len(rts) > 1:
                rt_variances.append(np.std(rts))
        
        # Calculate Cohen's kappa for images with exactly 2 responses
        images_with_2_responses = {k: v for k, v in multi_response_images.items() 
                                   if len(v) == 2}
        kappa = None
        if len(images_with_2_responses) > 0:
            rater1_choices = [responses[0]['human_choice'] 
                            for responses in images_with_2_responses.values()]
            rater2_choices = [responses[1]['human_choice'] 
                            for responses in images_with_2_responses.values()]
            kappa = cohen_kappa_score(rater1_choices, rater2_choices)
        
        # RT correlation
        rt_corr = None
        rt_pairs_array = None
        if len(human_rt_pairs) > 0:
            rt_pairs_array = np.array(human_rt_pairs)
            rt_corr, rt_p = pearsonr(rt_pairs_array[:, 0], rt_pairs_array[:, 1])
            rt_corr_dict = {'r': float(rt_corr), 'p': float(rt_p)}
        else:
            rt_corr_dict = None
        
        return {
            'n_images_total': len(image_groups),
            'n_images_multi_response': len(multi_response_images),
            'avg_pairwise_agreement': float(np.mean(pairwise_agreements)),
            'avg_unique_choices': float(np.mean(choice_variances)),
            'avg_rt_std': float(np.mean(rt_variances)),
            'cohens_kappa': float(kappa) if kappa is not None else None,
            'n_images_for_kappa': len(images_with_2_responses),
            'rt_correlation': rt_corr_dict,
            'rt_pairs': rt_pairs_array
        }

