"""RT distribution distances: KS, quantile and moment comparisons (§3)."""

import numpy as np
from scipy import stats
from scipy.stats import ks_2samp
from collections import defaultdict


class RTDistributionMetrics:

    def __init__(self, quantiles=None):
        if quantiles is None:
            self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        else:
            self.quantiles = sorted(quantiles)

    def calculate_ks_distance(self, sample1, sample2):
        if len(sample1) < 2 or len(sample2) < 2:
            return None
        ks_stat, p_value = ks_2samp(sample1, sample2)
        return {
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(p_value),
            'n_sample1': len(sample1),
            'n_sample2': len(sample2),
        }

    def calculate_quantile_differences(self, sample1, sample2):
        if len(sample1) < 2 or len(sample2) < 2:
            return None
        q1 = np.quantile(sample1, self.quantiles)
        q2 = np.quantile(sample2, self.quantiles)
        abs_diff = np.abs(q1 - q2)
        rel_diff = 100 * abs_diff / (q1 + 1e-10)
        return {
            'quantiles': self.quantiles,
            'sample1_quantiles': q1.tolist(),
            'sample2_quantiles': q2.tolist(),
            'absolute_differences': abs_diff.tolist(),
            'relative_differences': rel_diff.tolist(),
            'mean_absolute_error': float(np.mean(abs_diff)),
            'mean_relative_error': float(np.mean(rel_diff)),
        }

    def calculate_moment_differences(self, sample1, sample2):
        if len(sample1) < 4 or len(sample2) < 4:
            return None
        moments1 = {
            'mean': float(np.mean(sample1)),
            'std': float(np.std(sample1, ddof=1)),
            'variance': float(np.var(sample1, ddof=1)),
            'skewness': float(stats.skew(sample1)),
            'kurtosis': float(stats.kurtosis(sample1)),
        }
        moments2 = {
            'mean': float(np.mean(sample2)),
            'std': float(np.std(sample2, ddof=1)),
            'variance': float(np.var(sample2, ddof=1)),
            'skewness': float(stats.skew(sample2)),
            'kurtosis': float(stats.kurtosis(sample2)),
        }
        differences = {
            'mean_diff': moments2['mean'] - moments1['mean'],
            'mean_rel_diff': 100 * (moments2['mean'] - moments1['mean']) / (moments1['mean'] + 1e-10),
            'std_diff': moments2['std'] - moments1['std'],
            'std_rel_diff': 100 * (moments2['std'] - moments1['std']) / (moments1['std'] + 1e-10),
            'variance_diff': moments2['variance'] - moments1['variance'],
            'skewness_diff': moments2['skewness'] - moments1['skewness'],
            'kurtosis_diff': moments2['kurtosis'] - moments1['kurtosis'],
        }
        return {
            'sample1_moments': moments1,
            'sample2_moments': moments2,
            'differences': differences,
        }

    def compare_human_human_distributions_split_half(self, image_indices, rts,
                                                     n_splits=10, random_state=None):
        rng = np.random.default_rng(random_state)

        stimulus_rts = defaultdict(list)
        for img_idx, rt in zip(image_indices, rts):
            stimulus_rts[img_idx].append(rt)

        per_stimulus_metrics = {}
        all_ks_values = []

        for img_idx, rt_list in stimulus_rts.items():
            rt_array = np.asarray(rt_list)
            n_obs = len(rt_array)

            if n_obs < 4:
                continue

            ks_values = []
            for _ in range(n_splits):
                perm = rng.permutation(n_obs)
                half = n_obs // 2
                group_a = rt_array[perm[:half]]
                group_b = rt_array[perm[half:]]

                if len(group_a) < 2 or len(group_b) < 2:
                    continue

                ks_stat = ks_2samp(group_a, group_b).statistic
                ks_values.append(ks_stat)

            if len(ks_values) > 0:
                per_stimulus_metrics[img_idx] = {
                    'mean_ks': float(np.mean(ks_values)),
                    'median_ks': float(np.median(ks_values)),
                    'std_ks': float(np.std(ks_values)),
                    'n_splits': len(ks_values),
                }
                all_ks_values.extend(ks_values)

        overall_metrics = {
            'mean_ks_distance': float(np.mean(all_ks_values)) if len(all_ks_values) > 0 else None,
            'median_ks_distance': float(np.median(all_ks_values)) if len(all_ks_values) > 0 else None,
            'std_ks_distance': float(np.std(all_ks_values)) if len(all_ks_values) > 0 else None,
            'n_total_comparisons': len(all_ks_values),
            'n_stimuli_evaluated': len(per_stimulus_metrics),
        }

        return {
            'per_stimulus': per_stimulus_metrics,
            'overall': overall_metrics,
        }

    def compare_model_human_distributions(self, image_indices, human_rts,
                                          model_rt_samples):
        human_rts_by_stimulus = defaultdict(list)
        for img_idx, rt in zip(image_indices, human_rts):
            human_rts_by_stimulus[img_idx].append(rt)

        per_stimulus_metrics = {}

        all_ks = []
        all_ks_pvalues = []
        all_quantile_mae = []
        all_quantile_rel_errors = []
        all_moment_diffs = {
            'mean': [], 'std': [], 'skewness': [], 'kurtosis': [],
        }

        n_stimuli_evaluated = 0

        for img_idx in human_rts_by_stimulus.keys():
            if img_idx not in model_rt_samples:
                continue

            human_sample = np.array(human_rts_by_stimulus[img_idx])
            model_sample = np.array(model_rt_samples[img_idx])

            if len(human_sample) < 2 or len(model_sample) < 2:
                continue

            metrics = {}

            ks_result = self.calculate_ks_distance(human_sample, model_sample)
            if ks_result is not None:
                metrics['ks'] = ks_result
                all_ks.append(ks_result['ks_statistic'])
                all_ks_pvalues.append(ks_result['ks_pvalue'])

            quantile_result = self.calculate_quantile_differences(human_sample, model_sample)
            if quantile_result is not None:
                metrics['quantiles'] = quantile_result
                all_quantile_mae.append(quantile_result['mean_absolute_error'])
                all_quantile_rel_errors.append(quantile_result['mean_relative_error'])

            moment_result = self.calculate_moment_differences(human_sample, model_sample)
            if moment_result is not None:
                metrics['moments'] = moment_result
                all_moment_diffs['mean'].append(moment_result['differences']['mean_diff'])
                all_moment_diffs['std'].append(moment_result['differences']['std_diff'])
                all_moment_diffs['skewness'].append(moment_result['differences']['skewness_diff'])
                all_moment_diffs['kurtosis'].append(moment_result['differences']['kurtosis_diff'])

            if len(metrics) > 0:
                per_stimulus_metrics[img_idx] = metrics
                n_stimuli_evaluated += 1

        overall_metrics = {
            'ks_distance': {
                'mean': float(np.mean(all_ks)) if all_ks else None,
                'median': float(np.median(all_ks)) if all_ks else None,
                'std': float(np.std(all_ks)) if all_ks else None,
                'mean_pvalue': float(np.mean(all_ks_pvalues)) if all_ks_pvalues else None,
            },
            'quantile_mae': {
                'mean': float(np.mean(all_quantile_mae)) if all_quantile_mae else None,
                'median': float(np.median(all_quantile_mae)) if all_quantile_mae else None,
            },
            'quantile_relative_error': {
                'mean': float(np.mean(all_quantile_rel_errors)) if all_quantile_rel_errors else None,
                'median': float(np.median(all_quantile_rel_errors)) if all_quantile_rel_errors else None,
            },
            'moment_differences': {
                k: {
                    'mean': float(np.mean(v)) if v else None,
                    'std': float(np.std(v)) if v else None,
                }
                for k, v in all_moment_diffs.items()
            },
            'n_stimuli_evaluated': n_stimuli_evaluated,
        }

        return {
            'per_stimulus': per_stimulus_metrics,
            'overall': overall_metrics,
        }
