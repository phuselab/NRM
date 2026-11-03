"""Training losses: joint (choice, RT) negative log-likelihood (§2.6, Eqs. 9–10)."""

import torch
import torch.nn.functional as F


def per_group_nll_loss(choice_probs, first_cross_density, ts, true_choices,
                       true_rts, group, epoch_n, batch_no, ndt,
                       plot=False, plot_dir="plots"):
    """Joint NLL over the model's (choice, RT) distribution (Eqs. 9–10).

    Computes p(t_j, i) = p_FPT(t_j) * P_choice(i) and evaluates the NLL
    at the observed (choice, RT) pairs. When group is provided, the joint
    distribution is averaged within groups before scoring.
    """
    B = first_cross_density.shape[0]
    T = first_cross_density.shape[1]
    N = choice_probs.shape[1]

    # Joint distribution p(t, i) = p_FPT(t) * P_choice(i) (Eq. 9)
    fcd_weighted = first_cross_density.unsqueeze(2) * choice_probs.unsqueeze(1)

    if group is None:
        model_rt_dist = fcd_weighted / (fcd_weighted.sum(dim=(1, 2), keepdim=True) + 1e-8)
    else:
        num_groups = group.max().item() + 1
        group_sums = torch.zeros(num_groups, T, N, device=fcd_weighted.device)
        group_sums.index_add_(0, group, fcd_weighted)
        group_means = group_sums / group_sums.sum(dim=(1, 2), keepdim=True)
        model_rt_dist = group_means[group]

    rt_indices = torch.bucketize(true_rts, ts) - 1
    rt_indices = rt_indices.clamp(min=0, max=ts.numel() - 1)

    # NLL at observed (choice, RT) (Eq. 10)
    rt_probs = model_rt_dist[torch.arange(B), rt_indices, true_choices]
    rt_loss = -torch.log(rt_probs + 1e-8).mean()

    return None, None, rt_loss


def per_sample_nll_loss(choice_probs, first_cross_density, ts, true_choices,
                        true_rts, epoch_n, batch_no, ndt, beta=1.0,
                        plot=False):
    """Per-sample NLL variant with KL choice loss (Eqs. 9–10)."""
    B = first_cross_density.shape[0]
    T = first_cross_density.shape[1]
    N = choice_probs.shape[1]

    target_choice_dist = torch.bincount(true_choices, minlength=N).float() + 1e-8
    target_choice_dist = target_choice_dist / target_choice_dist.sum()

    model_choice_dist = choice_probs.mean(dim=0) + 1e-8
    model_choice_dist = model_choice_dist / model_choice_dist.sum()

    choice_loss = F.kl_div(model_choice_dist.log(), target_choice_dist,
                           reduction="batchmean")

    fcd_weighted = first_cross_density.unsqueeze(2) * choice_probs.unsqueeze(1)
    model_rt_dist = fcd_weighted / (fcd_weighted.sum(dim=(1, 2), keepdim=True) + 1e-8)

    rt_indices = torch.bucketize(true_rts, ts) - 1
    rt_indices = rt_indices.clamp(min=0, max=ts.numel() - 1)
    rt_probs = model_rt_dist[torch.arange(B), rt_indices, true_choices]

    rt_loss = -(torch.log(rt_probs + 1e-8)).mean()
    combined = rt_loss

    return choice_loss, rt_loss, combined


def nll_loss(choice_probs, first_cross_density, ts, true_choices, true_rts,
             epoch_n, ndt, beta=1.0, plot=False):
    """NLL with non-decision time shift and linear interpolation (Eqs. 9–10)."""
    B = first_cross_density.shape[0]
    T = first_cross_density.shape[1]
    N = choice_probs.shape[1]

    target_choice_dist = torch.bincount(true_choices, minlength=N).float() + 1e-8
    target_choice_dist = target_choice_dist / target_choice_dist.sum()
    model_choice_dist = choice_probs.mean(dim=0) + 1e-8
    model_choice_dist = model_choice_dist / model_choice_dist.sum()
    choice_loss = F.kl_div(model_choice_dist.log(), target_choice_dist,
                           reduction="batchmean")

    fcd_weighted = first_cross_density.unsqueeze(2) * choice_probs.unsqueeze(1)
    model_rt_dist = fcd_weighted.sum(dim=0) + 1e-8
    model_rt_dist = model_rt_dist / model_rt_dist.sum()

    sample_ndt = ndt[true_choices]
    dt = ts[1] - ts[0]
    shifted_true_rts = (true_rts - sample_ndt).clamp(min=0.0, max=ts[-1])
    idx_float = shifted_true_rts / dt
    idx_low = idx_float.floor().long()
    idx_high = (idx_low + 1).clamp(max=T - 1).long()
    w_high = idx_float - idx_low.float()
    w_low = 1.0 - w_high
    p_low = model_rt_dist[idx_low, true_choices]
    p_high = model_rt_dist[idx_high, true_choices]
    rt_probs = w_low * p_low + w_high * p_high

    rt_loss = -(torch.log(rt_probs + 1e-8)).mean()
    combined = rt_loss

    return choice_loss, beta * rt_loss, combined
