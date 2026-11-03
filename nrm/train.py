"""Training loop with early stopping and sensory noise regularisation."""

import torch
from nrm.losses import per_group_nll_loss


def validate_model(model, val_loader, n_reps=150, device='cuda',
                   epoch_num=None, plot_dir="plots"):
    """Evaluate on the validation set using per_group_nll_loss (§2.6)."""
    model.eval()
    total_loss = 0
    batch_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            embeddings = batch['embeddings'].to(device)
            true_choices = batch['choices'].to(device)
            true_rts = batch['rts'].to(device)

            if len(true_rts) < 2:
                continue

            cov = {"img_embedding": embeddings}
            if "time_limits" in batch:
                cov["time_limit"] = batch["time_limits"].to(device, dtype=torch.float32)

            sim_choices, first_cross_density, ts, ndt = model.forward_pass(
                batch_size=len(true_rts),
                covariates=cov,
                n_reps=n_reps
            )

            _, _, loss = per_group_nll_loss(
                choice_probs=sim_choices,
                first_cross_density=first_cross_density,
                ts=ts,
                true_choices=true_choices,
                true_rts=true_rts,
                group=None,
                epoch_n=epoch_num,
                batch_no=None,
                ndt=ndt,
                plot=False,
                plot_dir=plot_dir,
            )

            total_loss += loss.item()
            batch_count += 1

    model.train()
    avg_val_loss = total_loss / batch_count if batch_count > 0 else float('inf')
    return avg_val_loss


def train_model(model, train_loader, val_loader, optimizer, num_epochs=200,
                patience=10, save_path=None, n_reps=150, device='cuda',
                plot_dir="plots", noise_sigma=0.0, early_stop=None):
    """Train with early stopping on validation NLL (§2.6).

    ``early_stop=N`` stops after epoch N instead and keeps that epoch's
    weights, whatever the validation loss.
    """
    if early_stop is not None:
        num_epochs = early_stop
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    print(f"\nStarting training (patience={patience})")
    print("=" * 80)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            embeddings = batch['embeddings'].to(device)
            true_choices = batch['choices'].to(device)
            true_rts = batch['rts'].to(device)

            # Sensory noise regularisation on backbone embeddings
            if noise_sigma > 0.0:
                noise_scale = embeddings.std() * noise_sigma
                embeddings = embeddings + noise_scale * torch.randn_like(embeddings)

            if len(true_rts) < 2:
                continue

            cov = {"img_embedding": embeddings}
            if "time_limits" in batch:
                cov["time_limit"] = batch["time_limits"].to(device, dtype=torch.float32)

            sim_choices, first_cross_density, ts, ndt = model.forward_pass(
                batch_size=len(true_rts),
                covariates=cov,
                n_reps=n_reps
            )

            _, _, loss = per_group_nll_loss(
                choice_probs=sim_choices,
                first_cross_density=first_cross_density,
                ts=ts,
                true_choices=true_choices,
                true_rts=true_rts,
                group=None,
                epoch_n=epoch,
                batch_no=batch_idx,
                ndt=ndt,
                plot=False,
                plot_dir=plot_dir,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"\tEpoch {epoch+1:03d} | Batch {batch_idx+1:3d}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f}", end='\r')

            total_loss += loss.item()
            batch_count += 1

        avg_train_loss = total_loss / batch_count if batch_count > 0 else float('inf')

        print(f"\n\tEvaluating on validation set...", end='')
        avg_val_loss = validate_model(model, val_loader, n_reps=n_reps,
                                      device=device, epoch_num=epoch,
                                      plot_dir=plot_dir)

        print(f"\rEpoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | No Improve: {epochs_no_improve}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_train_loss = avg_train_loss
            epochs_no_improve = 0

            if save_path is not None and early_stop is None:
                torch.save(model.state_dict(), save_path)
                print(f"\tBest model saved (Val Loss: {avg_val_loss:.4f})")

            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if early_stop is None and epochs_no_improve >= patience:
            print(f"\n{'=' * 80}")
            print(f"Early stopping triggered at epoch {epoch+1}")
            print(f"Best Val Loss: {best_val_loss:.4f} (Train Loss: {best_train_loss:.4f})")
            print(f"{'=' * 80}")
            break

    if early_stop is not None:
        if save_path is not None:
            torch.save(model.state_dict(), save_path)
        print(f"\nStopped at epoch {early_stop}"
              + (f"; weights saved to {save_path}" if save_path is not None else ""))
    elif best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best model with Val Loss: {best_val_loss:.4f}")

    return model
