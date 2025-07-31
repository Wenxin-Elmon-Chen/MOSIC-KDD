import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class PolynomialDecayLR(_LRScheduler):
    def __init__(self, optimizer, initial_lr=0.1, p=0.1, last_epoch=-1):
        """
        Polynomial Growth Scheduler

        Args:
        - optimizer (torch.optim.Optimizer): Wrapped optimizer.
        - initial_lr (float): Initial learning rate multiplier.
        - p (float): Exponent for polynomial growth.
        - last_epoch (int): The index of the last epoch. Default: -1.
        """
        self.initial_lr = initial_lr
        self.p = p
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.initial_lr / ((1 + self.last_epoch) ** self.p) for _ in self.base_lrs]

def cal_weights_align_torch(golds_treatment, logits_treatment, normalized, stabilized=True, clip=True):
    if not isinstance(golds_treatment, torch.Tensor):
        golds_treatment = torch.tensor(golds_treatment, dtype=torch.float32)
    if not isinstance(logits_treatment, torch.Tensor):
        logits_treatment = torch.tensor(logits_treatment, dtype=torch.float32)
    if normalized:
        ps = logits_treatment
    else:
        ps = torch.sigmoid(logits_treatment) if len(logits_treatment.shape) == 1 else torch.softmax(logits_treatment, dim=1)[:, 1]
    ones_idx = torch.where(golds_treatment == 1)[0]
    zeros_idx = torch.where(golds_treatment == 0)[0]
    p_T = len(ones_idx) / len(golds_treatment)
    if stabilized:
        treated_w = p_T / ps[ones_idx]
        controlled_w = (1 - p_T) / (1 - ps[zeros_idx])
    else:
        treated_w = 1.0 / ps[ones_idx]
        controlled_w = 1.0 / (1 - ps[zeros_idx])
    treated_w = torch.where(torch.isinf(treated_w), torch.tensor(0.0, device=treated_w.device), treated_w)
    controlled_w = torch.where(torch.isinf(controlled_w), torch.tensor(0.0, device=controlled_w.device), controlled_w)
    if clip:
        all_weights = torch.cat([treated_w, controlled_w])
        amin = torch.quantile(all_weights, 0.01)
        amax = torch.quantile(all_weights, 0.99)
        if amax > 50:
            amax = torch.quantile(all_weights, 0.8)
        if amin <= 1e-6:
            amin = torch.quantile(all_weights, 0.2)
        treated_w = torch.clamp(treated_w, min=amin, max=amax)
        controlled_w = torch.clamp(controlled_w, min=amin, max=amax)
    treated_w = treated_w.unsqueeze(1)
    controlled_w = controlled_w.unsqueeze(1)
    all_w = torch.zeros(len(golds_treatment), 1, dtype=torch.float32, device=golds_treatment.device)
    for arr_idx, all_idx in enumerate(ones_idx):
        all_w[all_idx, 0] = treated_w[arr_idx, 0]
    for arr_idx, all_idx in enumerate(zeros_idx):
        all_w[all_idx, 0] = controlled_w[arr_idx, 0]
    return treated_w, controlled_w, all_w

class MOSIC:
    def __init__(self, 
                 identifier, 
                 l1=0.01, 
                 expect_group_size=0.5,
                 alpha=0.01,
                 ps_stablized=False,
                 ps_truncation=False,
                 identifier_lr=0.01, 
                 lambda_lr=0.01, 
                 beta=1e-5, 
                 verbose=False, 
                 device='cpu',
                 lr_decay_rate = 0.1,
                 logger=None,
                 convergence_tol=1e-5,
                 min_epochs=500,
                 ma_window=10,
                 patience=10):
        self.device = torch.device(device)
        self.identifier = identifier.to(self.device)
        self.l1 = l1
        self.identifier_lr = identifier_lr
        self.lambda_lr = lambda_lr
        # Group size and overlap constraints are always included by default
        self.expect_group_size = expect_group_size
        self.alpha = alpha
        self.ps_stablized = ps_stablized
        self.ps_truncation = ps_truncation
        self.beta = beta
        self.verbose = verbose
        self.lr_decay_rate = lr_decay_rate
        self.lambda_params = None
        self._lambda_initialized = False
        self.logger = logger
        # Convergence criteria parameters
        self.convergence_tol = convergence_tol
        self.min_epochs = min_epochs
        self.ma_window = ma_window
        self.patience = patience
        self.fit_counter = 0

    def evaluate_constraint(self, S, constraints_coef, constraint_a, constraint_normalize=None):
        denominator = torch.where(constraint_normalize, S.detach().sum(), 1.) # block the gradient flow in the denominator
        constraint_value = constraint_a + torch.sum(constraints_coef * S, dim=0)/denominator
        return constraint_value

    def compute_main_objective(self, S, A, Y, pred_Y0, pred_Y1, ipw_weights, ate_method="aiptw"):
        SW = S * ipw_weights
        if ate_method == "aiptw":
            item1 = torch.sum(S * (pred_Y1 - pred_Y0).view(-1, 1))
            item2 = torch.sum(SW * A.view(-1, 1) * (Y - pred_Y1).view(-1, 1))
            item3 = torch.sum(SW * (1 - A).view(-1, 1) * (Y - pred_Y0).view(-1, 1))
            loss_ate = -((item1 + item2 - item3) / torch.sum(S))
        elif ate_method == "iptw":
            avg1 = torch.sum(SW * A * Y) / torch.sum(SW * A)
            avg0 = torch.sum(SW * (1 - A) * Y) / torch.sum(SW * (1 - A))
            loss_ate = -(avg1 - avg0)
        else:
            raise ValueError(f"Unknown ATE method: {ate_method}")
        return loss_ate

    def fit(self, X, A, Y, pred_Y0, pred_Y1, PS, constraint_a=None, constraint_coeffs=None, constraint_normalize=None, epochs=500,
            val_X=None, val_A=None, val_Y=None, val_pred_Y0=None, val_pred_Y1=None, val_PS=None,
            val_constraint_a=None, val_constraint_coeffs=None, val_constraint_normalize=None):
        self.fit_counter += 1
        # re-initialize identifier_params
        for param in self.identifier.parameters():
            param.data = torch.randn_like(param.data) * 0.01

        # Move all data to device and convert to tensor if needed
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        A = torch.as_tensor(A, dtype=torch.float32, device=self.device)
        Y = torch.as_tensor(Y, dtype=torch.float32, device=self.device)
        pred_Y0 = torch.as_tensor(pred_Y0, dtype=torch.float32, device=self.device)
        pred_Y1 = torch.as_tensor(pred_Y1, dtype=torch.float32, device=self.device)
        PS = torch.as_tensor(PS, dtype=torch.float32, device=self.device)
        if constraint_a is not None:
            constraint_a = torch.as_tensor(constraint_a, dtype=torch.float32, device=self.device)
        if constraint_coeffs is not None:
            constraint_coeffs = torch.as_tensor(constraint_coeffs, dtype=torch.float32, device=self.device)
        if constraint_normalize is not None:
            constraint_normalize = torch.as_tensor(constraint_normalize, dtype=torch.bool, device=self.device)
        if self.alpha != 0:
            h = 1 - (PS * (1 - PS)) / (self.alpha * (1 - self.alpha)) 
        else:
            h = torch.zeros_like(PS)
        _, _, ipw_weights = cal_weights_align_torch(A, PS, normalized=True, stabilized=self.ps_stablized, clip=self.ps_truncation)

        if val_X is not None:
            val_X = torch.as_tensor(val_X, dtype=torch.float32, device=self.device)
            val_A = torch.as_tensor(val_A, dtype=torch.float32, device=self.device)
            val_Y = torch.as_tensor(val_Y, dtype=torch.float32, device=self.device)
            val_pred_Y0 = torch.as_tensor(val_pred_Y0, dtype=torch.float32, device=self.device)
            val_pred_Y1 = torch.as_tensor(val_pred_Y1, dtype=torch.float32, device=self.device)
            val_PS = torch.as_tensor(val_PS, dtype=torch.float32, device=self.device)
            if val_constraint_a is not None and not isinstance(val_constraint_a, torch.Tensor):
                val_constraint_a = torch.as_tensor(val_constraint_a, dtype=torch.float32, device=self.device)
            if val_constraint_coeffs is not None and not isinstance(val_constraint_coeffs, torch.Tensor):
                val_constraint_coeffs = torch.as_tensor(val_constraint_coeffs, dtype=torch.float32, device=self.device)
            if val_constraint_normalize is not None and not isinstance(val_constraint_normalize, torch.Tensor):
                val_constraint_normalize = torch.as_tensor(val_constraint_normalize, dtype=torch.bool, device=self.device)
            if self.alpha != 0:
                val_h = 1 - (val_PS * (1 - val_PS)) / (self.alpha * (1 - self.alpha)) 
            else:
                val_h = torch.zeros_like(val_PS)
            _, _, ipw_weights_val = cal_weights_align_torch(val_A, val_PS, normalized=True, stabilized=self.ps_stablized, clip=self.ps_truncation)

        # Configure optimizers and scheduler at the beginning
        train_n = len(X)
        num_extra_constraints = 0 if constraint_a is None else constraint_a.shape[0]
        total_constraints = 1 + train_n + num_extra_constraints
        lambda_params = nn.Parameter(torch.zeros(total_constraints, device=self.device), requires_grad=True)
        lambda_params.data[0] = 0.5
        self.lambda_params = lambda_params

        identifier_optimizer = torch.optim.Adam(self.identifier.parameters(), lr=self.identifier_lr)
        lambda_optimizer = torch.optim.Adam([lambda_params], lr=self.lambda_lr)
        identifier_scheduler = PolynomialDecayLR(identifier_optimizer, initial_lr=self.identifier_lr, p=self.lr_decay_rate)

        # Convergence tracking
        loss_history = []
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            # Step 1: identifier update
            identifier_optimizer.zero_grad()
            S = self.identifier(X)
            group_size_constraint = self.expect_group_size - S.mean()
            overlap_constraint = S.flatten() * h
            constraint_term = lambda_params[0] * torch.relu(group_size_constraint)
            constraint_term += torch.sum(lambda_params[1:1+len(overlap_constraint)] * torch.relu(overlap_constraint))
            if constraint_a is not None:
                extra_constraints = self.evaluate_constraint(S, constraint_coeffs, constraint_a, constraint_normalize)
                constraint_term += torch.sum(lambda_params[1+len(overlap_constraint):] * torch.relu(extra_constraints))
            l1_penalty = sum(torch.mean(torch.abs(p)) for p in self.identifier.parameters())
            main_loss = self.compute_main_objective(S, A, Y, pred_Y0, pred_Y1, ipw_weights)
            total_loss = main_loss + constraint_term - torch.sum(0.5 * self.beta * lambda_params ** 2) + self.l1 * l1_penalty
            total_loss.backward()
            identifier_optimizer.step()
            identifier_scheduler.step()
            # Step 2: lambda update
            lambda_optimizer.zero_grad()
            with torch.no_grad():
                S = self.identifier(X)
            group_size_violation = self.expect_group_size - S.mean()
            overlap_violation = S.flatten() * h
            violation_term = lambda_params[0] * torch.relu(group_size_violation)
            violation_term += torch.sum(lambda_params[1:1+len(overlap_violation)] * torch.relu(overlap_violation))
            if constraint_a is not None:
                extra_violations = self.evaluate_constraint(S, constraint_coeffs, constraint_a, constraint_normalize)
                violation_term += torch.sum(lambda_params[1+len(overlap_violation):] * torch.relu(extra_violations))
            total_violation = violation_term - torch.sum(0.5 * self.beta * lambda_params ** 2)
            (-total_violation).backward()
            lambda_optimizer.step()
            with torch.no_grad():
                lambda_params.data = torch.clamp(lambda_params.data, min=0)
            
            # Store loss for convergence checking
            current_loss = total_loss.item()
            loss_history.append(current_loss)
            
            # Patience-based moving average convergence
            if epoch >= self.min_epochs and len(loss_history) > self.ma_window:
                recent_losses = loss_history[-self.ma_window:]
                mean_abs_change = float(torch.mean(torch.abs(torch.tensor(recent_losses[1:]) - torch.tensor(recent_losses[:-1]))))
                if mean_abs_change < self.convergence_tol:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Convergence reached at epoch {epoch}: moving average loss change={mean_abs_change:.2e} < {self.convergence_tol} for {self.patience} consecutive checks")
                        break
                else:
                    patience_counter = 0
            
            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={total_loss.item():.4f}, main_loss={main_loss.item():.4f}, group_size={S.mean().item():.4f}, violation_term={violation_term.item():.4f}, violation_and_beta={total_violation.item():.4f}")

            if self.logger is not None:
                self.logger.add_scalar('train/total_loss', total_loss.item(), epoch)
                self.logger.add_scalar('train/main_loss', main_loss.item(), epoch)
                self.logger.add_scalar('train/group_size', S.mean().item(), epoch)
                self.logger.add_scalar('train/constraint_violation', violation_term.item(), epoch)
                self.logger.add_scalar('train/identifier_lr', identifier_scheduler.get_last_lr()[0], epoch)

            # Validation
            if val_X is not None:
                S_val = self.identifier(val_X)
                main_loss_val = self.compute_main_objective(S_val, val_A, val_Y, val_pred_Y0, val_pred_Y1, ipw_weights_val)
                group_size_violation_val = self.expect_group_size - S_val.mean()
                overlap_violation_val = S_val.flatten() * val_h
                total_violation_val = torch.relu(group_size_violation_val) + torch.sum(torch.relu(overlap_violation_val))
                if val_constraint_a is not None:
                    extra_violations_val = self.evaluate_constraint(S_val, val_constraint_coeffs, val_constraint_a, val_constraint_normalize)
                    total_violation_val += torch.sum(torch.relu(extra_violations_val))
                
                if self.verbose and epoch % 100 == 0:
                    print(f"Epoch {epoch}: val_main_loss={main_loss_val.item():.4f}, val_group_size={S_val.mean().item():.4f}, val_violation_term={total_violation_val.item():.4f}")

                if self.logger is not None:
                    self.logger.add_scalar('val/total_violation', total_violation_val.item(), epoch)
                    self.logger.add_scalar('val/main_loss', main_loss_val.item(), epoch)
                    self.logger.add_scalar('val/group_size', S_val.mean().item(), epoch)

        # Training completed
        if self.verbose:
            print(f"Training completed after {epoch + 1} epochs")

        # if group size is too small, reset the identifier and restart the training
        if S.mean() < 0.05:
            if self.fit_counter > 3:
                print(f"Group size collapsed, but the identifier has been reset more than 3 times, stopping the training")
                return
            print(f"Group size collapsed, resetting the identifier and restarting the training")
            self.fit(X, A, Y, pred_Y0, pred_Y1, PS, constraint_a, constraint_coeffs, constraint_normalize, epochs=epochs,
                    val_X=val_X, val_A=val_A, val_Y=val_Y, val_pred_Y0=val_pred_Y0, val_pred_Y1=val_pred_Y1, val_PS=val_PS,
                    val_constraint_a=val_constraint_a, val_constraint_coeffs=val_constraint_coeffs, val_constraint_normalize=val_constraint_normalize)

        # self.identifier = self.identifier.cpu()
        # self.lambda_params = self.lambda_params.cpu()

    def predict(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        # Ensure identifier is on the same device as input
        self.identifier = self.identifier.to(self.device)
        self.identifier.eval()
        with torch.no_grad():
            S = self.identifier(X)
        return S