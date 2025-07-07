# -*- coding: utf-8 -*-
from packaging import version
import torch

def sinkhorn_uot(dot, max_iter=100, tau=0.1):
    """
    UOT Sinkhorn algorithm: Perform unbalanced optimal transport.
    dot: Tensor of shape (n, in_size, out_size), similarity matrix.
    max_iter: Number of Sinkhorn iterations.
    tau: Regularization parameter for mass conservation (KL divergence penalty).
    Returns:
        K: Tensor of shape (n, in_size, out_size), transport plan.
    """
    n, in_size, out_size = dot.shape
    K = dot
    # Initialize u and v
    u = K.new_ones((n, in_size))  # n x in_size
    v = K.new_ones((n, out_size))  # n x out_size

    for _ in range(max_iter):
        # Update `u` and `v` with relaxed constraints
        u = torch.exp(-tau * u) * (1.0 / (torch.bmm(K, v.view(n, out_size, 1)).view(n, in_size) + 1e-8))
        v = torch.exp(-tau * v) * (1.0 / (torch.bmm(u.view(n, 1, in_size), K).view(n, out_size) + 1e-8))

    # Compute the final transport plan
    K = u.view(n, in_size, 1) * (K * v.view(n, 1, out_size))
    return K

def UOT(q, k, eps=1.0, tau=0.1, max_iter=100, cost_type=None):
    """
    Unbalanced Optimal Transport (UOT) using Sinkhorn algorithm.
    q: Tensor of shape (n, in_size, in_dim), source features.
    k: Tensor of shape (m, out_size, in_dim), target features.
    eps: Entropic regularization parameter.
    tau: Relaxation parameter for UOT (controls mass balancing via KL divergence).
    max_iter: Number of Sinkhorn iterations.
    cost_type: Cost type ('easy' or 'hard').
    Returns:
        K: Transport plan of shape (n, out_size, m, in_size).
    """
    n, in_size, in_dim = q.shape
    m, out_size = k.shape[:-1]

    # Compute the cost matrix
    C = torch.einsum('bid,bod->bio', q, k)  # Shape: (n, m, in_size, out_size)
    if cost_type == 'easy':
        K = 1 - C.clone()
    elif cost_type == 'hard':
        K = C.clone()

    # Mask diagonal to prevent self-matching
    npatches = q.size(1)
    mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
    diagonal = torch.eye(npatches, device=q.device, dtype=mask_dtype)[None, :, :]
    K.masked_fill_(diagonal, -10)

    # Reshape and exponentiate the cost matrix
    K = K.reshape(-1, in_size, out_size)  # Shape: (n*m, in_size, out_size)
    K = torch.exp(K / eps)

    # Use the UOT sinkhorn function
    K = sinkhorn_uot(K, max_iter=max_iter, tau=tau)

    # Reshape back to the original format
    K = K.permute(0, 2, 1).contiguous()  # Final transport plan
    return K


