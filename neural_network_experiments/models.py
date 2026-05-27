from __future__ import annotations

import torch
from torch import nn


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        weight = scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, x)
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            x.shape[0],
            self.out_channels,
            x_ft.shape[-1],
            dtype=torch.cfloat,
            device=x.device,
        )
        n_modes = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :n_modes] = torch.einsum("bix,iox->box", x_ft[:, :, :n_modes], self.weight[:, :, :n_modes])
        return torch.fft.irfft(out_ft, n=x.shape[-1])


class FNO1d(nn.Module):
    def __init__(
        self,
        n_x: int,
        coef_dim: int,
        n_t: int,
        width: int = 64,
        modes: int = 16,
        depth: int = 4,
    ):
        super().__init__()
        self.n_x = n_x
        self.coef_dim = coef_dim
        self.n_t = n_t
        in_channels = 1 + coef_dim + 1
        self.lift = nn.Linear(in_channels, width)
        self.spectral_layers = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(depth)])
        self.local_layers = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.activation = nn.GELU()
        self.project = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, n_t),
        )

    def forward(self, x: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_x), coef: (batch, coef_dim)
        batch_size = x.shape[0]
        grid = torch.linspace(0, 1, self.n_x, device=x.device, dtype=x.dtype)
        grid = grid.view(1, self.n_x, 1).expand(batch_size, -1, -1)
        coef_grid = coef.unsqueeze(1).expand(-1, self.n_x, -1)
        features = torch.cat([x.unsqueeze(-1), coef_grid, grid], dim=-1)

        z = self.lift(features).permute(0, 2, 1)
        for spectral, local in zip(self.spectral_layers, self.local_layers):
            z = self.activation(spectral(z) + local(z))
        z = z.permute(0, 2, 1)
        # (batch, n_x, n_t) -> (batch, n_t, n_x)
        return self.project(z).permute(0, 2, 1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int, activation=nn.GELU):
        super().__init__()
        layers = []
        current = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(activation())
            current = hidden_dim
        layers.append(nn.Linear(current, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConcatDeepONet(nn.Module):
    def __init__(
        self,
        n_x: int,
        coef_dim: int,
        n_t: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        branch_depth: int = 3,
        trunk_depth: int = 3,
    ):
        super().__init__()
        self.n_x = n_x
        self.n_t = n_t
        self.latent_dim = latent_dim
        self.branch = MLP(n_x + coef_dim, hidden_dim, latent_dim, branch_depth)
        self.trunk = MLP(2, hidden_dim, latent_dim, trunk_depth)
        self.bias = nn.Parameter(torch.zeros(1))

    def coordinate_grid(self, device, dtype):
        t = torch.linspace(0, 1, self.n_t, device=device, dtype=dtype)
        x = torch.linspace(0, 1, self.n_x, device=device, dtype=dtype)
        tt, xx = torch.meshgrid(t, x, indexing="ij")
        return torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=-1)

    def forward(self, x: torch.Tensor, coef: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        branch_input = torch.cat([coef, x], dim=-1)
        branch_features = self.branch(branch_input)
        if coords is None:
            coords = self.coordinate_grid(x.device, x.dtype)
        trunk_features = self.trunk(coords)
        values = branch_features @ trunk_features.T + self.bias
        return values.view(x.shape[0], -1) if coords.shape[0] != self.n_t * self.n_x else values.view(x.shape[0], self.n_t, self.n_x)


class TensorizedConcatDeepONet(nn.Module):
    """DeepONet matching neural_network_experiments/main.ipynb.

    Branch input is [coefficients, initial condition]. The trunk input is (t, x).
    The output uses the notebook's tensorized contraction:

        einsum('bdt,btk,tk->bd', trunk, branch, const)
    """

    def __init__(
        self,
        n_x: int,
        coef_dim: int,
        n_t: int,
        num_trunk: int = 100,
        num_branch: int = 100,
        hidden_dim: int = 200,
        trunk_depth: int = 2,
        branch_depth: int = 2,
    ):
        super().__init__()
        self.n_x = n_x
        self.n_t = n_t
        self.num_trunk = num_trunk
        self.num_branch = num_branch
        self.trunk = MLP(2, hidden_dim, num_trunk, max(trunk_depth - 1, 0), activation=nn.ReLU)
        self.branch = MLP(n_x + coef_dim, hidden_dim, num_trunk * num_branch, max(branch_depth - 1, 0), activation=nn.ReLU)
        self.const = nn.Parameter(torch.randn(num_trunk, num_branch))
        self.activation = nn.ReLU()

    def coordinate_grid(self, device, dtype):
        t = torch.linspace(0, 1, self.n_t, device=device, dtype=dtype)
        x = torch.linspace(0, 1, self.n_x, device=device, dtype=dtype)
        tt, xx = torch.meshgrid(t, x, indexing="ij")
        return torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=-1)

    def forward(self, x: torch.Tensor, coef: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.shape[0]
        if coords is None:
            coords = self.coordinate_grid(x.device, x.dtype)
            full_grid = True
        else:
            full_grid = coords.shape[0] == self.n_t * self.n_x

        trunk_features = self.activation(self.trunk(coords)).view(1, coords.shape[0], self.num_trunk)
        trunk_features = trunk_features.expand(batch_size, -1, -1)
        branch_input = torch.cat([coef, x], dim=-1)
        branch_features = self.activation(self.branch(branch_input)).view(batch_size, self.num_trunk, self.num_branch)
        values = torch.einsum("bdt,btk,tk->bd", trunk_features, branch_features, self.const)
        return values.view(batch_size, self.n_t, self.n_x) if full_grid else values


class MNO(nn.Module):
    """MNO-style model from main.ipynb with a clean `(x, coef)` interface.

    Unlike concatenated DeepONet, coefficients get their own leaf network:

    - trunk: coordinates `(t, x)`
    - branch: initial condition `u0`
    - leaf: coefficients

    The default dimensions are intentionally smaller than the original
    `num_leaf=100, num_trunk=100, num_branch=100` notebook setting, because
    that would make the branch output enormous. With `num_leaf=2`,
    `num_trunk=100`, and `num_branch=50`, the branch output has size 10,000,
    matching the default DeepONet branch output `100 * 100`.
    """

    def __init__(
        self,
        n_x: int,
        coef_dim: int,
        n_t: int,
        num_leaf: int = 2,
        num_trunk: int = 100,
        num_branch: int = 50,
        hidden_dim: int = 200,
        trunk_depth: int = 2,
        branch_depth: int = 2,
        leaf_depth: int = 2,
    ):
        super().__init__()
        self.n_x = n_x
        self.n_t = n_t
        self.num_leaf = num_leaf
        self.num_trunk = num_trunk
        self.num_branch = num_branch
        self.trunk = MLP(2, hidden_dim, num_leaf * num_trunk, max(trunk_depth - 1, 0), activation=nn.ReLU)
        self.branch = MLP(n_x, hidden_dim, num_leaf * num_trunk * num_branch, max(branch_depth - 1, 0), activation=nn.ReLU)
        self.leaf = MLP(coef_dim, hidden_dim, num_leaf, max(leaf_depth - 1, 0), activation=nn.ReLU)
        self.const = nn.Parameter(torch.randn(num_leaf, num_trunk, num_branch))
        self.activation = nn.ReLU()

    def coordinate_grid(self, device, dtype):
        t = torch.linspace(0, 1, self.n_t, device=device, dtype=dtype)
        x = torch.linspace(0, 1, self.n_x, device=device, dtype=dtype)
        tt, xx = torch.meshgrid(t, x, indexing="ij")
        return torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=-1)

    def forward(self, x: torch.Tensor, coef: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.shape[0]
        if coords is None:
            coords = self.coordinate_grid(x.device, x.dtype)
            full_grid = True
        else:
            full_grid = coords.shape[0] == self.n_t * self.n_x

        n_points = coords.shape[0]
        c_out = self.activation(self.leaf(coef)).reshape(batch_size, self.num_leaf)
        y_out = self.activation(self.trunk(coords)).reshape(1, n_points, self.num_leaf, self.num_trunk)
        y_out = y_out.expand(batch_size, -1, -1, -1)
        u_out = self.activation(self.branch(x)).reshape(batch_size, self.num_leaf, self.num_trunk, self.num_branch)
        # Contract the branch modes (k) and fold in the leaf weights before
        # touching the points axis (d), avoiding a huge (b, d, l, t, k) tensor.
        branch_weighted = torch.einsum("bltk,ltk->blt", u_out, self.const)
        leaf_weighted = c_out.unsqueeze(-1) * branch_weighted
        values = torch.einsum("bdlt,blt->bd", y_out, leaf_weighted)
        return values.view(batch_size, self.n_t, self.n_x) if full_grid else values


class TensorizedDeepONet(nn.Module):
    """DeepONet WITHOUT coefficient input (comparison setting #1).

    Identical tensorized contraction to ``TensorizedConcatDeepONet`` except the
    branch sees only the initial condition ``u0``; the parametric function /
    coefficients are ignored. This is the standard single-input DeepONet for
    ``G: u0 -> u(t, x)`` and serves as the "no coeff" baseline.
    """

    def __init__(
        self,
        n_x: int,
        coef_dim: int,
        n_t: int,
        num_trunk: int = 100,
        num_branch: int = 100,
        hidden_dim: int = 200,
        trunk_depth: int = 2,
        branch_depth: int = 2,
    ):
        super().__init__()
        self.n_x = n_x
        self.n_t = n_t
        self.num_trunk = num_trunk
        self.num_branch = num_branch
        self.trunk = MLP(2, hidden_dim, num_trunk, max(trunk_depth - 1, 0), activation=nn.ReLU)
        self.branch = MLP(n_x, hidden_dim, num_trunk * num_branch, max(branch_depth - 1, 0), activation=nn.ReLU)
        self.const = nn.Parameter(torch.randn(num_trunk, num_branch))
        self.activation = nn.ReLU()

    def coordinate_grid(self, device, dtype):
        t = torch.linspace(0, 1, self.n_t, device=device, dtype=dtype)
        x = torch.linspace(0, 1, self.n_x, device=device, dtype=dtype)
        tt, xx = torch.meshgrid(t, x, indexing="ij")
        return torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=-1)

    def forward(self, x: torch.Tensor, coef: torch.Tensor | None = None, coords: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.shape[0]
        if coords is None:
            coords = self.coordinate_grid(x.device, x.dtype)
            full_grid = True
        else:
            full_grid = coords.shape[0] == self.n_t * self.n_x

        trunk_features = self.activation(self.trunk(coords)).view(1, coords.shape[0], self.num_trunk)
        trunk_features = trunk_features.expand(batch_size, -1, -1)
        branch_features = self.activation(self.branch(x)).view(batch_size, self.num_trunk, self.num_branch)
        # Contract the branch modes first to avoid materializing a
        # (batch, points, num_trunk, num_branch) intermediate.
        branch_weighted = torch.einsum("btk,tk->bt", branch_features, self.const)
        values = torch.einsum("bdt,bt->bd", trunk_features, branch_weighted)
        return values.view(batch_size, self.n_t, self.n_x) if full_grid else values


class MIONet(nn.Module):
    """Multiple-input operator network (Jin, Meng & Lu, 2022) — setting #3.

    Two independent branch networks encode the two input functions:

    - ``branch_u``: initial condition ``u0``,
    - ``branch_c``: parametric function / coefficients.

    Their latent codes are merged by an element-wise (Hadamard) product and
    contracted with a shared trunk over the query coordinates ``(t, x)``:

        out(b, y) = sum_p branch_u(b)_p * branch_c(b)_p * trunk(y)_p + bias
    """

    def __init__(
        self,
        n_x: int,
        coef_dim: int,
        n_t: int,
        latent: int = 128,
        hidden_dim: int = 200,
        branch_depth: int = 2,
        trunk_depth: int = 2,
    ):
        super().__init__()
        self.n_x = n_x
        self.n_t = n_t
        self.latent = latent
        self.branch_u = MLP(n_x, hidden_dim, latent, max(branch_depth - 1, 0), activation=nn.ReLU)
        self.branch_c = MLP(coef_dim, hidden_dim, latent, max(branch_depth - 1, 0), activation=nn.ReLU)
        self.trunk = MLP(2, hidden_dim, latent, max(trunk_depth - 1, 0), activation=nn.ReLU)
        self.bias = nn.Parameter(torch.zeros(1))
        self.activation = nn.ReLU()

    def coordinate_grid(self, device, dtype):
        t = torch.linspace(0, 1, self.n_t, device=device, dtype=dtype)
        x = torch.linspace(0, 1, self.n_x, device=device, dtype=dtype)
        tt, xx = torch.meshgrid(t, x, indexing="ij")
        return torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=-1)

    def forward(self, x: torch.Tensor, coef: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.shape[0]
        if coords is None:
            coords = self.coordinate_grid(x.device, x.dtype)
            full_grid = True
        else:
            full_grid = coords.shape[0] == self.n_t * self.n_x

        merged = self.branch_u(x) * self.branch_c(coef)
        trunk_features = self.activation(self.trunk(coords))
        values = merged @ trunk_features.T + self.bias
        return values.view(batch_size, self.n_t, self.n_x) if full_grid else values


class NeuralOperatorFNO2d(nn.Module):
    """Official neuraloperator FNO wrapper for [coeff, u0] -> u(t, x).

    Requires:

        pip install neuraloperator

    The 2D input grid has axes (t, x). Channels are:
    - u0(x) broadcast across time,
    - coefficients broadcast across (t, x),
    - normalized t coordinate,
    - normalized x coordinate.
    """

    def __init__(
        self,
        n_x: int,
        coef_dim: int,
        n_t: int,
        hidden_channels: int = 64,
        modes_t: int = 16,
        modes_x: int = 16,
        n_layers: int = 4,
    ):
        super().__init__()
        try:
            from neuralop.models import FNO
        except ImportError as exc:
            raise ImportError("NeuralOperatorFNO2d requires `pip install neuraloperator`.") from exc

        self.n_x = n_x
        self.n_t = n_t
        self.coef_dim = coef_dim
        self.model = FNO(
            n_modes=(modes_t, modes_x),
            hidden_channels=hidden_channels,
            in_channels=1 + coef_dim + 2,
            out_channels=1,
            n_layers=n_layers,
        )

    def forward(self, x: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        t = torch.linspace(0, 1, self.n_t, device=x.device, dtype=x.dtype)
        grid_x = torch.linspace(0, 1, self.n_x, device=x.device, dtype=x.dtype)
        tt, xx = torch.meshgrid(t, grid_x, indexing="ij")
        u0 = x[:, None, None, :].expand(-1, 1, self.n_t, self.n_x)
        coef_grid = coef[:, :, None, None].expand(-1, -1, self.n_t, self.n_x)
        t_grid = tt.view(1, 1, self.n_t, self.n_x).expand(batch_size, -1, -1, -1)
        x_grid = xx.view(1, 1, self.n_t, self.n_x).expand(batch_size, -1, -1, -1)
        features = torch.cat([u0, coef_grid, t_grid, x_grid], dim=1)
        return self.model(features).squeeze(1)
