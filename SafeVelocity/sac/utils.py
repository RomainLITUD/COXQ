
import jax
from jax import numpy as jp

def randomizer(sys, rng):
    """Randomizes gravity in the Brax system for each environment."""
    
    # Generate a random gravity offset for each environment
    @jax.vmap
    def get_random_para(rng):

        key1, key = jax.random.split(rng, 2)
        friction = jax.random.uniform(key1, minval=-0., maxval=0.)
        friction = sys.geom_friction.at[-1, 0].add(friction)

        gravity_offset = jax.random.uniform(key, minval=-9.81*1.2, maxval=-9.81*0.8)
        angle_deg = jax.random.uniform(rng, minval=-10., maxval=10.)

        # Convert angle to radians
        angle_rad = jp.deg2rad(angle_deg)

        gravity_vector = jp.zeros(3)  # Initialize a zero gravity vector
        gravity_vector = gravity_vector.at[2].set(gravity_offset*jp.cos(angle_rad))  # Set the z-axis (last dimension)
        gravity_vector = gravity_vector.at[0].set(gravity_offset*jp.sin(angle_rad))

        #vertical = sys.gravity.at[2].set(gravity_offset)
        #bias = sys.gravity.at[0].set(wind_offset)

        return gravity_vector, friction

    # Apply the random gravity to the system
    random_gravity, random_friction = get_random_para(rng)
    
    new_opt = sys.opt.tree_replace({'gravity': random_gravity,
                                    })
    # Use `tree_replace` to update the gravity in the system's configuration
    sys_v = sys.tree_replace({
        'opt': new_opt,
        'gravity': random_gravity,
        'geom_friction': random_friction
    })

    in_axes = jax.tree_map(lambda x: None, sys)

    updated_opt = in_axes.opt.tree_replace({'gravity': 0,
                                            })  # Replace only the wind attribute in opt

    # Now update the full in_axes tree with the updated opt and density
    in_axes = in_axes.tree_replace({
        'opt': updated_opt,
        'gravity': 0,
        'geom_friction': 0
    })

    
    return sys_v, in_axes


def randomizer_flat(sys, rng):
    """Randomizes gravity in the Brax system for each environment."""
    
    # Generate a random gravity offset for each environment
    @jax.vmap
    def get_random_para(rng):

        key1, key = jax.random.split(rng, 2)
        friction = jax.random.uniform(key1, minval=-0., maxval=0.)
        friction = sys.geom_friction.at[-1, 0].add(friction)

        gravity_offset = jax.random.uniform(key, minval=-9.81*1.2, maxval=-9.81*0.8)
        angle_deg = jax.random.uniform(rng, minval=0., maxval=0.)

        # Convert angle to radians
        angle_rad = jp.deg2rad(angle_deg)

        gravity_vector = jp.zeros(3)  # Initialize a zero gravity vector
        gravity_vector = gravity_vector.at[2].set(gravity_offset*jp.cos(angle_rad))  # Set the z-axis (last dimension)
        gravity_vector = gravity_vector.at[0].set(gravity_offset*jp.sin(angle_rad))

        #vertical = sys.gravity.at[2].set(gravity_offset)
        #bias = sys.gravity.at[0].set(wind_offset)

        return gravity_vector, friction

    # Apply the random gravity to the system
    random_gravity, random_friction = get_random_para(rng)
    
    new_opt = sys.opt.tree_replace({'gravity': random_gravity,
                                    })
    # Use `tree_replace` to update the gravity in the system's configuration
    sys_v = sys.tree_replace({
        'opt': new_opt,
        'gravity': random_gravity,
        'geom_friction': random_friction
    })

    in_axes = jax.tree_map(lambda x: None, sys)

    updated_opt = in_axes.opt.tree_replace({'gravity': 0,
                                            })  # Replace only the wind attribute in opt

    # Now update the full in_axes tree with the updated opt and density
    in_axes = in_axes.tree_replace({
        'opt': updated_opt,
        'gravity': 0,
        'geom_friction': 0
    })

    
    return sys_v, in_axes


def randomize_val_biped(sys, rng):
    """Randomizes gravity in the Brax system for each environment."""
    
    # Generate a random gravity offset for each environment
    @jax.vmap
    def get_random_para(rng):

        key1, key2, key3, key = jax.random.split(rng, 4)

        choice = jax.random.bernoulli(key3, 0.5)
        
        friction = jax.random.uniform(key, minval=-0., maxval=0.)
        friction = sys.geom_friction.at[-1, 0].add(friction)

        grav1 = jax.random.uniform(key1,minval=-15, maxval=-12)
        grav2 = jax.random.uniform(key2,minval=-8, maxval=-5)
        gravity_offset = jax.lax.cond(choice, lambda _: grav1, lambda _: grav2, operand=None)
        angle_deg = jax.random.uniform(rng, minval=15., maxval=20.)

        # Convert angle to radians
        angle_rad = jp.deg2rad(angle_deg)

        gravity_vector = jp.zeros(3)  # Initialize a zero gravity vector
        gravity_vector = gravity_vector.at[2].set(gravity_offset*jp.cos(angle_rad))  # Set the z-axis (last dimension)
        gravity_vector = gravity_vector.at[0].set(gravity_offset*jp.sin(angle_rad))

        #vertical = sys.gravity.at[2].set(gravity_offset)
        #bias = sys.gravity.at[0].set(wind_offset)

        return gravity_vector, friction

    # Apply the random gravity to the system
    random_gravity, random_friction = get_random_para(rng)
    
    new_opt = sys.opt.tree_replace({'gravity': random_gravity,
                                    })
    # Use `tree_replace` to update the gravity in the system's configuration
    sys_v = sys.tree_replace({
        'opt': new_opt,
        'gravity': random_gravity,
        'geom_friction': random_friction
    })

    in_axes = jax.tree_map(lambda x: None, sys)

    updated_opt = in_axes.opt.tree_replace({'gravity': 0,
                                            })  # Replace only the wind attribute in opt

    # Now update the full in_axes tree with the updated opt and density
    in_axes = in_axes.tree_replace({
        'opt': updated_opt,
        'gravity': 0,
        'geom_friction': 0
    })

    
    return sys_v, in_axes


def randomize_train_quadped(sys, rng):
    """Randomizes gravity in the Brax system for each environment."""
    
    # Generate a random gravity offset for each environment
    @jax.vmap
    def get_random_para(rng):

        key1, key, key_angle = jax.random.split(rng, 3)
        ng = sys.ngeom
        friction = jax.random.uniform(key1, minval=0.6, maxval=1.)*jp.ones(ng)
        friction = sys.geom_friction.at[:, 0].set(friction)

        gravity_offset = jax.random.uniform(key, minval=-12, maxval=-8)
        angle_deg = jax.random.uniform(key_angle, minval=0., maxval=0.)

        # Convert angle to radians
        angle_rad = jp.deg2rad(angle_deg)

        gravity_vector = jp.zeros(3)  # Initialize a zero gravity vector
        gravity_vector = gravity_vector.at[2].set(gravity_offset*jp.cos(angle_rad))  # Set the z-axis (last dimension)
        gravity_vector = gravity_vector.at[0].set(gravity_offset*jp.sin(angle_rad))

        #vertical = sys.gravity.at[2].set(gravity_offset)
        #bias = sys.gravity.at[0].set(wind_offset)

        return gravity_vector, friction

    # Apply the random gravity to the system
    random_gravity, random_friction = get_random_para(rng)
    
    new_opt = sys.opt.tree_replace({'gravity': random_gravity,
                                    })
    # Use `tree_replace` to update the gravity in the system's configuration
    sys_v = sys.tree_replace({
        'opt': new_opt,
        'gravity': random_gravity,
        'geom_friction': random_friction
    })

    in_axes = jax.tree_map(lambda x: None, sys)

    updated_opt = in_axes.opt.tree_replace({'gravity': 0,
                                            })  # Replace only the wind attribute in opt

    # Now update the full in_axes tree with the updated opt and density
    in_axes = in_axes.tree_replace({
        'opt': updated_opt,
        'gravity': 0,
        'geom_friction': 0
    })

    
    return sys_v, in_axes


def randomize_val_quadped(sys, rng):
    """Randomizes gravity in the Brax system for each environment."""
    
    # Generate a random gravity offset for each environment
    @jax.vmap
    def get_random_para(rng):

        key1, key2, key3, key4, key5, key6, key7, key, keyp = jax.random.split(rng, 9)

        choice = jax.random.bernoulli(key, 0.5)
        choice1 = jax.random.bernoulli(keyp, 0.5)
        choice2 = jax.random.bernoulli(key1, 0.5)
        
        ng = sys.ngeom
        friction1 = jax.random.uniform(key4, minval=0.4, maxval=0.6)*jp.ones(ng)
        friction2 = jax.random.uniform(key5, minval=1., maxval=1.2)*jp.ones(ng)
        friction = jax.lax.cond(choice1, lambda _: friction1, lambda _: friction2, operand=None)

        friction = sys.geom_friction.at[:, 0].set(friction)
    
        grav1 = jax.random.uniform(key2,minval=-15, maxval=-12)
        grav2 = jax.random.uniform(key3,minval=-8, maxval=-5)
        gravity_offset = jax.lax.cond(choice, lambda _: grav1, lambda _: grav2, operand=None)
        
        ang1 = jax.random.uniform(key6, minval=0, maxval=0)
        ang2 = jax.random.uniform(key7, minval=0, maxval=0)
        angle_deg = jax.lax.cond(choice2, lambda _: ang1, lambda _: ang2, operand=None)
        #angle_deg = jax.random.uniform(key1, minval=-10., maxval=10.)

        # Convert angle to radians
        angle_rad = jp.deg2rad(angle_deg)

        gravity_vector = jp.zeros(3)  # Initialize a zero gravity vector
        gravity_vector = gravity_vector.at[2].set(gravity_offset*jp.cos(angle_rad))  # Set the z-axis (last dimension)
        gravity_vector = gravity_vector.at[0].set(gravity_offset*jp.sin(angle_rad))

        #vertical = sys.gravity.at[2].set(gravity_offset)
        #bias = sys.gravity.at[0].set(wind_offset)

        return gravity_vector, friction

    # Apply the random gravity to the system
    random_gravity, random_friction = get_random_para(rng)
    
    new_opt = sys.opt.tree_replace({'gravity': random_gravity,
                                    })
    # Use `tree_replace` to update the gravity in the system's configuration
    sys_v = sys.tree_replace({
        'opt': new_opt,
        'gravity': random_gravity,
        'geom_friction': random_friction
    })

    in_axes = jax.tree_map(lambda x: None, sys)

    updated_opt = in_axes.opt.tree_replace({'gravity': 0,
                                            })  # Replace only the wind attribute in opt

    # Now update the full in_axes tree with the updated opt and density
    in_axes = in_axes.tree_replace({
        'opt': updated_opt,
        'gravity': 0,
        'geom_friction': 0
    })

    
    return sys_v, in_axes



# # pip install torch==2.3 -q
# import math, torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils import clip_grad_norm_, clip_grad_value_


# def robust_maha(y1, y2, kind="huber", delta=1.5):
#     """
#     kind="none" -> 0.5*(y1^2 + y2^2)
#     kind="huber" -> Pseudo-Huber：大残差梯度有上界，更抗离群且不再爆
#     """
#     if kind == "none":
#         return 0.5 * (y1.pow(2) + y2.pow(2))
#     # Pseudo-Huber on the L2 radius d
#     d = torch.sqrt(y1.pow(2) + y2.pow(2) + 1e-18)
#     return (delta**2) * (torch.sqrt(1.0 + (d / delta).pow(2)) - 1.0)

# # -----------------------------
# # 1) 高斯 2D 头（Cholesky：L=[[l11,0],[l21,l22]]）
# # -----------------------------
# class Gaussian2DHead(nn.Module):
#     def __init__(self, in_dim, hidden=128, rho_max=0.98):
#         super().__init__()
#         self.backbone = nn.Sequential(
#             nn.Linear(in_dim, hidden), nn.ReLU(),
#             nn.Linear(hidden, hidden), nn.ReLU(),
#         )
#         # 均值
#         self.mean_head = nn.Linear(hidden, 2)
#         # 协方差（原始输出）：对角 s11,s22；相关 u21
#         self.cov_head  = nn.Linear(hidden, 3)
#         self.rho_max = rho_max

#         # 经验上：协方差头更小 lr，更稳
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
#                 if m.bias is not None: nn.init.zeros_(m.bias)

#     def forward(self, x,
#                 sigma_min=1e-3, sigma_max=10.0,
#                 eps=1e-6, warmup_rho=1.0):
#         """
#         warmup_rho ∈ (0,1]：训练前期逐步放开相关强度（如线性从0->1）
#         """
#         h = self.backbone(x)
#         mu = self.mean_head(h)                         # (...,2)
#         s11, s22, u21 = torch.split(self.cov_head(h), [1,1,1], dim=-1)
#         # 对角：正且夹紧（避免 0 和过大导致爆/消）
#         L11 = (F.softplus(s11) + eps).clamp(sigma_min, sigma_max)  # (...,1)
#         L22 = (F.softplus(s22) + eps).clamp(sigma_min, sigma_max)  # (...,1)
#         # 相关限幅 + warmup（避免一开始把相关拉满）
#         rho = torch.tanh(u21) * (self.rho_max * warmup_rho)        # (...,1)
#         # 在 Cholesky 域使用稳定构造：让 |L21| 与尺度相称
#         L21 = rho * L22                                            # (...,1)

#         return mu, L11, L22, L21

#     @staticmethod
#     def nll_from_L(x, mu, L11, L22, L21, robust="huber", huber_delta=1.5):
#         """
#         返回：mean NLL 以及中间监控指标（便于调参与观测）
#         NLL = robust_maha(y)/mean + logdiag(L)（省略常数 2log(2π)）
#         """
#         r1 = x[..., 0:1] - mu[..., 0:1]
#         r2 = x[..., 1:2] - mu[..., 1:2]

#         # y = L^{-1} r，通过三角“前代”求解，避免显式求逆
#         y1 = r1 / L11
#         y2 = (r2 - L21 * y1) / L22

#         maha = robust_maha(y1, y2, kind=robust, delta=huber_delta)  # (...,1)
#         logdet = torch.log(L11) + torch.log(L22)

#         # 训练用稳健项；评估请把 robust="none"
#         nll = maha.mean() + logdet.mean()

#         with torch.no_grad():
#             stats = dict(
#                 max_abs_y1=float(y1.abs().max()),
#                 max_abs_y2=float(y2.abs().max()),
#                 min_Ldiag=float(torch.minimum(L11, L22).min()),
#                 max_Ldiag=float(torch.maximum(L11, L22).max()),
#                 rho_abs_max=float((L21 / (L22 + 1e-12)).abs().max()),
#             )
#         return nll, stats

# # -----------------------------
# # 2) 一个稳健训练循环（含：warmup、分头裁剪、全局兜底）
# # -----------------------------
# def train_loop(model, data_loader, steps=1000, device="cuda" if torch.cuda.is_available() else "cpu"):
#     model.to(device)
#     # 分组学习率：协方差头更小
#     opt = torch.optim.Adam([
#         {"params": model.mean_head.parameters(), "lr": 3e-4, "weight_decay": 1e-4},
#         {"params": model.cov_head.parameters(),  "lr": 1e-4, "weight_decay": 1e-4},
#         {"params": model.backbone.parameters(),  "lr": 3e-4, "weight_decay": 1e-4},
#     ])

#     # 可根据 batch 的覆盖程度调整这几个阈值
#     MAX_NORM_MEAN = 5.0
#     MAX_NORM_COV  = 1.0
#     MAX_NORM_ALL  = 5.0

#     model.train()
#     it = iter(data_loader)
#     for step in range(1, steps + 1):
#         try:
#             x, feats = next(it)                     # x: (...,2), feats: (..., in_dim)
#         except StopIteration:
#             it = iter(data_loader)
#             x, feats = next(it)
#         x, feats = x.to(device), feats.to(device)

#         # 相关 warm-up（前 10% 步骤线性放开）
#         warmup_rho = min(1.0, step / max(1, int(0.1 * steps)))

#         # 前向：稳健 NLL（训练期：robust="huber"；评估时用 "none"）
#         mu, L11, L22, L21 = model(feats, warmup_rho=warmup_rho)
#         loss, stats = model.nll_from_L(
#             x, mu, L11, L22, L21,
#             robust="huber", huber_delta=1.5
#         )

#         opt.zero_grad(set_to_none=True)
#         loss.backward()

#         # ---- 分头裁剪（更精准）：协方差头更紧 ----
#         clip_grad_norm_(model.mean_head.parameters(), MAX_NORM_MEAN)
#         clip_grad_norm_(model.cov_head.parameters(),  MAX_NORM_COV)
#         # （可选）按值裁剪，抑制极端 outlier 的单点梯度
#         # clip_grad_value_(model.cov_head.parameters(), 1.0)

#         # ---- 全局兜底 ----
#         clip_grad_norm_(model.parameters(), MAX_NORM_ALL)

#         opt.step()

#         if step % 100 == 0 or step == 1:
#             print(f"[{step:04d}] loss={float(loss):.4f}  "
#                   f"y|max=({stats['max_abs_y1']:.2f},{stats['max_abs_y2']:.2f})  "
#                   f"Ldiag[min,max]=({stats['min_Ldiag']:.3e},{stats['max_Ldiag']:.3e})  "
#                   f"|rho|max={stats['rho_abs_max']:.3f}")

# # -----------------------------
# # 3) 一个最小可运行 DataLoader（示例数据）
# # -----------------------------
# class Toy2DGaus(torch.utils.data.Dataset):
#     """
#     生成一个简单的二维条件高斯回归数据：x ~ N(mu(feats), Σ(feats))
#     这里只是示例。实际用你的数据替换掉即可。
#     """
#     def __init__(self, n=4096, in_dim=16, seed=0):
#         g = torch.Generator().manual_seed(seed)
#         self.feats = torch.rand(n, in_dim, generator=g)*2-1
#         # 真实均值 = A * feats
#         A = torch.randn(in_dim, 2, generator=g) * 0.5
#         mu = self.feats @ A
#         # 真实协方差：给定一个温和相关与方差
#         l11 = torch.full((n,1), 0.8)
#         l22 = torch.full((n,1), 1.2)
#         rho = torch.full((n,1), 0.3)
#         l21 = rho * l22
#         # 采样：先采 z ~ N(0,I)，再 x = mu + L z
#         z = torch.randn(n, 2, generator=g)
#         x = torch.empty_like(mu)
#         x[:,0] = mu[:,0] + l11[:,0] * z[:,0]
#         x[:,1] = mu[:,1] + l21[:,0] * z[:,0] + l22[:,0] * z[:,1]
#         self.x = x
#     def __len__(self): return self.x.size(0)
#     def __getitem__(self, i): return self.x[i], self.feats[i]

# def get_loader(bs=256):
#     ds = Toy2DGaus()
#     return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

# # -----------------------------
# # 4) 运行（示例）
# # -----------------------------
# if __name__ == "__main__":
#     torch.set_float32_matmul_precision("high")
#     model = Gaussian2DHead(in_dim=16, hidden=128, rho_max=0.98)
#     loader = get_loader(bs=256)
#     train_loop(model, loader, steps=1000)


# import torch, torch.nn.functional as F

# def constrain_L_from_kmax(s11_raw, s22_raw, u21_raw,
#                           k_max=100.0, # 条件数上限
#                           sigma_min=1e-3, sigma_max=5.0,
#                           eps=1e-6, tau=0.75, margin=0.95):
#     L11 = (F.softplus(s11_raw) + eps).clamp(sigma_min, sigma_max)
#     L22 = (F.softplus(s22_raw) + eps).clamp(sigma_min, sigma_max)

#     # ---- 第一次估计：用自由的 L21_free 计算 r -> gamma(r) ----
#     L21_free = u21_raw                           # 也可先缩放，如 * L22
#     sigma2_free = torch.sqrt(L21_free**2 + L22**2 + 1e-12)
#     r = (sigma2_free / L11).clamp(min=1e-6)

#     rho_max = (k_max / (r**2) - 1.0) / (k_max / (r**2) + 1.0)
#     rho_max = rho_max.clamp(min=0.0, max=0.9)
#     gamma = rho_max / torch.sqrt(1 - rho_max**2 + 1e-12)   # L21/L22 的比例上限

#     # 可微参数化：|L21| <= margin*gamma*L22
#     L21 = (margin * gamma) * L22 * torch.tanh(L21_free / tau)
#     return L11, L22, L21
