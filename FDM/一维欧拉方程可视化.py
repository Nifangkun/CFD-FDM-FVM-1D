import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sqrt

from 一维欧拉方程_SW分裂 import Euler


# 初始化参数
x = np.linspace(-0.5, 0.5, 1000)
dx = x[1] - x[0]
dt = 0.0002

# 初始条件（精确解和数值解共用）
u1, rho1, p1 = 0, 1, 1  # x < 0
u2, rho2, p2 = 0, 0.125, 0.1  # x > 0

# 精确解中用到的额外参数
Z_h2 = 1.7521557
Z_t2 = 1.7521557
Z_h1 = -1.18321595
Z_t1 = -0.070272
rrho1 = 0.426319428178517
rrho2 = 0.26557371170530203
pp = 0.30313017805066167
uu = 0.9274526200489557
gama = 1.4

# 数值解初始条件
rho = np.where(x < 0, rho1, rho2)
u = np.zeros_like(x)
p = np.where(x < 0, p1, p2)
rho_u = rho * u
e = p / (gama - 1) + 0.5 * rho * u**2

def compute_numerical_state(rho, rho_u, u, e, p):
    return Euler(rho, rho_u, u, e, p)

# 计算密度、速度、压力随时间和空间的变化
def compute_state(x, t):
    t += 1e-5  # 避免除以零
    rho = np.where(x < 0, rho1, rho2)
    v = np.zeros_like(x)
    p = np.where(x < 0, p1, p2)

    if Z_h1 == Z_t1:
        if Z_h2 == Z_t2: # 左右都是激波

            cond = (x >= Z_h1 * t) & (x <= Z_h2 * t)
            rho[cond] = np.where(x[cond] < uu * t, rrho1, rrho2)
            p[cond] = pp
            v[cond] = uu

            return rho,v,p
        else:# 左激波右稀疏波
            c2 =  sqrt(gama*p1/rho1)
            cc2 = (gama - 1) / (gama + 1) * (-1 * x / t) + 2 / (gama + 1) * c2

            cond = (x >= Z_h1 * t) & (x <= Z_h2 * t)
            # 使用np.where进行条件赋值
            rho[cond] = np.where(x[cond] < uu * t, rrho1, rrho2)
            p[cond] = pp
            v[cond] = uu

            cond_sparse = (x >= Z_t2 * t) & (x <= Z_h2 * t)
            v[cond_sparse] = x[cond_sparse] / t + cc2[cond_sparse]
            p[cond_sparse] = p1 * (cc2[cond_sparse] / c2) ** (2 * gama / (gama - 1))
            rho[cond_sparse] = gama * p[cond_sparse] / cc2[cond_sparse] ** 2

            return rho,v,p
    else:
        c1 = sqrt(gama * p1 / rho1)
        cc1 = (gama - 1) / (gama + 1) * (-1 * x / t) + 2 / (gama + 1) * c1

        if Z_h2 == Z_t2: # 左稀疏波右激波

            # 稀疏波和激波之间的状态
            cond = (x >= Z_h1 * t) & (x <= Z_h2 * t)
            # 使用np.where进行条件赋值
            rho[cond] = np.where(x[cond] < uu * t, rrho1, rrho2)
            p[cond] = pp
            v[cond] = uu

        # 稀疏波内的状态
            cond_sparse = (x <= Z_t1 * t) & (x >= Z_h1 * t)
            v[cond_sparse] = x[cond_sparse] / t + cc1[cond_sparse]
            p[cond_sparse] = p1 * (cc1[cond_sparse] / c1) ** (2 * gama / (gama - 1))
            rho[cond_sparse] = gama * p[cond_sparse] / cc1[cond_sparse] ** 2

            return rho, v, p
        else: # 左右都是稀疏波
            c2 = sqrt(gama * p1 / rho1)
            cc2 = (gama - 1) / (gama + 1) * (-1 * x / t) + 2 / (gama + 1) * c2

            cond = (x >= Z_t1 * t) & (x <= Z_t2 * t)
            # 使用np.where进行条件赋值
            rho[cond] = np.where(x[cond] < uu * t, rrho1, rrho2)
            p[cond] = pp
            v[cond] = uu

            cond_sparse1 = (x <= Z_t1 * t) & (x >= Z_h1 * t)
            v[cond_sparse1] = x[cond_sparse1] / t + cc1[cond_sparse1]
            p[cond_sparse1] = p1 * (cc1[cond_sparse1] / c1) ** (2 * gama / (gama - 1))
            rho[cond_sparse1] = gama * p[cond_sparse1] / cc1[cond_sparse1] ** 2

            cond_sparse2 = (x >= Z_t2 * t) & (x <= Z_h2 * t)
            v[cond_sparse2] = x[cond_sparse2] / t + cc2[cond_sparse2]
            p[cond_sparse2] = p1 * (cc2[cond_sparse2] / c2) ** (2 * gama / (gama - 1))
            rho[cond_sparse2] = gama * p[cond_sparse2] / cc2[cond_sparse2] ** 2

            return rho,v,p

# 绘图设置
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

def animate(t):
    global rho, rho_u, u, e, p

    for ax in axes:
        ax.clear()
        ax.grid(True)
        ax.set_xlim([-0.5, 0.5])
    ax.set_title(f'Riemann Problem Solution t={t+dt:.4f}s')
    # 精确解
    rho_exact, v_exact, p_exact = compute_state(x, t)

    # 数值解
    rho, u, p = compute_numerical_state(rho, rho_u, u, e, p)
    rho_u = rho * u
    e = p / (gama - 1) + 0.5 * rho * u**2

    # 绘制精确解
    axes[0].plot(x, rho_exact, 'grey', label='Density (Exact)')
    axes[1].plot(x, v_exact, 'grey', label='Velocity (Exact)')
    axes[2].plot(x, p_exact, 'grey', label='Pressure (Exact)')

    # 绘制数值解
    axes[0].plot(x, rho, 'b--', label='Density (Numerical)')
    axes[1].plot(x, u, 'r--', label='Velocity (Numerical)')
    axes[2].plot(x, p, 'g--', label='Pressure (Numerical)')


    for ax in axes:
        ax.legend(loc='upper right')

ani = FuncAnimation(fig, animate, frames=np.arange(0, 0.14,dt), interval=1, repeat=False)

plt.tight_layout()
plt.show()
