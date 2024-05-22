import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sqrt
# 初始化参数
x = np.linspace(-0.5, 0.5, 1000)  # 空间范围
dx = x[1] - x[0]  # 空间步长
dt = 0.0002  # 时间步长

# 初始条件
u1, rho1, p1 = 0, 1, 1  # x < 0
u2, rho2, p2 = 0, 0.125, 0.1  # x > 0

Z_h2 = 1.7521557 # 右波波头速度
Z_t2 = 1.7521557  # 右波波尾速度
Z_h1 = -1.18321595  # 左波波头速度
Z_t1 = -0.070272  # 左波波尾速度
rrho1 = 0.426319428178517  # 激波与稀疏波之间的接触间断左侧密度
rrho2 = 0.26557371170530203  # 激波与稀疏波的接触间断右侧密度
pp = 0.30313017805066167  # 激波与稀疏波间的压力
uu = 0.9274526200489557  # 激波与稀疏波间的速度

# pp =  0.0018938734201858082
# uu =  -0.7500754172374372
# Z_h1,Z_t1,rrho1,Z_h2,Z_t2,rrho2 =  -2.748331477354788 , -1.248421978039713 , 0.010676184089240653 , 2.748331477354788 , 0.5482711435648383 , 0.0015728877166477548

gama = 1.4


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

# 绘图
fig, axes = plt.subplots(3, 1, figsize=(10, 8))


def animate(t):
    # 每次更新前清除之前的图像
    for ax in axes:
        ax.clear()  # 清除之前的绘图
        ax.grid(True)  # 重新显示网格
        ax.set_xlim([-0.5, 0.5])  # 重新设置x轴范围
        # 可以根据需要添加更多的轴设置，如轴标签等

    # 计算当前时间点的状态
    rho, v, p = compute_state(x, t)

    # 绘制新的状态
    axes[0].plot(x, rho, 'b', label='Density')
    axes[1].plot(x, v, 'r', label='Velocity')
    axes[2].plot(x, p, 'g', label='Pressure')

    # 对每个轴添加图例
    for ax in axes:
        ax.legend(loc='upper right')

    # 设置图表标题
    axes[0].set_title('Density')
    axes[1].set_title('Velocity')
    axes[2].set_title('Pressure')


# 创建动画
# ani = FuncAnimation(fig, animate, frames=np.arange(0, 10/1.7521557, dt), interval=120,repeat=0)
ani = FuncAnimation(fig, animate, frames=np.arange(0, 1, dt), interval=10,repeat=0)

plt.tight_layout()
plt.show()
