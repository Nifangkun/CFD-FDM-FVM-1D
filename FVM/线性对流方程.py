import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

n_cells = 50
L = 1
dx = L / n_cells

delta_t = 0.001
t_end = 2
n_steps = int(t_end / delta_t)

u = np.zeros(n_cells)
# 设定初始条件
x = np.arange(dx / 2, L + dx / 2, dx)  # 居中点
u = np.where(x < 0.5, 0, 1)
# u = np.sin(2*np.pi*x)

# 零阶重构
# def get_flux(u):
#     flux = np.zeros(n_cells + 1)
#     for i in range(n_cells + 1):
#         # 线性对流方程
#         flux[i] = u[i - 1]  # flux定义在单元左侧边界
#     return flux

# 线性重构
def get_flux(u):
    flux = np.zeros(n_cells + 1)
    # 计算梯度
    gradients = np.zeros(n_cells)
    gradients[0] = (u[0] - u[-1]) / dx  # 对于第一个单元
    gradients[-1] = (u[-1] - u[-2]) / dx  # 对于最后一个单元
    for i in range(1, n_cells - 1):
        gradients[i] = (u[i] - u[i - 1]) / dx

    # 线性重构至单元界面
    flux[0] = 0.5 * (u[-1] + u[0])  # 对于最左侧边界
    flux[-1] = flux[0]  # 最右侧边界与最左侧边界相同
    for i in range(1, n_cells):
        u_left = u[i - 1] + gradients[i - 1] * dx / 2  # 左侧单元的右界面
        u_right = u[i] - gradients[i] * dx / 2  # 右侧单元的左界面
        flux[i] = 0.5 * u_left + 0.5 * u_right   # 取平均作为界面通量
    return flux

def solve(u, delta_t, dx):
    u_new = np.zeros(n_cells)
    flux = get_flux(u)
    for i in range(n_cells):
        u_new[i] = u[i] - delta_t / dx * (flux[i + 1] - flux[i])  # 此处flux[i+1]相当于flux[i+1/2]
    u = u_new
    return u


def update(frame):
    global u
    u = solve(u, delta_t, dx)
    line.set_ydata(u)
    # 画精确解
    u_exact = np.where(x < 0.5 + frame * delta_t, 0, 1)
    line_exact.set_ydata(u_exact)
    return line, line_exact

# 初始化图像
fig, ax = plt.subplots()
line, = ax.plot(x, u, label='Numerical Solution')
line_exact, = ax.plot(x, np.zeros_like(x), label='Exact Solution')
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('Finite Volume Method')
ax.legend()

# 设定坐标轴范围
ax.set_xlim(0, L)
ax.set_ylim(-0.05, 1.05)

# 创建动画
ani = FuncAnimation(fig, update, frames=n_steps, blit=True, interval=50)

# 显示动画
plt.show()
