import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n_points = 200
h = 2 * np.pi / n_points
delta_t = 0.001
x = np.arange(0, 2 * np.pi, h)
t_end = 2

# 初始条件
# u0 = np.sin(x)
# 间断初始条件
u0 = np.where(x < np.pi, 0, 1)


def phi_for_TVD(r):
    # return 2 * r / (1 + r ** 2)
    # return (np.abs(r)+r)/(np.abs(r)+1)
    # return np.where(r < 0, 0, np.where(r > 1, 1, r))
    return 0 # 一阶迎风
def calculate_r(u):
    u_extended = np.concatenate([u[-1:], u, u[:1]])
    r = np.zeros_like(u)
    for i in range(0, len(u) ):
        r[i] = (u_extended[i] - u_extended[i - 1]) / (u_extended[i + 1] - u_extended[i] + 1e-6)
    return r

def spatial_derivative(u):
    # 三阶迎风格式 uj-2, uj-1, uj, uj+1
    # u_extended = np.concatenate([u[-2:], u, u[:2]])
    # u_x = 1 / 6 / h * (u_extended[0:-4] - 6 * u_extended[1:-3] + 3 * u_extended[2:-2] + 2 * u_extended[3:-1])
    # 二阶迎风格式 uj-2, uj-1, uj
    u_extended = np.concatenate([u[-2:], u, u[:2]])
    u_x = 1 / 2 / h * (u_extended[0:-4] - 4*u_extended[1:-3] + 3*u_extended[2:-2])
    # 二阶中心差分 uj-1, uj+1
    # u_extended = np.concatenate([u[-1:], u, u[:1]])
    # u_x = 1 / 2 / h * (u_extended[2:] - u_extended[:-2])
    # 五阶迎风格式
    # u_extended = np.concatenate([u[-3:], u, u[:3]])
    # u_x = 1 / 60 / h * (-2 * u_extended[0:-6] + 15 * u_extended[1:-5] - 60 * u_extended[2:-4] + 20 * u_extended[3:-3] + 30 * u_extended[4:-2] - 3 * u_extended[5:-1])

    # TVD格式
    # u_extended = np.concatenate([u[-2:], u, u[:2]])
    # r_j = calculate_r(u_extended[2:-2])
    # r_j_minus_1 = calculate_r(u_extended[1:-3])
    # phi_j = phi_for_TVD(r_j)
    # phi_j_minus_1 = phi_for_TVD(r_j_minus_1)
    # u_x = 1 / h * (u_extended[2:-2] - u_extended[1:-3] + 1 / 2 * phi_j * (u_extended[3:-1] - u_extended[2:-2]) + 1 / 2 * phi_j_minus_1 * (u_extended[2:-2] - u_extended[1:-3]))

    # GVC格式
    # u_extended = np.concatenate([u[-3:], u, u[:3]])
    # abs_diff1 = np.abs(u_extended[3:-3] - u_extended[2:-4])
    # abs_diff2 = np.abs(u_extended[4:-2] - u_extended[3:-3])
    # u_x = np.where(
    #     abs_diff1 <= abs_diff2,
    #     1 / 2 / h * (u_extended[1:-5] - 4 * u_extended[2:-4] + 3 * u_extended[3:-3]),
    #     1 / 2 / h * (u_extended[4:-2] - u_extended[2:-4])
    # )

    # WENO5
    # u_extended = np.concatenate([u[-3:], u, u[:3]])
    # C = np.array([1 / 10, 6 / 10, 3 / 10]).reshape(3, 1)
    # epsilon = 1e-6
    # 
    # IS_j = np.zeros((3, len(u)))  # -6 为了确保所有操作都在数组界限内
    # 
    # IS_j[0] = 13 / 12 * (u_extended[1:-5] - 2 * u_extended[2:-4] + u_extended[3:-3]) ** 2 + 1 / 4 * (
    #             u_extended[1:-5] - 4 * u_extended[2:-4] + 3 * u_extended[3:-3]) ** 2
    # IS_j[1] = 13 / 12 * (u_extended[2:-4] - 2 * u_extended[3:-3] + u_extended[4:-2]) ** 2 + 1 / 4 * (
    #             u_extended[2:-4] - u_extended[4:-2]) ** 2
    # IS_j[2] = 13 / 12 * (u_extended[3:-3] - 2 * u_extended[4:-2] + u_extended[5:-1]) ** 2 + 1 / 4 * (
    #             u_extended[3:-3] - 4 * u_extended[4:-2] + 3 * u_extended[5:-1]) ** 2
    # beta_j = C / (epsilon + IS_j) ** 2
    # omega_j = beta_j / np.sum(beta_j, axis=0)
    # u_x_j = np.zeros((3, len(u)))
    # u_x_j[0] = 1 / 3 * u_extended[1:-5] - 7 / 6 * u_extended[2:-4] + 11 / 6 * u_extended[3:-3]
    # u_x_j[1] = -1 / 6 * u_extended[2:-4] + 5 / 6 * u_extended[3:-3] + 1 / 3 * u_extended[4:-2]
    # u_x_j[2] = 1 / 3 * u_extended[3:-3] + 5 / 6 * u_extended[4:-2] - 1 / 6 * u_extended[5:-1]
    # 
    # IS_j_minus_1 = np.zeros((3,len(u)))
    # IS_j_minus_1[0] = 13 / 12 * (u_extended[0:-6] - 2 * u_extended[1:-5] + u_extended[2:-4]) ** 2 + 1 / 4 * (u_extended[0:-6] - 4 * u_extended[1:-5] + 3 * u_extended[2:-4]) ** 2
    # IS_j_minus_1[1] = 13 / 12 * (u_extended[1:-5] - 2 * u_extended[2:-4] + u_extended[3:-3]) ** 2 + 1 / 4 * (u_extended[1:-5] - u_extended[3:-3]) ** 2
    # IS_j_minus_1[2] = 13 / 12 * (u_extended[2:-4] - 2 * u_extended[3:-3] + u_extended[4:-2]) ** 2 + 1 / 4 * (u_extended[2:-4] - 4 * u_extended[3:-3] + 3 * u_extended[4:-2]) ** 2
    # beta_j_minus_1 = C/(epsilon + IS_j_minus_1)**2
    # omega_j_minus_1 = beta_j_minus_1/np.sum(beta_j_minus_1, axis=0)
    # u_x_minus_1 = np.zeros((3,len(u)))
    # u_x_minus_1[0] = 1/3*u_extended[0:-6] -7/6 * u_extended[1:-5] +11/6* u_extended[2:-4]
    # u_x_minus_1[1] =-1/6*u_extended[1:-5] +5/6 * u_extended[2:-4] +1/3* u_extended[3:-3]
    # u_x_minus_1[2] =1/3*u_extended[2:-4] +5/6 * u_extended[3:-3] -1/6* u_extended[4:-2]
    # u_x = 1/h*(omega_j[0]*u_x_j[0] + omega_j[1]*u_x_j[1] + omega_j[2]*u_x_j[2]-omega_j_minus_1[0]*u_x_minus_1[0]-omega_j_minus_1[1]*u_x_minus_1[1]-omega_j_minus_1[2]*u_x_minus_1[2])

    return u_x


def RK_3(u, delta_t):
    k1 = -1 * spatial_derivative(u)
    u1 = u + delta_t * k1
    k2 = -1 * spatial_derivative(u1)
    u2 = 3 / 4 * u + 1 / 4 * (u1 + delta_t * k2)
    k3 = -1 * spatial_derivative(u2)
    u3 = 1 / 3 * u + 2 / 3 * (u2 + delta_t * k3)
    return u3


def Euler(u, delta_t):
    return u - delta_t * spatial_derivative(u)


def solve(u0, delta_t, t_end):
    u = u0.copy()
    t = 0
    frames = []
    l2_errors = []
    while t < t_end:
        # 计算数值解
        # u = Euler(u, delta_t)
        u = RK_3(u, delta_t)
        frames.append(u.copy())
        # 计算精确解 *****************************************************************************************************
        current_time = t + delta_t
        # exact_solution = np.sin(x - current_time)
        # exact_solution = np.sin(2 * x-2*current_time) - np.cos(x-current_time) * 1 / 3
        exact_solution = np.where(x < np.pi + current_time, 0, 1)
        # 计算L2模误差
        l2_error = np.sqrt(np.sum((u - exact_solution) ** 2) / len(x))
        l2_errors.append(l2_error)
        t += delta_t
    return frames, l2_errors


frames, l2_errors = solve(u0, delta_t, t_end)

fig, ax = plt.subplots()
line1, = ax.plot(x, frames[0], label='Numerical Solution')  # 数值解
line2, = ax.plot(x, np.sin(x), label='Exact Solution', linestyle='--')  # 精确解
ax.legend(loc='upper right')


def update(frame_number):
    frame = frames[frame_number]
    current_time = frame_number * delta_t
    # 更新精确解的图像*****************************************************************************************************
    # exact_solution = np.sin(x - current_time)
    # exact_solution = np.sin(2 * x-2*current_time) - np.cos(x-current_time) * 1 / 3
    exact_solution = np.where(x < np.pi + current_time, 0, 1)
    line1.set_ydata(frame)
    line2.set_ydata(exact_solution)
    ax.set_title(f'Time: {current_time + delta_t:.2f} seconds')

    for text in ax.texts:
        text.remove()
    # 计算并显示L2模误差
    l2_error = l2_errors[frame_number]
    ax.text(0.05, 0.9, f'L2 Error: {l2_error:.6f}', transform=ax.transAxes)

    return line1, line2


ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=10, repeat=False)
plt.show()
