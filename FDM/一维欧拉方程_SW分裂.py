import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 参数设置
delta_x = 0.001
delta_t = 0.0002
t_end = 0.14
gama = 1.4
time_steps = int(t_end / delta_t)
# 初始条件

lenght_l = 0.5
lenght_r = 0.5
n_l = int(lenght_l / delta_x)
n_r = int(lenght_r / delta_x)
p = np.concatenate([np.ones(n_l), 0.1 * np.ones(n_r)])
rho = np.concatenate([np.ones(n_l), 0.125 * np.ones(n_r)])
u = np.zeros(n_l + n_r)
rho_u = rho * u
e = p / (gama - 1) + 0.5 * rho * u ** 2


def Steger_Warming(p, rho, u):
    c = np.sqrt(gama * p / rho)
    lamda1 = u
    lamda2 = u + c
    lamda3 = u - c

    F_plus = np.zeros((3, len(rho)))
    F_minus = np.zeros((3, len(rho)))

    F_plus[0, :] = rho / (2 * gama) * (
            2 * (gama - 1) * np.maximum(lamda1, 0) + np.maximum(lamda2, 0) + np.maximum(lamda3, 0))
    F_plus[1, :] = rho / (2 * gama) * (
            2 * (gama - 1) * np.maximum(lamda1, 0) * u + np.maximum(lamda2, 0) * (u + c) + np.maximum(lamda3, 0) * (
            u - c))
    F_plus[2, :] = rho / (2 * gama) * ((gama - 1) * np.maximum(lamda1, 0) * u ** 2 + (3 - gama) / (2 * (gama - 1)) * (
            np.maximum(lamda2, 0) + np.maximum(lamda3, 0)) * c ** 2 + 0.5 * np.maximum(lamda2, 0) * (
                                               u + c) ** 2 + 0.5 * np.maximum(lamda3, 0) * (u - c) ** 2)

    F_minus[0, :] = rho / (2 * gama) * (
            2 * (gama - 1) * np.minimum(lamda1, 0) + np.minimum(lamda2, 0) + np.minimum(lamda3, 0))
    F_minus[1, :] = rho / (2 * gama) * (
            2 * (gama - 1) * np.minimum(lamda1, 0) * u + np.minimum(lamda2, 0) * (u + c) + np.minimum(lamda3, 0) * (
            u - c))
    F_minus[2, :] = rho / (2 * gama) * ((gama - 1) * np.minimum(lamda1, 0) * u ** 2 + (3 - gama) / (2 * (gama - 1)) * (
            np.minimum(lamda2, 0) + np.minimum(lamda3, 0)) * c ** 2 + 0.5 * np.minimum(lamda2, 0) * (
                                                u + c) ** 2 + 0.5 * np.minimum(lamda3, 0) * (u - c) ** 2)
    return F_plus, F_minus

def spatial_derivative(F_plus, F_minus, quantity_index,delta_x=delta_x):
    # quantity_index 对应 0 为 rho，1 为 rho_u，2 为 e
    derivative = np.zeros_like(F_plus[quantity_index])
    N = len(derivative)

    # # NND
    #
    # F_plus_minus_1 = np.zeros_like(F_plus)
    # F_minus_minus_1 = np.zeros_like(F_minus)
    # F_plus_plus_1 = np.zeros_like(F_plus)
    # F_minus_plus_1 = np.zeros_like(F_minus)
    # F_plus_minus_3 = np.zeros_like(F_plus)
    # F_minus_minus_3 = np.zeros_like(F_minus)
    # F_plus_plus_3 = np.zeros_like(F_plus)
    # F_minus_plus_3 = np.zeros_like(F_minus)
    #
    # for m in range(2, n_l + n_r - 2):
    #     for j in range(3):
    #         F_plus_minus_1[j, m] = F_plus[j, m] - F_plus[j, m - 1]
    #         F_minus_minus_1[j, m] = F_minus[j, m] - F_minus[j, m - 1]
    #
    #         F_plus_plus_1[j, m] = F_plus[j, m + 1] - F_plus[j, m]
    #         F_minus_plus_1[j, m] = F_minus[j, m + 1] - F_minus[j, m]
    #
    #         F_plus_minus_3[j, m] = F_plus[j, m - 1] - F_plus[j, m - 2]
    #         F_minus_minus_3[j, m] = F_minus[j, m - 1] - F_minus[j, m - 2]
    #
    #         F_plus_plus_3[j, m] = F_plus[j, m + 2] - F_plus[j, m + 1]
    #         F_minus_plus_3[j, m] = F_minus[j, m + 2] - F_minus[j, m + 1]
    #
    # # 计算 min mod 系数，简化方程复杂度
    # x1 = (np.sign(F_plus_minus_1) == np.sign(F_plus_plus_1))
    # x2 = (np.sign(F_minus_plus_1) == np.sign(F_minus_plus_3))
    # x3 = (np.sign(F_plus_minus_3) == np.sign(F_plus_minus_1))
    # x4 = (np.sign(F_minus_minus_1) == np.sign(F_minus_plus_1))
    #
    # for m in range(2, N - 2):
    #     for j in range(3):
    #         derivative[m] += (
    #                 F_plus[j, m] +
    #                 0.5 * x1[j, m] * np.sign(F_plus_minus_1[j, m]) * np.minimum(np.abs(F_plus_minus_1[j, m]),
    #                                                                             np.abs(F_plus_plus_1[j, m])) +
    #                 F_minus[j, m + 1] -
    #                 0.5 * x2[j, m] * np.sign(F_minus_plus_1[j, m]) * np.minimum(np.abs(F_minus_plus_1[j, m]),
    #                                                                             np.abs(F_minus_plus_3[j, m])) -
    #                 F_plus[j, m - 1] -
    #                 0.5 * x3[j, m] * np.sign(F_plus_minus_3[j, m]) * np.minimum(np.abs(F_plus_minus_3[j, m]),
    #                                                                             np.abs(F_plus_minus_1[j, m])) -
    #                 F_minus[j, m] +
    #                 0.5 * x4[j, m] * np.sign(F_minus_minus_1[j, m]) * np.minimum(np.abs(F_minus_minus_1[j, m]),
    #                                                                              np.abs(F_minus_plus_1[j, m]))
    #         )

    # GVC格式
    # for m in range(2, N - 2):
    #     abs_diff1_plus = np.abs(F_plus[quantity_index, m] - F_plus[quantity_index, m - 1])
    #     abs_diff2_plus = np.abs(F_plus[quantity_index, m + 1] - F_plus[quantity_index, m])
    #
    #     derivative[m] = np.where(
    #         abs_diff1_plus <= abs_diff2_plus,
    #         1 / 2 * (F_plus[quantity_index, m - 1] - 4 * F_plus[quantity_index, m] + 3 * F_plus[quantity_index, m + 1]
    #                  - 2* F_minus[quantity_index, m] + 2*F_minus[quantity_index, m + 1]),
    #         1 / 2 * (F_plus[quantity_index, m + 1] - F_plus[quantity_index, m - 1] + 2*F_minus[quantity_index, m + 1] - 2*F_minus[quantity_index, m]))

    # 一阶迎风
    for m in range(1, N - 1):
        # print(m)
        derivative[m] = (F_plus[quantity_index, m] - F_plus[quantity_index, m - 1] )+(
                         F_minus[quantity_index, m + 1] - F_minus[quantity_index, m])
    return derivative / delta_x


def Euler(rho, rho_u, u, e, p):

    F_plus, F_minus = Steger_Warming(p, rho, u)

    # 更新所有字段
    rho = rho - delta_t * spatial_derivative(F_plus, F_minus, 0)
    rho_u -= delta_t * spatial_derivative(F_plus, F_minus, 1)
    e -= delta_t * spatial_derivative(F_plus, F_minus, 2)

    # 根据新的 rho 和 e 更新 u 和 p
    # u[2:len(rho) - 1] = rho_u[2:len(rho) - 1] / rho[2:len(rho) - 1]
    # p[2:len(rho) - 1] = (gama - 1) * (e[2:len(rho) - 1] - 0.5 * rho[2:len(rho) - 1] * u[2:len(rho) - 1] ** 2)
    u = rho_u / rho
    p = (gama - 1) * (e - 0.5 * rho * u ** 2)

    return rho, u, p

def test2(rho, rho_u, u, e, p): # 存在问题

    F_plus, F_minus = Steger_Warming(p, rho, u)
    k1_rho = -1*spatial_derivative(F_plus,F_minus,0)
    k1_rho_u = -1 * spatial_derivative(F_plus,F_minus,1)
    k1_e = -1 * spatial_derivative(F_plus,F_minus,2)

    rho += delta_t * k1_rho
    # rho_u += delta_t * k1_rho_u
    rho_u1 = rho_u + delta_t * k1_rho_u
    e += delta_t * k1_e
    u = rho_u1 / rho
    p = (gama - 1) * (e - 0.5 * rho * u ** 2)

    return rho, u, p

def RK3(rho, rho_u, u, e, p):
    # 计算 k1
    F_plus, F_minus = Steger_Warming(p, rho, u)
    k1_rho = -delta_t * spatial_derivative(F_plus, F_minus, 0)
    k1_rho_u = -delta_t * spatial_derivative(F_plus, F_minus, 1)
    k1_e = -delta_t * spatial_derivative(F_plus, F_minus, 2)

    # 计算临时的更新后的值
    rho1 = rho + k1_rho
    rho_u1 = rho_u + k1_rho_u
    e1 = e + k1_e

    # 计算 k2
    F_plus, F_minus = Steger_Warming(p, rho1, u)
    k2_rho = -delta_t * spatial_derivative(F_plus, F_minus, 0)
    k2_rho_u = -delta_t * spatial_derivative(F_plus, F_minus, 1)
    k2_e = -delta_t * spatial_derivative(F_plus, F_minus, 2)

    # 适当更新为第二个临时值
    rho2 = 0.75 * rho + 0.25 * (rho1 + k2_rho)
    rho_u2 = 0.75 * rho_u + 0.25 * (rho_u1 + k2_rho_u)
    e2 = 0.75 * e + 0.25 * (e1 + k2_e)

    # 计算 k3
    F_plus, F_minus = Steger_Warming(p, rho2, u)
    k3_rho = -delta_t * spatial_derivative(F_plus, F_minus, 0)
    k3_rho_u = -delta_t * spatial_derivative(F_plus, F_minus, 1)
    k3_e = -delta_t * spatial_derivative(F_plus, F_minus, 2)

    # 最终更新
    rho += (1/3 * k1_rho + 2/3 * k3_rho)
    rho_u += (1/3 * k1_rho_u + 2/3 * k3_rho_u)
    e += (1/3 * k1_e + 2/3 * k3_e)

    u = rho_u / rho
    p = (gama - 1) * (e - 0.5 * rho * u ** 2)

    return rho, u, p


if __name__ == '__main__':

    x = np.linspace(0, 1, n_l + n_r)
    fig, ax = plt.subplots()
    line_rho, = ax.plot(x, rho, label='rho')
    line_u, = ax.plot(x, u, label='u')
    line_p, = ax.plot(x, p, label='p')
    ax.legend()
    def update(frame_number):
        global rho, u, p
        rho, u, p = RK3(rho, rho_u, u, e, p)
        line_rho.set_ydata(rho)
        line_u.set_ydata(u)
        line_p.set_ydata(p)
        ax.set_title(f'Riemann Problem Numerical Solution t={delta_t+delta_t * frame_number:.4f}s')
        return line_rho, line_u, line_p


    ani = animation.FuncAnimation(fig, update, frames=range(time_steps), interval=1, repeat=False)
    plt.show()




