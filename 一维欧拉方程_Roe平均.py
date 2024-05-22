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
H = e/rho +p/rho

def Roe_avg(rho,u,H):
    rho_avg = np.zeros_like(rho)
    u_avg = np.zeros_like(u)
    H_avg = np.zeros_like(H)

    for m in range(1,len(u)-1):
        rho_avg[m] = (np.sqrt(rho[m]) + np.sqrt(rho[m + 1]) / 2) ** 2
        u_avg[m] = (np.sqrt(rho[m]) * u[m] + np.sqrt(rho[m + 1] * u[m + 1]) / (np.sqrt(rho[m]) + np.sqrt(rho[m + 1]))
        H_avg[m] = (np.sqrt(rho[m]) * H[m] + np.sqrt(rho[m + 1] * H[m + 1]) / (np.sqrt(rho[m]) + np.sqrt(rho[m + 1]))

    return rho_avg,u_avg,H_avg

