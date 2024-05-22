import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n_points = 100
h = 1 / n_points
delta_t = 0.002
x = np.arange(-0.5, 0.5, h)
t_end = 1

gama = 1.4

def UtoW(U):
    rho = U[0]
    u = U[1] / rho
    p = (U[2] - 1 / 2 * rho * u ** 2) * (gama - 1)
    return np.array([rho, u, p])

def WtoU(W):
    rho = W[0]
    u = W[1]
    p = W[2]
    return np.array([rho, rho * u, 1 / 2 * rho * u ** 2 + p / (gama - 1)])

U1 = np.where(x<0, 1,0.125) # rho
U2 = np.where(x<0, 0,0) # u
# U2 = np.zeros_like(x) # u
U3 = np.where(x<0, 1, 0.1) # p
U = np.array([U1, U2, U3])
rho,u,p = U
def F(U):
    rho,u,p = U
    # print("rho",rho)
    c = np.sqrt(gama*p/(rho+1e-6))
    # print("c",c)
    lamda = np.array([u-c, u, u+c])
    # print("lamda",lamda)
    lamda_plus = np.where(lamda>0, lamda, 0)

    # lamda_plus = np.zeros_like(lamda)
    # for i in range(3):
    #     lamda_plus[i] = (lamda[i] + (lamda[i] ** 2 + 1e-6 ** 2) ** 0.5) / 2

    # print("lamda_plus",lamda_plus)
    lamda_minus = np.where(lamda<0, lamda, 0)

    # print("lamda_minus",lamda_minus)
    tem1 = 2*(gama-1)*lamda_plus[0]+lamda_plus[1]+lamda_plus[2]
    tem2 = 2*(gama-1)*lamda_plus[0]*u+lamda_plus[1]*(u+c)+lamda_plus[2]*(u-c)
    tem3 = (gama-1)*lamda_plus[0]*u**2+1/2*lamda_plus[1]*(u+c)**2+1/2*lamda_plus[2]*(u-c)**2+(3-gama)*(lamda_plus[1]*c**2+lamda_plus[2]*c**2)/2/(gama-1)
    f_plus = np.array([tem1 * rho / 2 / gama, tem2 * rho / 2 / gama, tem3 * rho / 2 / gama])

    tem1 = 2*(gama-1)*lamda_minus[0]+lamda_minus[1]+lamda_minus[2]
    tem2 = 2*(gama-1)*lamda_minus[0]*u+lamda_minus[1]*(u+c)+lamda_minus[2]*(u-c)
    tem3 = (gama-1)*lamda_minus[0]*u**2+lamda_minus[1]*(u+c)**2/2+lamda_minus[2]*(u-c)**2/2+((3-gama)*(lamda_minus[1]*c**2+lamda_minus[2]*c**2))/2/(gama-1)
    f_minus = np.array([tem1*rho/2/gama,tem2*rho/2/gama,tem3*rho/2/gama])

    return f_plus, f_minus



def spatial_derivative(U):
    f_plus, f_minus = F(U)
    # print("f_plus", f_plus)
    # print("f_minus", f_minus)

    # 初始化导数数组
    f_x_plus = np.zeros_like(f_plus)
    f_x_minus = np.zeros_like(f_minus)

    # 使用迎风差分方案
    # f_x_plus 使用向右的迎风差分
    f_x_plus[:, 1:] = (f_plus[:, 1:] - f_plus[:, :-1]) / h
    # f_x_minus 使用向左的迎风差分
    f_x_minus[:, :-1] = (f_minus[:, 1:] - f_minus[:, :-1]) / h

    # 左边界处理：使用右边值减去左边值
    f_x_plus[:, 0] = (f_plus[:, 1] - f_plus[:, 0]) / h
    # 右边界处理：使用右边值减去左边值
    f_x_minus[:, -1] = (f_minus[:, -1] - f_minus[:, -2]) / h

    # print("f_x_plus", f_x_plus)
    # print("f_x_minus", f_x_minus)
    U_x = f_x_plus + f_x_minus
    # print("U_x", U_x)
    return U_x


def RK4(U):
    k1 = -1*spatial_derivative(U)
    k2 = -1*spatial_derivative(U + delta_t / 2 * k1)
    k3 = -1*spatial_derivative(U + delta_t / 2 * k2)
    k4 = -1*spatial_derivative(U + delta_t * k3)
    U_next = U + delta_t / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return U_next

def Euler(U):
    return U - delta_t * spatial_derivative(U)

def solve(U0, delta_t, t_end):
    t = 0
    U=U0.copy()
    frames = []
    while t < t_end:
        U = Euler(U)
        t += delta_t
        print(t)
        print("U",U)
        frames.append(U.copy())
    return frames

frames = solve(U, delta_t, t_end)

fig, ax = plt.subplots()
line1, = ax.plot(x, U[0], label='rho')
line2, = ax.plot(x, U[1], label='u')
line3, = ax.plot(x, U[2], label='p')

ax.legend(loc='upper right')


def update(frame_number):
    frame = frames[frame_number]
    line1.set_ydata(frame[0])
    line2.set_ydata(frame[1])
    line3.set_ydata(frame[2])
    return line1, line2, line3
    # return line1,

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=10)
plt.show()