import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.optimize import root_scalar

gama = 1.4
def fp(pp, p, rho):

    c=sqrt(gama*p/rho)
    if pp >= p:
        return (pp - p) / (rho * c * sqrt(((gama + 1) / 2 / gama) * (pp / p) + (gama - 1) / 2 / gama))
    else:
        return 2*c/(gama-1)*(pow((pp/p),(gama-1)/(2*gama))-1)

def Fp(pp,u1,u2,p1,rho1,p2,rho2):
    return fp(pp,p1,rho1)+fp(pp,p2,rho2)-u1+u2




def solve(u1, u2, p1, rho1, p2, rho2, pp_min, pp_max):
    Fp_function = lambda pp: Fp(pp, u1, u2, p1, rho1, p2, rho2)
    solution = root_scalar(Fp_function, bracket=[pp_min, pp_max])

    if solution.converged:
        return solution.root
    else:
        raise ValueError("Root finding failed. Try with different initial conditions or a wider bracket.")

def get_u(pp,u1,u2,p1,rho1,p2,rho2):
    return (u1+u2+fp(pp,p2,rho2)-fp(pp,p1,rho1))/2

def get_Z_rrho(pp,uu,u1,u2,p1,rho1,p2,rho2):
    c1 = sqrt(gama * p1 / rho1)
    cc1 = c1  +  (gama-1)*(u1-uu)/2
    A1 = rho1*c1*sqrt((gama+1)/2/gama*pp/p1+(gama-1)/2/gama)
    c2 = sqrt(gama * p2 / rho2)
    cc2 = c2 + (gama - 1) * (u2 - uu) / 2
    A2 = rho2 * c2 * sqrt((gama + 1) / 2 / gama * pp / p2 + (gama - 1) / 2 / gama)
    if pp >= p1:
        Z1 = u1-A1/rho1
        rrho1 = rho1*A1/(A1-rho1*(u1-uu))
        if pp>=p2:
            Z2=u2-A2/rho2
            rrho2=rho2*A2/(A2+rho2*(u2-uu))
            return Z1,Z1,rrho1,Z2,Z2,rrho2
        else:
            Z_h2=u2+c2
            Z_t2=uu+cc2
            rrho2 = gama*pp/cc2/cc2
            return Z1,Z1,rrho1,Z_h2,Z_t2,rrho2
    else:
        Z_h1=u1-c1
        Z_t1=uu-cc1
        rrho1=gama*pp/cc1/cc1
        if pp>=p2:
            Z2 = u2 + A2 / rho2
            rrho2 = rho2 * A2 / (A2 + rho2 * (u2 - uu))
            return Z_h1, Z_t1, rrho1, Z2, Z2, rrho2
        else:
            Z_h2 = u2 + c2
            Z_t2 = uu + cc2
            rrho2 = gama * pp / cc2 / cc2
            return Z_h1, Z_t1, rrho1, Z_h2, Z_t2, rrho2


def plot(u1,u2,p1, rho1, p2,rho2,pp_min, pp_max, num_points=10000):
    # 定义pp的范围
    pp_values = np.linspace(pp_min, pp_max, num_points)

    # 计算相应的Fp值
    Fp_values = [Fp(pp,u1,u2, p1, rho1,p2,rho2) for pp in pp_values]
    # 绘图
    plt.plot(pp_values, Fp_values)
    plt.title('Fp vs pp')
    plt.xlabel('pp')
    plt.ylabel('Fp')
    plt.grid(True)
    plt.show()


p1 = 0.4
rho1 = 1
p2=0.4
rho2=1
u1=-2
u2=2
pp_min = 0
pp_max = 2
# print(Fp(2,0,0,1,1,0.1,0.125))
# plot(u1,u2,p1, rho1, p2,rho2,pp_min, pp_max)

pp = solve(u1, u2, p1, rho1, p2, rho2, pp_min, pp_max)
uu = get_u(pp,u1,u2,p1,rho1,rho2,rho2)
print("pp = ",pp)
print("uu = ",uu)
Z_h1,Z_t1,rrho1,Z_h2,Z_t2,rrho2=get_Z_rrho(pp,uu,u1, u2, p1, rho1, p2, rho2)
print("Z_h1,Z_t1,rrho1,Z_h2,Z_t2,rrho2 = ",Z_h1,',',Z_t1,',',rrho1,',',Z_h2,',',Z_t2,',',rrho2)