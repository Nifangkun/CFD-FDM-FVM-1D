import numpy as np
import matplotlib.pyplot as plt
def k_r(alpha):
    k_r = 1/2 - 2/3*np.cos(alpha) + 1/6*np.cos(2*alpha)
    return k_r

def find_alpha(target_value,iskr, tol=1e-5, max_iter=1000):
    low, high = 0, np.pi
    iter_count = 0
    while low < high and iter_count < max_iter:
        mid = (low + high) / 2
        if iskr == 1:
            if np.abs(k_r(mid) - target_value) < tol:
                return mid
            elif k_r(mid) > target_value:
                high = mid
            else:
                low = mid
            iter_count += 1
        else:
            if np.abs(np.abs(1-k_i(mid)/mid) - target_value) < tol:
                return mid
            elif k_i(mid) > target_value:
                high = mid
            else:
                low = mid
            iter_count += 1
    return (low + high) / 2



def k_i(alpha):
    # k_i = np.sin(alpha)
    k_i = 2/3*np.sin(alpha) + 1/6*np.sin(2*alpha)
    return k_i


def plot_kr():
    x = np.linspace(0, np.pi, 100)
    y_r = [k_r(i) for i in x]
    alpha = find_alpha(err,1)
    kr= k_r(alpha)

    plt.plot(x, y_r, label='k_r')
    plt.plot(x, [0 for _ in x], label='k_r_ideal', linestyle='--')
    plt.scatter([alpha], [kr], color='red', zorder=5)
    plt.text(alpha, kr, f'({alpha:.3f}, {kr:.2f})', fontsize=9, verticalalignment='bottom')
    plt.grid(True)
    plt.xlabel('alpha (radians)')
    plt.ylabel('k_r')
    plt.legend()
    plt.show()

def plot_ki():
    x = np.linspace(0, np.pi, 100)
    y_i = [k_i(i) for i in x]
    y_i_ideal = x

    alpha = find_alpha(err,0)
    ki = k_i(alpha)


    plt.plot(x, y_i, label='k_i')
    plt.plot(x, y_i_ideal, label='k_i_ideal')
    plt.scatter([alpha], [ki], color='red', zorder=5)
    plt.text(alpha, ki, f'({alpha:.3f}, {ki:.2f})', fontsize=9, verticalalignment='bottom')

    plt.grid()
    plt.xlabel('alpha')
    plt.ylabel('ki')
    plt.legend()
    plt.show()

err = 0.05
plot_ki()