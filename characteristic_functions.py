
import cmath
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb


def plot_characteristic_function(phi_s, bunch, time_pnts, ind_tau):
    
    sb.set_style('white')
    plt.figure()
    n_time_pnts = len(time_pnts)
    cmap = plt.cm.get_cmap('RdYlBu')
    for n in bunch:
            x = [phi_s[n, ind_tau * n_time_pnts + 2 * j]
                 for j in range(n_time_pnts)]
            y = [phi_s[n, ind_tau * n_time_pnts + 2 * j + 1]
                 for j in range(n_time_pnts)]
            plt.scatter(x, y, c=cmap(n), label="node "+str(n), cmap=cmap)
    plt.legend(loc='upper left')
    plt.title('characteristic function of the distribution as s varies')
    plt.show()
    return


def plot_angle_chi(f, t=[], savefig=False, filefig='plots/angle_chi.png'):
   
    if len(t) == 0:
        t = range(f.shape[0])
    theta = np.zeros(f.shape[0])
    for tt in t:
        theta[tt] = math.atan(f[tt, 1] * 1.0 / f[tt, 0])
    return theta


def charac_function(time_points, temp):
    temp2 = temp.T.tolil()
    d = temp2.data
    n_timepnts = len(time_points)
    n_nodes = temp.shape[1]
    final_sig = np.zeros((2 * n_timepnts, n_nodes))
    zeros_vec = np.array([1.0 / n_nodes*(n_nodes - len(d[i])) for i in range(n_nodes)])
    for i in range(n_nodes):
        final_sig[::2, i] = zeros_vec[i] +\
                            1.0 / n_nodes *\
                            np.cos(np.einsum("i,j-> ij",
                                             time_points,
                                             np.array(d[i]))).sum(1)
    for it_t, t in enumerate(time_points):
        final_sig[it_t * 2 + 1, :] = 1.0 / n_nodes * ((t*temp).sin().sum(0))

    return final_sig


def charac_function_multiscale(heat, time_points):
    final_sig = []
    for i in heat.keys():
        final_sig.append(charac_function(time_points, heat[i]))
    return np.vstack(final_sig).T






def charac_function_hyper(time_points, temp):
    temp2 = temp.T.tolil()
    d = temp2.data
    n_timepnts = len(time_points)
    n_nodes = temp.shape[1]
    final_sig = np.zeros((2 * n_timepnts, n_nodes))
    zeros_vec = np.array([1.0 / n_nodes*(n_nodes - len(d[i])) for i in range(n_nodes)])
    for i in range(n_nodes):
        final_sig[::2, i] = zeros_vec[i] +\
                            1.0 / n_nodes *\
                            np.cosh(np.einsum("i,j-> ij",
                                             time_points,
                                             np.array(d[i]))).sum(1)
    for it_t, t in enumerate(time_points):
        final_sig[it_t * 2 + 1, :] = 1.0 / n_nodes * ((t*temp).sinh().sum(0))

    return final_sig


def charac_function_multiscale_hyper(heat, time_points):
    final_sig = []
    for i in heat.keys():
        final_sig.append(charac_function_hyper(time_points, heat[i]))
    return np.vstack(final_sig).T