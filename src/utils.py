import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def solve_euler(f_ode, y0, time):
    """
    Solves the given ODE system in f_ode using forward Euler.
    :param f_ode: the right hand side of the ordinary differential equation d/dt x = f_ode(x(t)).
    :param y0: the initial condition to start the solution at.
    :param time: np.array of time values (equally spaced), where the solution must be obtained.
    :returns: (solution[time,values], time) tuple.
    """
    yt = np.zeros((len(time), len(y0)))
    yt[0, :] = y0
    step_size = time[1]-time[0]
    for k in range(1, len(time)):
        yt[k, :] = yt[k-1, :] + step_size * f_ode(yt[k-1, :])
    return yt, time


def plot_phase_portrait(A, X, Y):
    """
    Plots a linear vector field in a streamplot, defined with X and Y coordinates and the matrix A.
    """
    UV = A@np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0,:].reshape(X.shape)
    V = UV[1,:].reshape(X.shape)

    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X, Y, U, V, density=[0.5, 1])
    ax0.set_title('Streamplot for linear vector field A*x');
    ax0.set_aspect(1)
    return ax0



def lorenz(x, y, z, sigma, rho, beta):
    """
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    x_dot = sigma*(y - x)
    y_dot = rho*x - y - x*z
    z_dot = x*y - beta*z
    return x_dot, y_dot, z_dot

def plot_lorenz(sigma, rho, beta, dt, end_Time):
    """
        Plots a 3-dimensional Lorenz System. 
        Takes the parameters sigma, rho and beta as well as the time increment dt and the end time end_Time)
        Returns three figures: The simulation results for a run with the initial values 10, 10, 10, 
        for a second run with the initial values 10*10^-8, 10, 10, 
        and a 2d plot of the difference between the two, plotted over the time. 
        It also returns the difference vector for further analysis.
    """
    
    num_Steps = end_Time/dt
    # setting up the arrays for the results

    xs_1 = np.empty(int(num_Steps) + 1)
    ys_1 = np.empty(int(num_Steps) + 1)
    zs_1 = np.empty(int(num_Steps) + 1)

    xs_2 = np.empty(int(num_Steps) + 1)
    ys_2 = np.empty(int(num_Steps) + 1)
    zs_2 = np.empty(int(num_Steps) + 1)

    # setting up the arrays for time and the difference between the systems

    diff = np.empty(int(num_Steps))
    time_Space = np.linspace(0, end_Time, int(num_Steps))

    # set initial values for both simulations

    xs_1[0], ys_1[0], zs_1[0] = (10.0, 10.0, 10.0)
    xs_2[0], ys_2[0], zs_2[0] = (10+10**(-8), 10.0, 10.0)

    diff[0] = np.sqrt((xs_1[0] - xs_2[0])**2 + (ys_1[0] - ys_2[0])**2 + (zs_1[0] - zs_2[0])**2)

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point

    for i in range(int(num_Steps)):
        x_dot, y_dot, z_dot = lorenz(xs_1[i], ys_1[i], zs_1[i], sigma, rho, beta)
        xs_1[i + 1] = xs_1[i] + (x_dot * dt)
        ys_1[i + 1] = ys_1[i] + (y_dot * dt)
        zs_1[i + 1] = zs_1[i] + (z_dot * dt)
    
        x_dot, y_dot, z_dot = lorenz(xs_2[i], ys_2[i], zs_2[i], sigma, rho, beta)
        xs_2[i + 1] = xs_2[i] + (x_dot * dt)
        ys_2[i + 1] = ys_2[i] + (y_dot * dt)
        zs_2[i + 1] = zs_2[i] + (z_dot * dt)
    
        diff[i] = np.sqrt((xs_1[i] - xs_2[i])**2 + (ys_1[i] - ys_2[i])**2 + (zs_1[i] - zs_2[i])**2)
    
    # Plot

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection='3d')

    ax1.plot(xs_1, ys_1, zs_1, lw = "0.5")
    ax1.set_xlabel("X Axis")
    ax1.set_ylabel("Y Axis")
    ax1.set_zlabel("Z Axis")
    ax1.set_title("Lorenz Attractor for Run 1")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')

    ax2.plot(xs_2, ys_2, zs_2, lw = "0.5")
    ax2.set_xlabel("X Axis")
    ax2.set_ylabel("Y Axis")
    ax2.set_zlabel("Z Axis")
    ax2.set_title("Lorenz Attractor for Run 2")

    fig3 = plt.figure()
    ax3 = fig3.add_subplot()

    ax3.plot(time_Space, diff, lw = "0.5")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Metric difference between vectors")
    ax3.set_title("Difference between Run 1 and Run 2")
    
    return fig1, fig2, fig3, diff

def logistic_map(x, r):
    """
    Returns the population for x(t+1) for a given population of x(t) and a given parameter r
    """
    return r*x*(1-x)

def print_logistic_map(x_init, r):
    """
    Plots a logistic map for a given initial value of the population and the parameter r for 100 generations. 
    """
    
    x = np.linspace(0, 99, 100)
    population = x_init
    y = np.empty(100, dtype='float')

    for i in range(0, 100):
        population = logistic_map(population, r)
        y[i] = population

    fig = plt.figure()
    fig, ax = plt.subplots()
    
    ax.plot(x, y)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Population")
    ax.set_title("Logistic Map for r = " + str(r))
    
    return fig