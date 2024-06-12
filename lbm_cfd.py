import numpy as np
from matplotlib import pyplot

plot_every = 2

def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def main():
    x_cells = 400
    y_cells = 100
    tau = 0.53
    num_iterations = 50000

    # Lattice speeds and weights
    num_lattices = 9
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])

    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    # Initial conditions
    F = np.ones((y_cells, x_cells, num_lattices)) + 0.01 * np.random.randn(y_cells, x_cells, num_lattices)
    F[:, :, 3] = 2.3

    obstacle = np.full((y_cells, x_cells), False)

    for y in range(0, y_cells):
        for x in range(0, x_cells):
            if(distance(x_cells//4, y_cells//2, x, y) < 13):
                obstacle[y][x] = True

    
    # Main loop

    for iter in range(num_iterations):
        print(iter)

        F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]
        F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]

        for i, cx, cy in zip(range(num_lattices), cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis = 1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis = 0)

        bndryF = F[obstacle, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

        # Fluid variables
        rho = np.sum(F, 2)
        ux = np.sum(F * cxs, 2) / rho
        uy = np.sum(F * cys, 2) / rho

        F[obstacle, :] = bndryF
        ux[obstacle] = 0
        uy[obstacle] = 0

        # Collision
        F_eq = np.zeros(F.shape)
        for i, cx, cy, w in zip(range(num_lattices), cxs, cys, weights):
            F_eq[:, :, i] = rho * w * (
                1 + 3 * (cx*ux + cy*uy) + 9 * (cx*ux + cy*uy)**2 / 2 - 3 * (ux**2 + uy**2)/2
            )

        F = F + -(1/tau) * (F - F_eq)

        if(iter % plot_every == 0):
            pyplot.imshow(np.sqrt(ux**2 + uy**2))
            pyplot.pause(0.01)
            pyplot.cla()


if __name__ == "__main__":
    main()