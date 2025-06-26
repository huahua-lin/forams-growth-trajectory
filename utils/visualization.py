from matplotlib import pyplot as plt


def plot_trajectory(points, path, name):
    """
    Plot the trajectory of the TSP path in 3D space.

    Args:
        points: Original points in 3D space.
        path: The ordered path of points after solving the TSP.

    """
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111, projection='3d')

    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], color='r', label='Points', s=50)
    ax1.plot(path[:, 0], path[:, 1], path[:, 2], marker='o', color='b', label='pred_path', linestyle='-',
             markersize=5)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()

    fig.suptitle(name, fontsize=16)
    plt.show()


def compare_trajectories(points, path1, path2, name, rank):
    """
    Plot two trajectories with same points side by side.

    Agrs:
        points: Points in 3D space.
        path1: The ordered path of points after solving the TSP.
        path2: The ordered path of points after solving the TSP.
        name: The name of the trajectory.
        rank: The spearman correlation between the trajectories.

    """
    # Plot the points and the path in 3D
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(121, projection='3d')

    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], color='r', label='Points', s=50)
    ax1.plot(path1[:, 0], path1[:, 1], path1[:, 2], marker='o', color='b', label='pred_path', linestyle='-',
             markersize=5)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')

    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], color='r', label='Points', s=50)
    ax2.plot(path2[:, 0], path2[:, 1], path2[:, 2], marker='o', color='g', label='gt_path', linestyle='-',
             markersize=5)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()

    fig.suptitle(f'The correlation of {name} is: {rank}', fontsize=16)
    plt.show()
