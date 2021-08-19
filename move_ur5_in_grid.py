import numpy as np
import quadpy
from warnings import warn
from LebedevGrid import genGrid
import matplotlib.pyplot as plt
import urx
import cv2
import time
from math import atan2
from scipy.spatial.transform import Rotation as sciRot


def cart2sph(x, y, z):
    r"""Cartesian to spherical coordinate transform.

    .. math::

        \phi = \arctan \left( \frac{y}{x} \right) \\
        \theta = \arccos \left( \frac{z}{r} \right) \\
        r = \sqrt{x^2 + y^2 + z^2}

    with :math:`\phi \in [-pi, pi], \theta \in [0, \pi], r \geq 0`

    Parameters
    ----------
    x : float or array_like
        x-component of Cartesian coordinates
    y : float or array_like
        y-component of Cartesian coordinates
    z : float or array_like
        z-component of Cartesian coordinates

    Returns
    -------
    phi : float or `numpy.ndarray`
            Azimuth angle in radiants
    theta : float or `numpy.ndarray`
            Colatitude angle in radiants (with 0 denoting North pole)
    r : float or `numpy.ndarray`
            Radius

    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)
    return np.array([phi, theta, r])


def sph2cart(alpha, beta, r):
    r"""Spherical to cartesian coordinate transform.

    .. math::

        x = r \cos \alpha \sin \beta \\
        y = r \sin \alpha \sin \beta \\
        z = r \cos \beta

    with :math:`\alpha \in [0, 2\pi), \beta \in [0, \pi], r \geq 0`

    Parameters
    ----------
    alpha : float or array_like
            Azimuth angle in radiants
    beta : float or array_like
            Colatitude angle in radiants (with 0 denoting North pole)
    r : float or array_like
            Radius

    Returns
    -------
    x : float or `numpy.ndarray`
        x-component of Cartesian coordinates
    y : float or `numpy.ndarray`
        y-component of Cartesian coordinates
    z : float or `numpy.ndarray`
        z-component of Cartesian coordinates

    """
    x = r * np.cos(alpha) * np.sin(beta)
    y = r * np.sin(alpha) * np.sin(beta)
    z = r * np.cos(beta)
    return np.array([x, y, z])


def grid_lebedev(n):
    """Lebedev sampling points on sphere.
    (Maximum n is 65. We use what is available in quadpy, some n may not be
    tight, others produce negative weights.
    Parameters
    ----------
    n : int
        Maximum order.
    Returns
    -------
    azi : array_like
        Azimuth.
    colat : array_like
        Colatitude.
    weights : array_like
        Quadrature weights.
    """

    def available_quadrature(d):
        """Get smallest availabe quadrature of of degree d.
        see:
        https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/sphere_lebedev_rule.html
        """
        l = list(range(1, 32, 2)) + list(range(35, 132, 6))
        matches = [x for x in l if x >= d]
        return matches[0]

    if n > 65:
        raise ValueError("Maximum available Lebedev grid order is 65. "
                         "(requested: {})".format(n))

    # this needs https://pypi.python.org/pypi/quadpy
    q = quadpy.sphere.Lebedev(degree=available_quadrature(2 * n))
    if np.any(q.weights < 0):
        warn("Lebedev grid of order {} has negative weights.".format(n))
    azi = q.azimuthal_polar[:, 0]
    colat = q.azimuthal_polar[:, 1]
    return azi, colat, 4 * np.pi * q.weights


def sort_colatitude(grid, type='ascend'):
    """
    Assuming grid is in spherical coords, sorts grid in accordance to colatitude (elevation)
    angle in ascending or descending order

    :param grid: (ndarray of float32) [3, N] : phi, theta, r
    :param type: (str) defines ascending or descending order, default is type = 'ascend'
    :return: newgrid: sorted grid
    :return:
    """
    indx = np.argsort(grid, axis=1)
    if type == 'descend':
        indx = indx[:, ::-1]
    newgrid = grid[:, indx[1]]
    return newgrid


def sort_ydim(grid, type='ascend'):
    """
    Assuming grid is in Cartesian coords, sorts grid in accordance to y dimension in ascending or descending order

    :param grid: (ndarray of float32) [3, N] : x, y, z
    :param type: (str) defines ascending or descending order, default is type = 'ascend'
    :return: newgrid: sorted grid
    """
    indx = np.argsort(grid, axis=1)
    if type == 'descend':
        indx = indx[:, ::-1]
    newgrid = grid[:, indx[1]]
    return newgrid


def euler_to_quaternion(Yaw, Pitch, Roll):
    yaw = Yaw * np.pi / 180
    pitch = Roll * np.pi / 180
    roll = Pitch * np.pi / 180

    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]


def goto_straight_status(robot):
    """
    Send the robot to a 'straight line' pose

    :param robot: python-urx robot object
    :return:
    """
    print("\n============ Press `Enter` to go back straight-up status ============")
    # input()
    v = 0.1
    a = 0.1

    joint = []
    joint.append(0)  # Base
    joint.append(-np.pi / 2)  # Shoulder
    joint.append(0)  # Elbow
    joint.append(-np.pi / 2)  # Wrist1
    joint.append(0)  # Wrist2
    joint.append(0)  # Wrist3
    robot.movej(joint, acc=a, vel=v, wait=False)

    print("Robot current joint positions:")
    print(robot.getj())


def initial_position(robot):
    """
    Send the robot to a reference (easy to reach) position to initialise movement

    :param robot: python-urx robot object
    :return:
    """
    print("Press `Enter` to go to initial position")
    # input()
    v = 0.8
    a = 0.3

    joint = []
    joint.append(-np.pi / 2)  # Base
    joint.append(np.deg2rad(-110))  # Shoulder
    joint.append(-np.pi / 2)  # Elbow
    joint.append(0)  # Wrist1
    joint.append(np.pi / 2)  # Wrist2
    joint.append(0)  # Wrist3
    robot.movej(joint, acc=a, vel=v, wait=False)

    print("Robot current joint positions:")
    print(robot.getj())


def euler_to_rotationVector_many(yaw, roll, pitch):
    """
    Unused

    :param yaw:
    :param roll:
    :param pitch:
    :return:
    """
    nums = euler_to_quaternion(yaw, roll, pitch)

    # R = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
    x = nums[0]
    y = nums[1]
    z = nums[2]
    w = nums[3]

    Rot = []
    # calculate element of matrix
    for ii in range(len(x)):
        R = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
        R[0][0] = np.square(w[ii]) + np.square(x[ii]) - np.square(y[ii]) - np.square(z[ii])
        R[0][1] = 2 * (x[ii] * y[ii] + w[ii] * z[ii])
        R[0][2] = 2 * (x[ii] * z[ii] - w[ii] * y[ii])
        R[1][0] = 2 * (x[ii] * y[ii] - w[ii] * z[ii])
        R[1][1] = np.square(w[ii]) - np.square(x[ii]) + np.square(y[ii]) - np.square(z[ii])
        R[1][2] = 2 * (w[ii] * x[ii] + y[ii] * z[ii])
        R[2][0] = 2 * (x[ii] * z[ii] + w[ii] * y[ii])
        R[2][1] = 2 * (y[ii] * z[ii] - w[ii] * x[ii])
        R[2][2] = np.square(w[ii]) - np.square(x[ii]) - np.square(y[ii]) + np.square(z[ii])
        rotationVector = cv2.Rodrigues(R)
        Rot.append(rotationVector[0])

    print("yaw, roll and pitch:")
    print(rotationVector[0])
    return rotationVector[0], Rot


def robot_orientation_grid(grid):
    """
    Unused

    Assuming grid is in spherical coords returns roll, pitch, yaw
    values relative to the tool coordinate system
    :param grid: [3, N] : x, y, z (dimension, points)
    :return orientation: [3, N] (roll, pitch, yaw)
    """
    grid_sph = cart2sph(grid[0], grid[1], grid[2])
    orientation = np.zeros((grid.shape[-1],))
    mask = grid_sph[1, :] > np.pi / 4
    orientation[mask] = 3 / 4 * np.pi

    # middle area close xy plane
    mask_1 = grid_sph[1] >= np.pi / 2
    mask_2 = np.logical_and((-1 / 4 <= grid[1]), (grid[1] <= 1 / 2))
    mask = np.logical_and(mask_1, mask_2)
    orientation[mask] = 3 / 4 * np.pi

    # far from robot close  xy plane
    mask_1 = grid_sph[1] >= np.pi / 4
    mask_2 = np.logical_and(grid_sph[1] >= 3 / 4 * np.pi, grid[1] <= -1 / 2)
    mask = np.logical_and(mask_1, mask_2)
    orientation[mask] = np.pi / 2

    # close to robot below 45 deg elevation
    mask = np.logical_and(grid_sph[1] >= np.pi / 2, grid[1] >= 1 / 2)
    orientation[mask] = np.pi

    # above xy plane close to the robot
    mask_1 = np.logical_and(grid_sph[1] <= np.pi / 2, grid_sph[1] > np.pi / 3)
    mask_2 = grid[1] > 3 / 4
    mask = np.logical_and(mask_1, mask_2)
    orientation[mask] = np.pi

    # above xy plane far away from robot
    mask_1 = np.logical_and(grid_sph[1] <= np.pi / 2, grid_sph[1] > np.pi / 3)
    mask_2 = grid[1] < 0
    mask = np.logical_and(mask_1, mask_2)
    orientation[mask] = np.pi / 2

    # above the xy plane middle region
    mask_1 = np.logical_and(grid_sph[1] <= np.pi / 2,
                            grid_sph[1] > np.pi / 3)
    mask_2 = np.logical_and(grid[1] >= 0, grid[1] < 3 / 4)
    mask = np.logical_and(mask_1, mask_2)
    orientation[mask] = 3 * np.pi / 4

    # northpole cap close to robot
    mask = np.logical_and(grid_sph[1] <= np.pi / 3, grid[1] > 0)
    orientation[mask] = 3 * np.pi / 4

    # northpole cap away from the robot
    mask = np.logical_and(grid_sph[1] <= np.pi / 3, grid[1] <= 0)
    orientation[mask] = np.pi / 2
    return orientation


def correct_robot_orientation(robot):
    """
    Correct the robot (UR5) pose so it does not reach a 'singularity' while moving in a grid

    :param robot: python-urx robot object
    :return:
    """
    orientation = robot.get_orientation()

    get_current_pose = robot.getl()
    current_rot = np.asarray(get_current_pose[3::])
    indx = abs(current_rot) > 1.8
    compensation_angle_rad = []
    axis_to_compensate = []
    for ii, ind in enumerate(indx):
        if ind * current_rot[ii] == 0:
            continue
        else:
            compensation_angle_rad.append(-np.sign(current_rot[ii]) * current_rot[ii] / 4)
            axis_to_compensate.append(ii)

    for jj, axis in enumerate(axis_to_compensate):
        if axis == 0:
            orientation.rotate_xb(compensation_angle_rad[jj])
        elif axis == 1:
            orientation.rotate_yb(compensation_angle_rad[jj])
        elif axis == 2:
            orientation.rotate_zb(compensation_angle_rad[jj])

    # robot.set_orientation(orientation, acc=0.5, vel=0.2)
    robot.set_orientation(orientation, acc=0.8, vel=0.3, wait=False)


def move_robot_joints_to_angles(robot, base, shoulder, elbow, wrist1, wrist2, wrist3):
    """
    Move robot (UR5) do a position in terms of its joints and their angles. This function is used as a safeguard for
    the robot positioning as it is much easier to move the robot with respect to the urx function 'movej' and the
    robot joints. All angles in degrees.

    :param robot: python-urx robot object
    :param base: (float32) angle of base joint (degrees)
    :param shoulder: (float32) angle of shoulder joint (degrees)
    :param elbow: (float32) angle of elbow joint (degrees)
    :param wrist1: (float32) angle of wrist1 joint (degrees)
    :param wrist2: (float32) angle of wrist2 joint (degrees)
    :param wrist3: (float32) angle of wrist3 joint (degrees)
    :return:
    """
    print("Press `Enter` to go to correct robot joint")
    # input()
    v = 0.3
    a = 0.8

    joint = []
    joint.append(np.deg2rad(base))  # Base
    joint.append(np.deg2rad(shoulder))  # Shoulder
    joint.append(np.deg2rad(elbow))  # Elbow
    joint.append(np.deg2rad(wrist1))  # Wrist1
    joint.append(np.deg2rad(wrist2))  # Wrist2
    joint.append(np.deg2rad(wrist3))  # Wrist3
    robot.movej(joint, acc=a, vel=v, wait=False)

    # print("Robot current joint positions:")
    # print(robot.getj())


def correct_robot_joints_y(robot):
    """
    Correct robot joints when measurement position is close to base to a 'safe' position
    :param robot: python-urx robot object
    :return:
    """
    # print("Press `Enter` to go to correct robot joint")
    # input()
    v = 0.3
    a = 0.8

    joint = []
    joint.append(np.deg2rad(140))  # Base
    joint.append(np.deg2rad(-60))  # Shoulder
    joint.append(np.deg2rad(90))  # Elbow
    joint.append(np.deg2rad(-70))  # Wrist1
    joint.append(np.deg2rad(95))  # Wrist2
    joint.append(0)  # Wrist3
    robot.movej(joint, acc=a, vel=v, wait=False)

    # print("Robot current joint positions:")
    # print(robot.getj())


def get_rot_vector(pos, angle):
    """
    Unused
    :param pos:
    :param angle:
    :return:
    """
    theta_x = 0
    theta_y = angle  # change this to -pi/2 to place the arm horizontally
    theta_z = atan2(pos[1], pos[0])
    r = sciRot.from_euler('zyx', [theta_z, theta_y, theta_x])
    return r.as_rotvec()


def move_to_single_position(robot, x, y, z, close_pose=False):
    """
    Moves the robot (UR5) to a single position defined by x, y, z in cartesian coordinates

    :param robot: python-urx robot object
    :param x: (float32) value of x-axis in cartesian coordinates (defined in meters)
    :param y: (float32) value of y-axis in cartesian coordinates (defined in meters)
    :param z: (float32) value of z-axis in cartesian coordinates (defined in meters)
    :param close_pose: (Bool) variable determining whether or not the robot is in a pose
                        to measure close to its base
    :return:
    """
    # print("\n============ Press `Enter` correct pose if necessary ============")
    # input()
    if close_pose:
        correct_robot_joints_y(robot)
        time.sleep(2)
    get_new_pose = robot.getl()

    # print("\n============ Press `Enter` to move to certain XYZRPY ============")
    # input()

    # rotation = euler_to_rotationVector(yaw, roll, pitch)  # yaw->roll->pitch
    pose = []
    pose.append(x)  # px
    pose.append(y)  # py
    pose.append(z)  # pz
    pose.append(get_new_pose[3])  # rx
    pose.append(get_new_pose[4])  # ry
    pose.append(get_new_pose[5])  # rz

    # if y < -0.56:
    #     pose.append(get_new_pose[3])  # rx
    #     pose.append(get_new_pose[4])  # ry
    #     pose.append(get_new_pose[5])  # rz
    # else:
    #     pose.append(rotation[0])  # rx
    #     pose.append(rotation[1])  # ry
    #     pose.append(rotation[2])  # rz
    #
    # robot.movel(pose, acc=0.005, vel=0.06)
    robot.movel(pose, acc=0.8, vel=0.3, wait=False)


def reference_grid(steps, xmin=-.7, xmax=.7, ymin=-.4, ymax=.4, z=0):
    """
    Define a uniformly spaced reference grid symmetric about its indices (e.g. square-like or rectangle-like)

    :param steps: number of steps to take in each of the x,y dimensions
    :param xmin: minimum distance in x-axis (defined in meters), default is xmin = -0.7
    :param xmax: maximum distance in x-axis (defined in meters), default is xmax = 0.7
    :param ymin: minimum distance in y-axis (defined in meters), default is ymin = -0.4
    :param ymax: maximum distance in x-axis (defined in meters), default is ymax = 0.4
    :param z: constant value for z axis (e.g. height, defined in meters), default is z = 0
    :return: X,Y,Z ~ [steps, steps] variables as determined by a 2D mesh grid
    """
    x = np.linspace(xmin, xmax, steps)
    y = np.linspace(ymin, ymax, steps)
    # z = tf.zeros(shape = (steps,))
    X, Y = np.meshgrid(x, y)
    Z = z * np.ones(X.shape)
    return X, Y, Z


def robot_pose_check(grid):
    """
    Check whether or not to change the robot pose.

    :param grid: grid in which robot is moving, [3, n_points]
    :return: close_robot_pose: boolean variable indicating whether pose must be changed
    """
    if grid[0] < .15 and grid[1] > -.6:
        close_robot_pose = True
    else:
        close_robot_pose = False

    return close_robot_pose


def move_robot_in_grid(robot, grid, begin_point=0, end_point=-1):
    """
    A function to move UR5 robot in a predefined grid with certain constraints, please see rest of script as an
    indication of how the grid must be defined for usage.

    :param robot: python-urx robot object
    :param grid: grid in which robot will move, [3, n_points]
    :param begin_point: index of first point to start sampling, default is first index of 'grid' (e.g. begin_point = 0)
    :param end_point: index of last point to sample, default is last index of 'grid' (e.g. end_point = -1)
    :return:
    """
    problematic_positions = []

    current_robot_pose = False  # is true when robot needs to be close to its base
    next_robot_pose = current_robot_pose  # the same as above
    change_pose = False  # changes when next_robot_pose != current_robot_pose
    if end_point == -1:
        N = grid.shape[-1]  # number of points in reference grid
    else:
        N = end_point
    for jj in range(begin_point, N):
        if grid[1, jj] > -.6 and next_robot_pose is False:
            if change_pose is True:
                initial_position(robot)
                time.sleep(8)
            if grid[0, jj] < .15:
                move_robot_joints_to_angles(robot, -30, -130, -90, -130, 60, 0)
                time.sleep(8)
            else:
                initial_position(robot)
        if grid[0, jj] > .15 and next_robot_pose is True:
            initial_position(robot)
            time.sleep(8)
        elif grid[0, jj] < .15 and next_robot_pose is True:
            move_robot_joints_to_angles(robot, -30, -130, -90, -130, 60, 0)
            time.sleep(8)

        if jj != N - 1:
            current_robot_pose = next_robot_pose
            next_robot_pose = robot_pose_check(grid[:, jj + 1])
            if current_robot_pose != next_robot_pose:
                change_pose = True
            else:
                change_pose = False

        move_to_single_position(robot, grid[0, jj], grid[1, jj], grid[2, jj], current_robot_pose)
        print("point number: {}, (x,y,z): ({:2f}, {:2f}, {:2f}) ".format(jj,
                                                                         grid[0, jj],
                                                                         grid[1, jj],
                                                                         grid[2, jj]))
        time.sleep(5)
        current_position = robot.get_pose()
        print("\nRobot is at (x_r, y_r, z_r) : ({:2f}, {:2f}, {:2f})".format(current_position.pos[0],
                                                                             current_position.pos[1],
                                                                             current_position.pos[2]))

        print("\npoint reached: ", np.allclose(grid[:1, jj], current_position.pos[:1], rtol=1e-02))
        if not np.allclose(grid[:1, jj], current_position.pos[:1], rtol=1e-02):
            problematic_positions.append(jj)


def plot_grids(grid1, grid2=None, mark_position_grid1=None, mark_position_grid2=None):
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    ax.scatter(grid1[0], grid1[1], grid1[2], marker='d', s=100, color='k')
    ax.view_init(90, 45)
    if mark_position_grid1 is not None:
        ax.text(grid1[0, mark_position_grid1],
                grid1[1, mark_position_grid1],
                grid1[2, mark_position_grid1],
                'number {}'.format(mark_position_grid1))
        ax.scatter(grid1[0, mark_position_grid1],
                   grid1[1, mark_position_grid1],
                   grid1[2, mark_position_grid1], marker='o', s=200, color='red')
    if grid2 is not None:
        ax.scatter(grid2[0], grid2[1], grid2[2], marker='d', s=100, color='g', alpha=0.2)
        if mark_position_grid2 is not None:
            ax.scatter(grid2[0, mark_position_grid2],
                       grid2[1, mark_position_grid2],
                       grid2[2, mark_position_grid2], marker='o', s=200, color='red')

    fig.show()


if __name__ == "__main__":
    l_adapter = 355  # input the length of the rod
    reach = np.ones((3, 1)) * 850 / 1e3
    r_outer = 0.4

    # the offset for the origin of the sampled sphere
    # robot cannot go beyond TCP: y= -1100 mm (1.1 m)
    center = reach[2] + l_adapter / 1e3 - r_outer

    # generate Lebedev grid for sphere (closed form for spherical harmonics)
    tempGridObject = genGrid(86)
    grid = np.array([tempGridObject.x, tempGridObject.y, tempGridObject.z])

    # Spherical coordinate system
    grid_sph = cart2sph(grid[0], grid[1], grid[2])  # phi, theta, r
    grid_sph[2] = grid_sph[2] * r_outer
    # grid_sph = sort_colatitude(grid_sph)

    # Cartesian coordinate system
    gridNew = sph2cart(grid_sph[0], grid_sph[1], grid_sph[2])
    # sort cartesian coordinates of spherical array according to y axis (ascending by default)
    gridNew = sort_ydim(gridNew)

    # connect to robot (insert correct robot IP address, check the 'about' button in \
    # the robot interface home screen
    robot = urx.Robot("192.168.56.102")

    # adjust all coordinates to be centered at 'center'
    gridNew[1] = gridNew[1] - center
    # go to rest position to begin with measurements
    initial_position(robot)
    time.sleep(8)

    """REFERENCE MEASUREMENT [28 x 28] square-like"""
    # reference grid
    X, Y, Z = reference_grid(28, -.35, .35, -.38, .5)
    gridref = np.array([X.ravel(), Y.ravel(), Z.ravel()])
    # sort by ascending y dimension
    gridref = sort_ydim(gridref)
    # center the reference grid to align with sphere
    gridref[1] = gridref[1] - center

    print("\nPress 'Enter' to move in spherical grid...")
    input()
    # move robot in spherical grid
    move_robot_in_grid(robot, gridNew)

    print("\nPress 'Enter' to move in reference grid...")
    input()
    # move robot in reference grid
    move_robot_in_grid(robot, gridref)

    # move robot to single position
    # position = 22
    # move_to_single_position(robot, gridNew[0, position], gridNew[1, position], gridNew[2, position])
