from move_ur5_in_grid import cart2sph, sph2cart
import math3d as m3


def gen_orientation(points, reference):
    """
    for each point returns orientation in [rx,ry,rz] away from reference
    """
    vec = points-reference
    return vec/np.linalg.norm(vec,axis=0)

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    tool_length = 3
    max_angle_deg=60
    # for long tools, all base points should be on the x-axis

    #generate point cloud
    # radius = .5
    points = np.random.randn(3,20)

    # points to ref, along x-axis
    ref = np.zeros_like(points)
    # find x-axis combination to have close movements
    # relate distance from target (x-axis) to tool_length
    distance_from_trace = np.linalg.norm(points[1:],axis=0)
    distance_from_trace = np.clip(distance_from_trace,0,tool_length)
    angle = np.arcsin(distance_from_trace/tool_length)
    angle = np.clip(angle,-max_angle_deg/180*np.pi,+max_angle_deg/180*np.pi)
    ref[0] = points[0]-tool_length*np.cos(angle)
    print(angle/np.pi*180)


    orientations = gen_orientation(points,ref)
    base = points-orientations*tool_length

    # order points along x, according to their base position
    idx = np.argsort(base[0])
    points = points[:,idx]
    ref = ref[:,idx]
    orientations = orientations[:,idx]
    base = base[:,idx]

    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111,projection = '3d')
    ax.scatter(*points)
    ax.plot3D(*base,'r')
    ax.quiver(*base, *orientations, length=tool_length, normalize=True,alpha=.3)
    # ax.scatter(*ref) # ref is tool base projection onto target trace
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.draw()
    plt.show()

