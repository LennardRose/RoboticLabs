import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def compute_error_ellipse(position, sigma):

    covariance = sigma[0:2,0:2]
    eigenvals, eigenvecs = np.linalg.eig(covariance)

    #get the largest eigenvalue and corresponding eigenvector
    max_ind = np.argmax(eigenvals)
    max_eigvec = eigenvecs[:,max_ind]
    max_eigval = eigenvals[max_ind]

    #get the smallest eigenvalue and corresponding eigenvector
    min_ind = 0
    if max_ind == 0:
        min_ind = 1

    min_eigvec = eigenvecs[:,min_ind]
    min_eigval = eigenvals[min_ind]

    #chi-square value for sigma confidence interval
    chisquare_scale = 2.2789  

    #compute width and height of confidence ellipse
    width = 2 * np.sqrt(chisquare_scale*max_eigval)
    height = 2 * np.sqrt(chisquare_scale*min_eigval)
    angle = np.arctan2(max_eigvec[1],max_eigvec[0])

    #generate covariance ellipse
    error_ellipse = Ellipse(xy=[position[0],position[1]], width=width, height=height, angle=angle/np.pi*180)
    error_ellipse.set_alpha(0.25)

    return error_ellipse

def plot_slam_state(particles, landmarks, timestamp):
    # Visualizes the state of the FastSLAM algorithm.
    #
    # Displays the particle cloud, the mean position and 
    # the estimated mean landmark positions and covariances.

    draw_mean_landmark_poses = False

    map_limits = [-1, 12, 0, 10]
    
    #particle positions
    xs = []
    ys = []

    #landmark mean positions
    lxs = []
    lys = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])
        
        for i in range(len(landmarks)):
            landmark = particle['landmarks'][i+1]
            lxs.append(landmark['mu'][0])
            lys.append(landmark['mu'][1])

    # ground truth landmark positions
    lx=[]
    ly=[]

    for i in range (len(landmarks)):
        lx.append(landmarks[i+1][0])
        ly.append(landmarks[i+1][1])

    # best particle
    estimated = best_particle(particles)
    robot_x = estimated['x']
    robot_y = estimated['y']
    robot_theta = estimated['theta']

    # estimated traveled path of best particle
    hist = estimated['history']
    hx = []
    hy = []

    for pos in hist:
        hx.append(pos[0])
        hy.append(pos[1])

    # plot FastSLAM state
    plt.clf()

    #particles
    plt.plot(xs, ys, 'r.')
    
    if draw_mean_landmark_poses:
        # estimated mean landmark positions of each particle
        plt.plot(lxs, lys, 'b.')

    # estimated traveled path of best particle
    plt.plot(hx, hy, 'r-')
    
    # true landmark positions
    plt.plot(lx, ly, 'b+',markersize=10)

    # draw error ellipse of estimated landmark positions of best particle 
    for i in range(len(landmarks)):
        landmark = estimated['landmarks'][i+1]

        ellipse = compute_error_ellipse(landmark['mu'], landmark['sigma'])
        plt.gca().add_artist(ellipse)

    # draw pose of best particle
    plt.quiver(robot_x, robot_y, np.cos(robot_theta), np.sin(robot_theta), angles='xy',scale_units='xy')
    
    plt.axis(map_limits)

    #for selected time steps
    if timestamp == 0 or timestamp == 7 or timestamp == 8 or timestamp == 100 or timestamp == 250 or timestamp == 300:
        #output the robot pose and map estimate of the best particle
        print("Robot pose estimate of best particle for time stamp ", timestamp, " is [x, y, theta] = [", "{:.3f}".format(robot_x), "{:.3f}".format(robot_y), "{:.3f}".format(robot_theta), "]")
        #output all estimated mean landmark positions
        for i in range(len(landmarks)):
            landmark = estimated['landmarks'][i+1]              
            if landmark['observed']: 
                print("Estimated mean position of landmark ", i+1, " for time stamp ", timestamp," is [x, y] = [", "{:.3f}".format(landmark['mu'][0]), "{:.3f}".format(landmark['mu'][1]), "]")           
        #save state of the particle filter
        plt.savefig("../output/png/fast_slam_state_for_time_stamp_" + str(timestamp) + ".png")
        plt.savefig("../output/pdf/fast_slam_state_for_time_stamp_" + str(timestamp) + ".pdf")
    
    plt.pause(0.01)

def best_particle(particles):
    #find the particle with the highest weight 

    highest_weight = 0

    best_particle = None
    
    for particle in particles:
        if particle['weight'] > highest_weight:
            best_particle = particle
            highest_weight = particle['weight']

    return best_particle
