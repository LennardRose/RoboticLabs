#!/usr/bin/env python

from read_files import read_world_data, read_sensor_data
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

""" Q 1 -- Particle Filter
"""

# DO NOT DELETE THIS LINE
np.random.seed(123)

# plot settings, interactive plotting mode
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()

# global variable timestamp
timestamp = 0


def plot_filter_state(particles, landmarks, map_limits):
    # Visualizes the particle filter state.
    #
    # Displays the particle cloud, the mean position and the landmarks.

    xs = []
    ys = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

    # landmark positions
    lx = []
    ly = []

    for i in range(len(landmarks)):
        # landmark indices start at one
        lx.append(landmarks[i + 1][0])
        ly.append(landmarks[i + 1][1])

    # mean pose as current estimate
    estimated_pose = mean_pose(particles)

    # plot filter state
    plt.clf()
    plt.plot(xs, ys, 'r.')
    plt.plot(lx, ly, 'bo', markersize=10)
    plt.title(f"Timestep: {timestamp}")
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy',
               scale_units='xy')
    plt.axis(map_limits)

    # for the first fifteen estimates
    if timestamp < 15:
        # output each on the console
        print("The robot pose estimate for time stamp ", timestamp, " is [x, y, theta] = [",
              "{:.3f}".format(estimated_pose[0]), "{:.3f}".format(estimated_pose[1]),
              "{:.3f}".format(estimated_pose[2]), "]")
        # save state of the particle filter
        plt.savefig("./output/pdf/particle_filter_state_for_time_stamp_" + str(timestamp) + ".pdf")
    plt.savefig("./output/png/particle_filter_state_for_time_stamp_" + str(timestamp) + ".png")

    plt.pause(0.01)


def initialize_particles(num_particles, map_limits):
    # randomly initialize the particles inside the map limits

    particles = []

    for i in range(num_particles):
        particle = dict()

        # draw x,y and theta coordinate from uniform distribution
        # inside map limits
        particle['x'] = np.random.uniform(map_limits[0], map_limits[1])
        particle['y'] = np.random.uniform(map_limits[2], map_limits[3])
        particle['theta'] = np.random.uniform(-np.pi, np.pi)

        particles.append(particle)

    return particles


def mean_pose(particles):
    # calculate the average pose of a particle set.
    #
    # for x and y, the average position is the mean of the particle coordinates
    #
    # for theta, we cannot simply average the angles because of the wraparound 
    # (jump from -pi to pi). Therefore, we generate unit vectors from the 
    # angles and calculate the angle of their mean 

    # save x and y particle coordinates
    xs = []
    ys = []

    # save unit vectors corresponding to orientations of particles
    vxs_theta = []
    vys_theta = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

        # create unit vector from particle orientation
        vxs_theta.append(np.cos(particle['theta']))
        vys_theta.append(np.sin(particle['theta']))

    # compute average coordinates
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    mean_theta = np.arctan2(np.mean(vys_theta), np.mean(vxs_theta))

    return [mean_x, mean_y, mean_theta]


def sample_odometry_motion_model(odometry, particles):
    # function according to the slides with particle as x and the odometry measurements as u
    def sample_motion_model(particle):
        # compute new odometry data
        delta_hat_rot1 = delta_rot1 + np.random.normal(0, noise[0] * abs(delta_rot1) + noise[1] * delta_trans)
        delta_hat_trans = delta_trans + np.random.normal(0, noise[2] * delta_trans + noise[3] * (
                    abs(delta_rot1) + abs(delta_rot2)))
        delta_hat_rot2 = delta_rot2 + np.random.normal(0, noise[0] * abs(delta_rot2) + noise[1] * delta_trans)

        # update particle pose parameters based on updated odometry above
        x_new = particle['x'] + delta_hat_trans * np.cos(particle['theta'] + delta_hat_rot1)
        y_new = particle['y'] + delta_hat_trans * np.sin(particle['theta'] + delta_hat_rot1)
        theta_new = particle['theta'] + delta_hat_rot1 + delta_hat_rot2

        # return dict of updated particle
        return {'x': x_new, 'y': y_new, 'theta': theta_new}


    # measurements
    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]

    # generate new particle set after motion update
    new_particles = []

    # iterate over particles
    for particle in particles:
        # sample new particle pose from sample motion model algorithm
        # from slide 13 of lecture "Probabilistic Motion Models - 2"
        # append new particle pose to new particles list
        new_particles.append(sample_motion_model(particle))

    # Output the pose of the first particle drawn from the motion model
    if timestamp == 1:
        print("First particle drawn from the motion model has the pose [x, y, theta] = "
              "[",
              "{:.3f}".format(new_particles[0]['x']),
              "{:.3f}".format(new_particles[1]['y']),
              "{:.3f}".format(new_particles[2]['theta']),
              "]")

    return new_particles


def compute_importance_weights(sensor_data, particles, landmarks):
    # Calculates the observation likelihood of all particles, given the
    # particle and landmark positions and sensor readings
    #
    # The sensor model used is range only.

    sigma_r = 0.2

    # measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']

    weights = []

    for particle in particles:

        # Initialize particles importance weight as 1
        importance_weight = 1

        for id, range in zip(ids, ranges):
            # Calculate expected distance by euclidean distance between landmark and particle
            expected_distance = np.linalg.norm(np.array(landmarks[id])-np.array([particle["x"], particle["y"]])) # euchlidean distance - l2 norm

            # Calculate the likelihood according to 1.b use a gaussian distribution
            likelihood = scipy.stats.norm.pdf(range, expected_distance, sigma_r)

            # update the particles importance weight by the likelihood of the current landmark
            importance_weight *= likelihood

        weights.append(importance_weight)

    # normalize weights
    normalizer = sum(weights)
    weights = weights / normalizer

    # output the weight of the first particle drawn from the motion model
    if timestamp == 1:
        print("First particle drawn from the motion model has the weight w_1 = ", "{:.7f}".format(weights[0]))

    return weights


def resampling(particles, weights):
    # Returns a new set of particles obtained by stochastic
    # universal sampling according to particle weights.

    """
    Line 2
    Initialize new particles and the cummulative sum of the weights
    """
    new_particles = []
    cumsum = [weights[0]]
    n = len(particles)

    """
    Line 3, 4
    compute cummulative sums
    numpy function would have been possible (even faster)
    but I try to stay consistent with the algorithm on slide 8
    """
    for i in range(1, n):
        cumsum.append(cumsum[i - 1] + weights[i])

    """
    Line 5
    Initialize first arrow u with random number r
    reset index to 0, slides start to count at 1, here 0
    """
    # draw random number from ]0,1/N]
    r = np.random.uniform(0, 1 / len(particles))
    # initialize arrow
    u = [r]
    # reset index
    i = 0

    """
    Line 6
    Loop over all particles
    """
    for j in range(n):

        """
        Line 7
        Skip invertals until particle reached (until arrow < cumsum weights)
        """
        while u[j] > cumsum[i]:
            """
            Line 8
            increment to skip
            """
            i += 1
        """
        Line 9
        Save particle as sampled
        """
        new_particles.append(particles[i])

        """
        Line 10
        Increment arrow
        """
        next_u = u[j] + 1/n
        u.append(next_u)

    # Output the pose of the first resampled particle
    if timestamp == 1:
        print("First resampled particle has the pose [x, y, theta] = [", "{:.3f}".format(new_particles[0]['x']),
              "{:.3f}".format(new_particles[1]['y']), "{:.3f}".format(new_particles[2]['theta']), "]")

    return new_particles


def main():
    # implementation of a particle filter for monte carlo localization

    global timestamp

    print("Reading landmark positions")
    landmarks = read_world_data("./data/world_data.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data("./data/sensor_data.dat")

    # initialize the particles
    map_limits = [-1, 12, 0, 10]
    particles = initialize_particles(1000, map_limits)

    # run particle filter for all timestamps
    for timestamp in range(len(sensor_readings) // 2):
        # plot the current state
        plot_filter_state(particles, landmarks, map_limits)

        # predict particles by sampling from motion model with odometry measurements
        new_particles = sample_odometry_motion_model(sensor_readings[timestamp, 'odometry'], particles)

        # compute importance weights according to sensor model
        weights = compute_importance_weights(sensor_readings[timestamp, 'sensor'], new_particles, landmarks)

        # resample new particle set according to importance weights
        particles = resampling(new_particles, weights)

    plt.show()


if __name__ == "__main__":
    main()
