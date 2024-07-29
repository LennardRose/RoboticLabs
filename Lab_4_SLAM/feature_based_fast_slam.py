#!/usr/bin/env python

from Lennard_Rose_AIIR_SS23.Lab_4.read_files import read_world_data, read_sensor_data
from Lennard_Rose_AIIR_SS23.Lab_4.plot_and_save import *
import copy
import math
import numpy as np
import scipy.stats

""" T 1 -- Feature-based FastSLAM
"""

# DO NOT DELETE THIS LINE
np.random.seed(123)

# plot settings, interactive plotting mode
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()


def initialize_particles(num_particles, num_landmarks):
    # initialize particle filter with a robot pose and an empty map

    particles = []

    # create one particle at a time
    for i in range(num_particles):
        particle = dict()

        # initialize pose
        particle['x'] = 0
        particle['y'] = 0
        particle['theta'] = 0

        # initial weight
        particle['weight'] = 1.0 / num_particles

        # particle history: modeling the robot's path
        particle['history'] = []

        # initialize landmarks of the particle
        landmarks = dict()

        # create one landmark at a time
        for i in range(num_landmarks):
            landmark = dict()

            # initialize the landmark mean and covariance
            landmark['mu'] = [0, 0]
            landmark['sigma'] = np.zeros([2, 2])
            landmark['observed'] = False

            # landmark indices start at 1
            landmarks[i + 1] = landmark

        # add landmarks to particle
        particle['landmarks'] = landmarks

        # add particle to set
        particles.append(particle)

    return particles


def sample_odometry_motion_model(odometry, particles):
    # Updates the positions of the particles, based on the old positions, odometry
    # measurements and motion noise (see "Take Home Exam / Assignment")

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]

    # compute standard deviations of motion noise
    sigma_delta_rot1 = noise[0] * abs(delta_rot1) + noise[1] * delta_trans
    sigma_delta_trans = noise[2] * delta_trans + noise[3] * (abs(delta_rot1) + abs(delta_rot2))
    sigma_delta_rot2 = noise[0] * abs(delta_rot2) + noise[1] * delta_trans

    # "move" each particle according to the odometry measurements plus sampled noise
    for particle in particles:
        # sample noisy motions
        noisy_delta_rot1 = delta_rot1 + np.random.normal(0, sigma_delta_rot1)
        noisy_delta_trans = delta_trans + np.random.normal(0, sigma_delta_trans)
        noisy_delta_rot2 = delta_rot2 + np.random.normal(0, sigma_delta_rot2)

        # remember last position as part of the path of the particle
        particle['history'].append([particle['x'], particle['y']])

        # compute new particle pose
        particle['x'] = particle['x'] + noisy_delta_trans * np.cos(particle['theta'] + noisy_delta_rot1)
        particle['y'] = particle['y'] + noisy_delta_trans * np.sin(particle['theta'] + noisy_delta_rot1)
        particle['theta'] = particle['theta'] + noisy_delta_rot1 + noisy_delta_rot2

    return


def measurement_prediction_and_jacobian(particle, landmark):
    # Calculate the expected measurement for a landmark
    # and the Jacobian with respect to the landmark.

    px = particle['x']
    py = particle['y']
    ptheta = particle['theta']

    lx = landmark['mu'][0]
    ly = landmark['mu'][1]

    # compute expected range and bearing measurements (see "Probabilistic Sensor Models - 2", slide 25)
    measured_range_exp = np.sqrt((lx - px) ** 2 + (ly - py) ** 2)
    measured_bearing_exp = math.atan2(ly - py, lx - px) - ptheta

    # create vector of expected measurements for Kalman correction
    h = np.array([measured_range_exp, measured_bearing_exp])

    # Calculate the Jacobian H of the measurement function h 
    # wrt the landmark location (see Tutorial 4 - Q 3)
    H = np.zeros((2, 2))
    H[0, 0] = (lx - px) / h[0]
    H[0, 1] = (ly - py) / h[0]
    H[1, 0] = (py - ly) / (h[0] ** 2)
    H[1, 1] = (lx - px) / (h[0] ** 2)

    return h, H


def calculate_landmark_cov(noise, jacobian):
    return np.linalg.inv(jacobian) @ noise @ np.linalg.inv(jacobian).T


def angle_diff(angle1, angle2):
    # Compute the difference between two angles
    # using arctan2 to correctly cope with the signs of the angles
    return np.arctan2(np.sin(angle1 - angle2), np.cos(angle1 - angle2))


def sensor_update(sensor_data, particles):
    # Correct landmark poses with a range and bearing measurement and
    # compute new particle weight

    # Noise of sensor measurements
    Q_t = np.array([[0.1, 0],
                    [0, 0.1]])

    # measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearings = sensor_data['bearing']

    # Perform sensor update for each particle
    """
    Line 2
    Loop over all particles
    """
    for particle in particles:

        """
        Line 5
        landmarks as the observed features
        """
        landmarks = particle['landmarks']

        """
        Line 4
        particle pose parameters
        """
        px = particle['x']
        py = particle['y']
        ptheta = particle['theta']

        """
        Line 5
        landmarks as the observed features
        """
        # loop over observed landmarks
        for i in range(len(ids)):

            """
            Line 4 
            landmark parameters
            """
            # current landmark
            lm_id = ids[i]
            # EKF for current landmark
            landmark = landmarks[lm_id]

            # measured range and bearing to current landmark
            measured_range = ranges[i]
            measured_bearing = bearings[i]

            """
            Line 4
            calculate landmark position from particle pose and the measurement
            """
            lm_x = px + measured_range * np.cos(ptheta + measured_bearing)
            lm_y = py + measured_range * np.sin(ptheta + measured_bearing)

            """
            Line 6
            Never seen landmark
            """
            if not landmark['observed']:

                """
                Line 7
                initialize the landmark position based on the measurement and particle pose
                """
                landmark['mu'] = [lm_x, lm_y]

                """
                Line 8
                calculate jacobian, measurement function h not needed
                """
                _, H = measurement_prediction_and_jacobian(particle=particle,
                                                           landmark=landmark)

                """
                Line 9
                Initialize landmark covariance (uncertainty) 
                """
                landmark['sigma'] = calculate_landmark_cov(noise=Q_t,
                                                         jacobian=H)

                landmark['observed'] = True

            #  landmark was observed before
            else:
                """
                EKF Update
                """
                """
                Line 12 - 13
                measure prediction, calculate jacobian
                """
                h, H = measurement_prediction_and_jacobian(particle=particle,
                                                           landmark=landmark)

                """
                Line 14
                measurement covariance (uncertainty)
                """
                Q = H @ landmark['sigma'] @ H.T + Q_t

                """
                Line 15
                calculate kalman gain
                """
                K = landmark['sigma'] @ H.T @ np.linalg.inv(Q)

                """
                Line 16
                update mean
                """
                predicted_range = h[0]  # readability
                predicted_bearing = h[1]
                landmark['mu'] = landmark['mu'] + K @ np.array([(measured_range - predicted_range),
                                                                    angle_diff(measured_bearing, predicted_bearing)])

                """
                Line 17
                update covariance
                """
                landmark['sigma'] = (np.eye(2) - K @ H) @ landmark['sigma']

                """
                Line 18
                update particle weight / importance factor based on likelihood
                """
                likelihood =  scipy.stats.multivariate_normal.pdf(x=[lm_x, lm_y],
                                                                          mean=landmark['mu'],
                                                                          cov=Q)
                particle['weight'] *= likelihood


    # normalize weights
    normalizer = sum([p['weight'] for p in particles])

    for particle in particles:
        particle['weight'] = particle['weight'] / normalizer


def resampling(particles):
    # Returns a new set of particles obtained by stochastic
    # universal sampling, according to particle weights.
    # (see "Take Home Exam / Assignment")

    # compute distance between pointers
    step = 1.0 / len(particles)

    # random position of first pointer
    u = np.random.uniform(0, step)

    # where we are located along the weights
    c = particles[0]['weight']

    # index of weight container and corresponding particle
    i = 0

    new_particles = []

    # loop over all particle weights
    for particle in particles:

        # go through the weights until you find the particle to which the pointer points
        while u > c:
            i = i + 1
            c = c + particles[i]['weight']

        # add that particle
        new_particle = copy.deepcopy(particles[i])
        new_particle['weight'] = 1.0 / len(particles)
        new_particles.append(new_particle)

        # increment the threshold
        u = u + step

    return new_particles


def main():
    # implementation of feature-based FastSLAM

    print("Reading landmark positions")
    landmarks = read_world_data("data/world_data.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data("data/sensor_data.dat")

    num_particles = 100
    num_landmarks = len(landmarks)

    # create particle set
    particles = initialize_particles(num_particles, num_landmarks)

    # run FastSLAM for all timestamps
    for timestamp in range(len(sensor_readings) // 2):
        # predict particles by sampling from motion model with odometry readings
        sample_odometry_motion_model(sensor_readings[timestamp, 'odometry'], particles)

        # update landmarks and compute particle weights using the measurement model
        sensor_update(sensor_readings[timestamp, 'sensor'], particles)

        # plot the current FastSLAM state
        plot_slam_state(particles, landmarks, timestamp)

        # compute new set of particles with equal weights
        particles = resampling(particles)

    plt.show()


if __name__ == "__main__":
    main()
