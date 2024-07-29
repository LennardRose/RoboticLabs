def read_world_data(filename):
    # Reads the world definition and returns a list of landmarks, our "map".
    # 
    # The returned dict contains a list of landmarks with the
    # following information: {id, [x, y]}

    # initialize dict
    landmarks = dict()

    # open the file with the provided filename
    f = open(filename)

    # read file line by line
    for line in f:
        # split by new line
        line_s = line.split('\n')
        # split again by space
        line_spl = line_s[0].split(' ')
        # add new key value pair to the dict
        # key is the id of the landmark (first column)
        # value is a list of the x and y coordinates of the landmark (second and thirdcolumn)
        landmarks[int(line_spl[0])] = [float(line_spl[1]), float(line_spl[2])]

    return landmarks


def read_sensor_data(filename):
    # Reads the odometry and sensor readings from a file.
    #
    # The data is returned in a dict where the u_t and z_t are stored
    # together as follows:
    # 
    # {odometry,sensor}
    #
    # where "odometry" contains the fields r1, r2, t which contain the values of
    # the motion model variables of the same name, and sensor is a list of
    # sensor readings with id, range, bearing as values.
    #
    # The odometry and sensor values are accessed as follows:
    # odometry_data = sensor_readings[timestep, 'odometry']
    # sensor_data = sensor_readings[timestep, 'sensor']

    # initialize dict and lists
    sensor_readings = dict()

    lm_ids = []
    ranges = []
    bearings = []

    # flag to skip saving sensordata to dict, because we dont have the sensordata at this step
    first_time = True
    # initialize timestamp to have the sensor readings in order
    timestamp = 0
    # open the file with the provided filename
    f = open(filename)

    # read file line by line
    for line in f:

        # split line by newline and space
        line_s = line.split('\n')
        line_spl = line_s[0].split(' ')

        # line starts with "ODOMETRY" handle odometry data
        if line_spl[0] == 'ODOMETRY':
            # write odometry data from the line to the dict for the current timestamp
            sensor_readings[timestamp, 'odometry'] = {'r1': float(line_spl[1]),
                                                      't': float(line_spl[2]),
                                                      'r2': float(line_spl[3])}
            # when reading the first odometry data we have no sensor data
            if first_time:
                first_time = False
            # when its not the first time we have sensor data and write it to the dict and reset the lists for
            else:
                sensor_readings[timestamp, 'sensor'] = {'id': lm_ids,
                                                        'range': ranges,
                                                        'bearing': bearings}
                lm_ids = []
                ranges = []
                bearings = []

            # increment timestamp only for every odometry line
            timestamp = timestamp + 1

        # line starts with "SENSOR" handle sensor data
        if line_spl[0] == 'SENSOR':
            # append data to the lists
            lm_ids.append(int(line_spl[1]))
            ranges.append(float(line_spl[2]))
            bearings.append(float(line_spl[3]))

        # write the last information we have of the sensor data to the previous timestamp
        sensor_readings[timestamp - 1, 'sensor'] = {'id': lm_ids,
                                                    'range': ranges,
                                                    'bearing': bearings}

    return sensor_readings
