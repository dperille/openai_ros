import rospy
import numpy as np
import time
import math
import os
from gym import spaces
from openai_ros.robot_envs import jackal_robot_env
#from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
from gym.envs.registration import register
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header

# The path is __init__.py of openai_ros, where we import the TurtleBot2MazeEnv directly
timestep_limit_per_episode = 100 # Can be any Value

# TODO - change names
class JackalMazeEnv(jackal_robot_env.JackalEnv):
    def __init__(self):
        """
        This Task Env is designed for having Jackal in some sort of maze.
        It will learn how to move around the maze without crashing.
        """

        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        # This parameter HAS to be set up in the MAIN launch of the AI RL script
        ros_ws_abspath = rospy.get_param("/jackal/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path "+ros_ws_abspath + \
            " DOESNT exist, execute: mkdir -p "+ros_ws_abspath + \
            "/src;cd "+ros_ws_abspath+";catkin_make"

        # TODO: modify this
        # TODO: create jackal_barn.launch and specify a BARN world
        ROSLauncher(rospackage_name="jackal_gazebo",
                    launch_file_name="myworld_launch.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        #LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               #rel_path_from_package_to_file="src/openai_ros/task_envs/jackal/config",
                               #yaml_file_name="jackal_barn.yaml")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(JackalMazeEnv, self).__init__()
        
        ### ACTIONS ###
        # TODO - max acceleration params
        # TODO: see if the params need to be transferred over to 
        # jackal_maze.yaml

        #TODO - actually get params from yaml
        self.min_linear_vel = 0
        #self.max_linear_vel = rospy.get_param('/jackal_velocity_controller/linear/x/max_velocity')
        self.max_linear_vel = 0.5

        self.min_angular_vel = 0
        #self.max_angular_vel = rospy.get_param('/jackal_velocity_controller/angular/z/max_velocity')
        self.max_angular_vel = 0.5

        # Action space is (linear velocity, angular velocity) pair
        self.action_space = spaces.Box(np.array([self.min_linear_vel, self.min_angular_vel]),
                                       np.array([self.max_linear_vel, self.max_angular_vel]),
                                       dtype=np.float32)
        
        
        ### OBSERVATIONS ###
        self.goal_precision = 0.10
        self.move_base_precision = 0.05 #TODO - precision parameters

        # Initial speeds
        self.init_linear_speed = 0
        self.init_angular_speed = 0
        
        # Laser scan parameters
        laser_scan = self._check_laser_scan_ready()
        self.n_laser_scan_values = len(laser_scan.ranges)
        self.max_laser_value = laser_scan.range_max
        self.min_laser_value = laser_scan.range_min
        
        # Pose and goal parameters - TODO (highest values possible for position, goal)
        self.max_odom_x = 10
        self.min_odom_x = -10
        self.max_odom_y = 10
        self.min_odom_y = -5
        self.max_odom_yaw = 3.14
        self.min_odom_yaw = -3.14

        self.max_goal_x = 10
        self.min_goal_x = -10
        self.max_goal_y = 10
        self.min_goal_y = -5
        self.max_goal_yaw = 3.14
        self.min_goal_yaw = -3.14

        self.goal_x = -1 # TODO - get from node
        self.goal_y = -1 # TODO - get from node
        self.goal_yaw = 0 # TODO - get from node

        # Assemble observation space -- [LaserScan vals, | x, y, yaw, | goal_x, goal_y, goal_yaw]
        high_laser = np.full((self.n_laser_scan_values), self.max_laser_value)
        low_laser = np.full((self.n_laser_scan_values), self.min_laser_value)

        high_odom = np.array([self.max_odom_x, self.max_odom_y, self.max_odom_yaw])
        low_odom = np.array([self.min_odom_x, self.min_odom_y, self.min_odom_yaw])

        high_goal = np.array([self.max_goal_x, self.max_goal_y, self.max_goal_yaw])
        low_goal = np.array([self.min_goal_x, self.min_goal_y, self.min_goal_yaw])

        high = np.concatenate([high_laser, high_odom, high_goal])
        low = np.concatenate([low_laser, low_odom, low_goal])

        self.observation_space = spaces.Box(low, high)


        ### REWARDS
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)

        # TODO - get reward params
        self.step_penalty = -1      # penalty for each step without reaching goal
        self.fail_penalty = -20     # penalty when episode finish without reaching goal
        self.goal_reward = 50       # reward for reaching goal

        self.cumulated_steps = 0.0
        self.cumulated_reward = 0.0
        
        

        # rospy.logdebug("")
        # self.laser_filtered_pub = rospy.Publisher('/turtlebot2/laser/scan_filtered', LaserScan, queue_size=1)

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        epsilon=0.05,
                        update_rate=10,
                        min_laser_distance=-1)

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        
        # We wait a small ammount of time to start everything because in very fast resets, laser scan values are sluggish
        # and sometimes still have values from the prior position that triguered the done.
        time.sleep(1.0)

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of Jackal.
        :param action: The action integer that set s what movement to do next.
        """
        
        # Action is (linear velocity, angular velocity) pair
        linear_velocity = action[0]
        angular_velocity = action[1]
        rospy.logdebug("Set Action ==> " + str(linear_velocity) + ", " + str(angular_velocity))

        # Check if larger than max velocity
        # TODO - check if acceleration is greater than allowed
        self.move_base(linear_velocity, angular_velocity, epsilon=self.move_base_precision, update_rate=10)

    def _get_obs(self):
        """
        Get the current observations -- laser scan data, odometry position, goal position.
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        
        # Get laser scan data
        laser_scan = self.get_laser_scan().ranges

        # Get odometry data
        odometry = self.get_odom()
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y

        roll, pitch, yaw = self.get_orientation_euler()
        odometry_array = [x_position, y_position, yaw] # TODO - round?

        # Get goal position
        desired_position = [self.goal_x, self.goal_y, self.goal_yaw] # TODO - round?

        # Concatenate observations
        observations = laser_scan + odometry_array + desired_position

        rospy.logdebug("END Get Observation ==>")

        return observations
        

    def _is_done(self, observations):
        """
        Episode is done if:
        1) Jackal is outside world boundaries
        2) Jackal is too close to an obstacle
        3) Jackal has reached the goal
        """

        # get current [x, y, yaw]
        current_position = [observations[-6], observations[-5], observations[-4]]

        is_done = self._is_outside_boundaries(current_position) or self._has_crashed(current_position) or self._reached_goal(current_position, self.goal_precision)

        rospy.logdebug("_IS_DONE? ==> " + str(is_done))

        return is_done

    def _compute_reward(self, observations, done):
        """
        Simple reward initially: small negative reward every
        step and on failure, big positive reward for reaching goal
        """

        reward = 0.0
        current_position = [observations[-6], observations[-5], observations[-4]]
        goal_position = [observations[-3], observations[-2], observations[-1]]

        if done:
            # If done and reached goal (success), big positive reward
            if self._reached_goal(current_position, goal_position, self.goal_precision):
                reward = self.goal_reward

            # If done because of failure, negative reward
            else:
                reward = self.fail_penalty

        else:
            reward = self.step_penalty

        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        
        return reward


    # Internal TaskEnv Methods

    def get_orientation_euler(self):
        # We convert from quaternions to euler
        orientation_list = [self.odom.pose.pose.orientation.x,
                            self.odom.pose.pose.orientation.y,
                            self.odom.pose.pose.orientation.z,
                            self.odom.pose.pose.orientation.w]
    
        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw

    # current_position: list of [x_pos, y_pos, yaw]
    def _is_outside_boundaries(self, current_position):

        x_pos = current_position[0]
        y_pos = current_position[1]

        if x_pos < self.min_odom_x or x_pos > self.max_odom_x:
            return True
        elif y_pos < self.min_odom_y or y_pos > self.max_odom_y:
            return True
        else:
            return False

    # return true if jackal has crashed into an obstacle
    def _has_crashed(self, laser_readings):

        for dist in laser_readings:
            if dist <= self.min_laser_value:
                rospy.logdebug("JACKAL CRASHED")
                return True

        return False

    # return true if jackal is within epsilon of the goal position
    def _reached_goal(self, current_position, goal_position, epsilon=0.1):

        x_pos = current_position[0]
        y_pos = current_position[1]

        x_goal = goal_position[0]
        y_goal = goal_position[1]

        return math.sqrt( (x_pos - x_goal) ** 2 + (y_pos - y_goal) ** 2) < epsilon



    # Formerly here but unused:
    # def discretize_scan_observation(self,data,new_ranges):
    # def update_desired_pos(self,new_position):
    # def publish_filtered_laser_scan(self, laser_original_data, new_filtered_laser_range):