from __future__ import print_function
import itertools
import matplotlib
import numpy as np
import tensorflow as tf
import collections
import sys
import socket
import select
import sklearn.pipeline
import sklearn.preprocessing

if "../" not in sys.path:
  sys.path.append("../") 
import plotting
matplotlib.style.use('ggplot')
from lib_robotis_hack import *

D = USB2Dynamixel_Device(dev_name="/dev/ttyUSB1",baudrate=1000000)
s_list = find_servos(D)
s1 = Robotis_Servo(D,s_list[0])
s2 = Robotis_Servo(D,s_list[1])

# go to the wheel mode
s1.init_cont_turn()
s2.init_cont_turn()

observation_space_size = 20  # tile coding of size 20
action_space_size = 2  # go left or right


class PolicyEstimator():

    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], "state")
            self.action = tf.placeholder(dtype=tf.int32, name="action")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just table lookup estimator
            state_one_hot = tf.one_hot(self.state, int(observation_space_size))
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=action_space_size,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer())

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            # Loss and train op
            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target, self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator():

    def __init__(self, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just table lookup estimator
            state_one_hot = tf.one_hot(self.state, int(observation_space_size))
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer())

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


def actor_critic(estimator_policy, estimator_value, num_episodes, discount_factor=1.0):

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    min_angle = -2.5830872929516078
    max_angle = 2.5830872929516078
    angle_range = max_angle - min_angle
    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = int(((s1.read_angle() + max_angle) / angle_range) * 20)
        distance = 30

        episode = []

        s1.set_angvel(0)
        s2.set_angvel(0)
        # fake_distance = 30
        print("Move me to the start line")
        time.sleep(10)

        # One step in the environment
        for t in itertools.count():

            # Take a step
            action_probs = estimator_policy.predict(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            if action == 0:
                s1.set_angvel(3)
                s2.set_angvel(-3)
            if action == 1:
                s1.set_angvel(-3)
                s2.set_angvel(3)

            time.sleep(3)

            # ======================================================================
            # time to get the reward which means connecting to pi
            socket_list = [sys.stdin, s]
            # Get the list sockets which are readable
            ready_to_read, ready_to_write, in_error = select.select(socket_list, [], [])
            for sock in ready_to_read:
                if sock == s:
                    # incoming message from remote server, s
                    data = sock.recv(4096)
                    if not data:
                        print('\nDisconnected from chat server')
                        sys.exit()
                    else:
                        try:
                            distance = float(data)
                        except ValueError:
                            print('Not float')
                else:
                    # user entered a message
                    msg = "Thanks, I client got " + str(distance)
                    s.send(msg)
            # ======================================================================
            # if action == 0:
            #     distance += 4
            # if action == 1:
            #     distance -= 4
            reward = -distance * 10
            # wtf is wrong with those big numbers
            if reward < -500:
                reward = -500
            print("reward is " + str(reward))

            done = False
            if distance < 12:
                done = True
                # reward = 1000  # DOES GIVING A HUGE REWARD AT THE END AFFECT ANYTHING??
                s1.set_angvel(0)
                s2.set_angvel(0)

            next_state = int(((s1.read_angle() + max_angle) / angle_range) * 20)  # mapping the state to 0-19 for tile coding

            # Keep track of the transition
            episode.append(Transition(
                state=state, action=action, reward=reward, next_state=next_state, done=done))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # Calculate TD Target
            value_next = estimator_value.predict(next_state)
            td_target = reward + discount_factor * value_next
            td_error = td_target - estimator_value.predict(state)

            # Update the value estimator
            estimator_value.update(state, td_target)

            # Update the policy estimator
            # using the td error as our advantage estimate
            estimator_policy.update(state, td_error, action)

            # Print out which step we're on, useful for debugging.
            print("Step {} @ Episode {}/{} ({})".format(
                t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]))
            if t == 0 and done:
                done = False
            if done:
                break

            state = next_state

    return stats

host = "192.168.0.42"
port = 9009

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(5)

# connect to remote host
try:
    s.connect((host, port))
except:
    print('Unable to connect')
    sys.exit()

print('Connected to remote host. You can start sending messages')

tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator()
value_estimator = ValueEstimator()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary.
    stats = actor_critic(policy_estimator, value_estimator, 30)

plotting.plot_episode_stats(stats, smoothing_window=10)
