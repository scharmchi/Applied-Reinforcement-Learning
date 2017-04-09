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

from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')

from lib_robotis_hack import *

D = USB2Dynamixel_Device(dev_name="/dev/ttyUSB0",baudrate=1000000)
s_list = find_servos(D)
s1 = Robotis_Servo(D,s_list[0])
s2 = Robotis_Servo(D,s_list[1])

# go to the wheel mode
s1.init_cont_turn()
s2.init_cont_turn()

# s1.set_angvel(3)
# s2.set_angvel(-3)
# for i in range(600):
#     print(s1.read_angle(), s2.read_angle())
#     time.sleep(0.1)
# s1.set_angvel(0)
# s2.set_angvel(0)
# sys.exit(-1)

# state representation is pos1, pos2, direction1, direction2
# direction 1 means positive vel and -1 os negative velocity (ccw, cw)
observation_examples = np.array([np.append(np.random.uniform(-2.5, 2.5, 2), np.append(np.random.choice([1, -1]), np.random.choice([1, -1]))) for x in range(10000)])
# print(observation_examples)
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=2.0, n_components=20)),
        ("rbf2", RBFSampler(gamma=1.0, n_components=20)),
        ("rbf3", RBFSampler(gamma=0.5, n_components=20))
        ])
featurizer.fit(scaler.transform(observation_examples))


def featurize_state(state):
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    return featurized[0]

# print(featurize_state([0.53, -0.9]))


class PolicyEstimator():
    """
    Policy Function approximator.
    """

    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [60], "state")
            self.actions = tf.placeholder(tf.float32, [2], name="actions")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just linear classifier
            self.mus = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=2,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer())
            # self.mu = tf.squeeze(self.mu)
            self.sigmas = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=2,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer())

            # self.sigma = tf.squeeze(self.sigma)
            self.sigmas = tf.nn.softplus(self.sigmas) + 1e-5
            self.normal_dists = tf.contrib.distributions.Normal(self.mus, self.sigmas)
            self.actions = self.normal_dists.sample(1)
            self.actions = tf.clip_by_value(self.actions, -2, 2)

            # Loss and train op: here my new loss should not be a vector form, just a single number so do a sum
            self.loss = -tf.reduce_sum(self.normal_dists.log_prob(self.actions) * self.target)
            # Add cross entropy cost to encourage exploration
            self.loss -= tf.reduce_sum(1e-1 * self.normal_dists.entropy())

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        return sess.run(self.actions, {self.state: state})

    def update(self, state, target, actions, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        feed_dict = {self.state: state, self.target: target, self.actions: actions}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator():
    """
    Value Function approximator.
    """

    def __init__(self, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [60], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just linear classifier
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
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
        state = featurize_state(state)
        return sess.run(self.value_estimate, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        feed_dict = {self.state: state, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


def actor_critic(estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    for i_episode in range(num_episodes):
        # first action
        state = np.array([s1.read_angle(), s2.read_angle(), 1, -1])
        episode = []
        print("Move me to the start line")
        time.sleep(3)
        # One step in the environment
        for t in itertools.count():
            done = False
            # for some reason the action here looks like [[[ 0.13049328  0.47686869]]]

            actions = estimator_policy.predict(state)
            # print(actions)
            angvel1 = actions[0][0][0] * 2
            angvel2 = actions[0][0][1] * 2

            s1.set_angvel(angvel1)
            s2.set_angvel(angvel2)

            print("angvels", angvel1, angvel2)
            time.sleep(2)  # wait for 1 seconds which is my fixed period for taking the actions while wheeling

            next_state = np.array([0, 0, 0, 0])
            next_state[0] = s1.read_angle()
            next_state[1] = s2.read_angle()

            # ======================================================================
            # time to get the reward which means connecting to pi
            socket_list = [sys.stdin, s]
            # Get the list sockets which are readable
            ready_to_read, ready_to_write, in_error = select.select(socket_list, [], [])
            # sys.stdin = StringIO.StringIO("\n")
            for sock in ready_to_read:
                distance = 100
                if sock == s:
                    # incoming message from remote server, s
                    # print("getting distance from pi..")
                    data = sock.recv(4096)
                    if not data:
                        print('\nDisconnected from chat server')
                        sys.exit()
                    else:
                        try:
                            distance = float(data)
                        except ValueError:
                            print('Not float')
                        # sys.stdout.write('[Me] ');

                else:
                    # user entered a message
                    # msg = sys.stdin.readline()
                    msg = "Thanks, I client got " + str(distance)
                    s.send(msg)
            # print("distance is " + str(distance))
            # ======================================================================
            reward = -distance * 10
            print("reward is " + str(reward))
            # reward = -np.abs(next_state[0] - my_target) - np.abs(next_state[1] - my_target)
            #             print(reward)
            if distance < 4:
                done = True

            # before getting new actions put the previous actions in the current state space to say how we got here
            if angvel1 < 0:
                next_state[2] = -1
            if angvel1 >= 0:
                next_state[2] = 1
            if angvel2 < 0:
                next_state[3] = -1
            if angvel2 >= 0:
                next_state[3] = 1
            episode.append(Transition(
                state=state, action=actions, reward=reward, next_state=next_state, done=done))

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            value_next = estimator_value.predict(next_state)
            td_target = reward + discount_factor * value_next
            td_error = td_target - estimator_value.predict(state)
            estimator_value.update(state, td_target)
            estimator_policy.update(state, td_error, actions)
            print("Step {} @ Episode {}/{} ({})".format(
                t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]))

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
policy_estimator = PolicyEstimator(learning_rate=0.001)
value_estimator = ValueEstimator(learning_rate=0.1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    stats = actor_critic(policy_estimator, value_estimator, 100, discount_factor=0.95)
