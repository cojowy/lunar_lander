#Imports
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'LunarLander-v2'
WINDOW_LENGTH = 1
record_video_every = 0.001


env = gym.make(ENV_NAME)
# Todo: Recording video output throwing an error.
# env = gym.wrappers.Monitor(env, 
#                            'recording', 
#                            resume=False,
#                            force=True,
#                            video_callable=None)
nb_actions = env.action_space.n


# #build a model.
# model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('linear'))
# print(model.summary())


#build a model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


#memory
memory = SequentialMemory(limit=50000, window_length=WINDOW_LENGTH)

# policy = EpsGreedyQPolicy()
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=1000000)

# dqn = DQNAgent(model=model, 
#                nb_actions=nb_actions, 
#                memory=memory, 
#                nb_steps_warmup=10,
#                target_model_update=1e-2, 
#                policy=policy)

dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               policy=policy,
               memory=memory,
               nb_steps_warmup=10,
               gamma=.99,
               target_model_update=10000,
               train_interval=4,
               delta_clip=1.)


# dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.compile(Adam(lr=.00025), metrics=['mae'])




#start training, visualise slows down.
dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)


#save  model after training.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)


#Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=200, visualize=True)


