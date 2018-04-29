#Imports
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.optimizers import Adam, Adamax, SGD, RMSprop

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

from keras import backend as K
K.set_image_data_format('channels_first')


ENV_NAME = 'LunarLander-v2'
WINDOW_LENGTH = 1 #we only look ahead by 1 state.



env = gym.make(ENV_NAME)
# Todo: Recording video output throwing an error.
# env = gym.wrappers.Monitor(env, 
#                            'recording', 
#                            resume=False,
#                            force=True,
#                            video_callable=None)
nb_actions = env.action_space.n



model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(32, kernel_initializer='lecun_uniform', activation='relu'))
model.add(Dense(32, kernel_initializer='lecun_uniform', activation='relu'))
model.add(Dense(32, kernel_initializer='lecun_uniform', activation='relu'))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))

memory = SequentialMemory(
    limit=1000000,                 
    window_length=WINDOW_LENGTH) 


#train first with eploration, then expoitation.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),#train starting with low esp, to high esp.
                              attr='eps',        #ammeal this attr
                              value_max=0.4,       #starting_eps
                              value_min=.001,      #ending:eps
                              value_test=.0001,  #test_mode:eps
                              nb_steps=2000000)   #take 1 million steps to slowly alter epsilon.


dqn = DQNAgent(model=model, 
               nb_actions=nb_actions, 
               memory=memory, 
               nb_steps_warmup=1000,     #Run this many before anealing (learn to fall)
               target_model_update=1000, #update model/adjust weights? every 1000 steps?
               policy=policy)

dqn.compile(Adam(lr=.1e-3), metrics=['mse'])
dqn.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))


dqn.test(env, nb_episodes=200, visualize=False, nb_max_episode_steps=2000)
