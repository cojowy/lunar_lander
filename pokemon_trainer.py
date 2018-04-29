#Imports
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.optimizers import Adam, Adamax, SGD, RMSprop

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'LunarLander-v2'
WINDOW_LENGTH = 1 #we only look ahead by 1 state.
record_video_every = 100

env = gym.make(ENV_NAME)

nb_actions = env.action_space.n

#build a model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(32, kernel_initializer='lecun_uniform', activation='relu'))
model.add(Dense(32, kernel_initializer='lecun_uniform', activation='relu'))
model.add(Dense(32, kernel_initializer='lecun_uniform', activation='relu'))
#model.add(LeakyReLU(alpha=0.3))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))
model.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))

memory = SequentialMemory(
    limit=4000000,                 #Remember previous 1 million states
    window_length=WINDOW_LENGTH)   #Look only 1 state ahead

#train first with eploration, then expoitation.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),#train starting with low esp, to high esp.
                              attr='eps',        #ammeal this attr
                              value_max=0.15,       #starting_eps
                              value_min=.001,      #ending:eps
                              value_test=.001,  #test_mode:eps
                              nb_steps=100000)   #take 1 million steps to slowly alter epsilon.


dqn = DQNAgent(model=model, 
               nb_actions=nb_actions, 
               memory=memory, 
               nb_steps_warmup=1000,     #Run this many before anealing (learn to fall)
               target_model_update=1000, #update model/adjust weights? every 1000 steps?
               policy=policy)

dqn.compile(Adam(lr=.1e-3), metrics=['mse'])

#zero training test
dqn.test(env, nb_episodes=50, visualize=False)

#first fit
dqn.fit(env, 
        nb_steps=100000, 
        verbose=0
           )

#save and test first fit
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
dqn.test(env, nb_episodes=10, visualize=False)
    
    
for i in range(100):
    #reset step count to begin anealing again!
    dqn.policy.agent.steps = 1000
    
    #We can re-paramatarise
    dqn.policy.value_max = 0.1
    dqn.policy.value_min = 0.001
    
    dqn.fit(env, 
        nb_steps=150000, 
        verbose=0
           )
    
    #save current setup
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
    
    #run tests
    dqn.test(env, nb_episodes=50, visualize=False)
    
    
    
    
    
    
