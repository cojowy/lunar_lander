import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.optimizers import Adam, Adamax, SGD, RMSprop

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy,  LinearAnnealedPolicy
from rl.memory import SequentialMemory

import io
import sys
import csv

# Path environment changed to make things work properly
# export DYLD_FALLBACK_LIBRARY_PATH=$DYLD_FALLBACK_LIBRARY_PATH:/usr/lib


# Get the environment and extract the number of actions.
ENV_NAME = 'LunarLander-v2'
WINDOW_LENGTH = 1 # we only look ahead by 1 state.
record_video_every = 100

# setting up Lunar Lander environment and the corresponding gameplay actions
env = gym.make(ENV_NAME)
nb_actions = env.action_space.n


#build a model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(32, kernel_initializer='lecun_uniform', activation='relu'))
model.add(Dense(32, kernel_initializer='lecun_uniform', activation='relu'))
model.add(Dense(32, kernel_initializer='lecun_uniform', activation='relu'))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(
    limit=1000000,                 #Remember previous 1 million states
    window_length=WINDOW_LENGTH) 

#train first with eploration, then expoitation.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),#train starting with low esp, to high esp.
                              attr='eps',        #anneal this attr
                              value_max=0.6,       #starting_eps
                              value_min=.001,      #ending_eps
                              value_test=.0001,  #test_mode_
                              nb_steps=2000000)   #take 2 million steps to slowly alter epsilon.

dqn = DQNAgent(model=model, 
               nb_actions=nb_actions, 
               memory=memory, 
               nb_steps_warmup=1000,     #Run this many before anealing (learn to fall)
               target_model_update=1000, #update model/adjust weights? every 1000 steps?
               policy=policy)

dqn.compile(Adam(lr=.1e-3), metrics=['mse'])

train_step_number=900000
# After training is done, we save the final weights.
dqn.load_weights('../weights/TESTpoint6eps2mil_weights_'+str(train_step_number)+'_.h5f')

# Redirect stdout to capture test results
old_stdout = sys.stdout
sys.stdout = mystdout = io.StringIO()

# Evaluate our algorithm for a few episodes.
dqn.test(env, nb_episodes=200, visualize=False)

# Reset stdout
sys.stdout = old_stdout

results_text = mystdout.getvalue()

# Print results text
print("results")
print(results_text)

# Extact a rewards list from the results
total_rewards = list()
for idx, line in enumerate(results_text.split('\n')):
    if idx > 0 and len(line) > 1:
        reward = float(line.split(':')[2].split(',')[0].strip())
        total_rewards.append(reward)

# Print rewards and average	
print("total rewards", total_rewards)
print("average total reward", np.mean(total_rewards))

# Write total rewards to file
f = open("lunarlander_rl_rewards_step"+str(train_step_number)+".csv",'w')
wr = csv.writer(f)
for r in total_rewards:
     wr.writerow([r,])
f.close()