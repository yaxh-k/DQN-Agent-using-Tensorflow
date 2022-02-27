import gym
from keras import models
from keras import layers
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import numpy as np
import tensorflow as tf
import keras
import tensorflow._api.v2.compat.v1 as tf


config = tf.compat.v1.ConfigProto( device_count = {'GPU': 0} ) 

tf.disable_v2_behavior() 




sess = tf.Session(config=config) 
keras.backend.set_session(sess)





class MountainCarTrain:




    #initializing function
    def __init__(self,env):

            self.env=env
            self.gamma=0.99
            self.epsilon = 1
            self.epsilon_decay = 0.05
            self.epsilon_min=0.01
            self.learingRate=0.001
            self.replayBuffer=deque(maxlen=20000)
            self.trainNetwork=self.createNetwork()
            self.episodeNum=400
            self.iterationNum=201 #max is 200
            self.numPickFromBuffer=32
            self.targetNetwork=self.createNetwork()
            self.targetNetwork.set_weights(self.trainNetwork.get_weights())









    #1 neural lad
    def createNetwork(self):

            model = models.Sequential()
            state_shape = self.env.observation_space.shape

            model.add(layers.Dense(24, activation='selu', input_shape=state_shape))
            model.add(layers.Dense(48, activation='selu'))
            model.add(layers.Dense(48, activation='selu'))
            model.add(layers.Dense(self.env.action_space.n,activation='linear'))

            model.compile(loss='mse', optimizer=Adam(lr=self.learingRate))

            return model











    #Epsilon Greedy w/ decay function
    def getBestAction(self,state):

            self.epsilon = max(self.epsilon_min, self.epsilon)

            if np.random.rand(1) < self.epsilon:
                action = np.random.randint(0, 3)        #random 0-1 (exploration)         
            else:
                action=np.argmax(self.trainNetwork.predict(state)[0])           #exploiatation    MAX value of Q

            return action

        












    def trainFromBuffer_Boost(self):
            if len(self.replayBuffer) < self.numPickFromBuffer:
                return
            samples = random.sample(self.replayBuffer,self.numPickFromBuffer) #Sampling from buffer  (train sample of previous learning )
            npsamples = np.array(samples)
            states_temp, actions_temp, rewards_temp, newstates_temp, dones_temp = np.hsplit(npsamples, 5)
            states = np.concatenate((np.squeeze(states_temp[:])), axis = 0) #states from replay memory
            rewards = rewards_temp.reshape(self.numPickFromBuffer,).astype(float)
            targets = self.trainNetwork.predict(states)
            newstates = np.concatenate(np.concatenate(newstates_temp))
            dones = np.concatenate(dones_temp).astype(bool) #Check if the state is terminal state
            notdones = ~dones
            notdones = notdones.astype(float)
            dones = dones.astype(float)
            Q_futures = self.targetNetwork.predict(newstates).max(axis = 1) #Q-values  [A Q-value is a p-value that has been adjusted for the False Discovery Rate(FDR)]
            targets[(np.arange(self.numPickFromBuffer), actions_temp.reshape(self.numPickFromBuffer,).astype(int))] = rewards * dones + (rewards + Q_futures * self.gamma)*notdones
            self.trainNetwork.fit(states, targets, epochs=1, verbose=0)











    def orginalTry(self,currentState,eps):
            rewardSum = 0
            max_position=-99


            for i in range(self.iterationNum):
                bestAction = self.getBestAction(currentState)


                #show the animation every 10 eps
                if eps%10==0:
                    env.render()

                new_state, reward, done, _ = env.step(bestAction)

                new_state = new_state.reshape(1, 2)


                # # Keep track of max position
                if new_state[0][0] > max_position:
                    max_position = new_state[0][0]



                # # Adjust reward for task completion
                if new_state[0][0] >= 0.5:
                    reward += 10



                self.replayBuffer.append([currentState, bestAction, reward, new_state, done])


                #God Function
                self.trainFromBuffer_Boost()

                rewardSum += reward
                currentState = new_state


                if done:
                    break



            if i >= 199:
                print("Failed to finish task in epsoide {}".format(eps))
            else:
                print("Success in epsoide {}, used {} iterations!".format(eps, i))
                self.trainNetwork.save('./trainNetworkInEPS{}.h5'.format(eps))











            #Sync
            self.targetNetwork.set_weights(self.trainNetwork.get_weights())

            print("now epsilon is {}, the reward is {} maxPosition is {}".format(max(self.epsilon_min, self.epsilon), rewardSum,max_position))
            self.epsilon -= self.epsilon_decay
















    def start(self):
            for eps in range(self.episodeNum):
                currentState=env.reset().reshape(1,2)
                self.orginalTry(currentState, eps)











env = gym.make('MountainCar-v0')
dqn=MountainCarTrain(env=env)
dqn.start()