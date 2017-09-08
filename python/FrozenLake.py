#-*- coding: utf-8 -*-
'''
Created on 01.09.2017
@author: steffl&flemmig du spast
'''

import gym
import numpy as np
''' Fuer die Matritzen '''
import random 
''' Fuer die Zufallszahlen'''
import NeuralNetwork as nn

if __name__ == '__main__':
    pass


print("loading Environment..")
env = gym.make('FrozenLake-v0')

print("loading Spaces..")
InputSum = env.observation_space
Inputlenght = 16
''' Discrete(4)'''
''' Das ist die Anzahl der Moeglichen Aktionen'''
''' Discrete = {0,1}'''
OutputSum = env.action_space
Outputlength = 4
''' Discrete(16)'''
''' das ist die Map also der State/Observation'''

print("initializing weights and nodes matrix..")
amountNodesHL = Inputlenght + 1
''' Anzahl der Nodes je Hidden Layer'''
#weights = [np.zeros((amountNodesHL,Inputlenght)),np.zeros((amountNodesHL,amountNodesHL)),np.zeros((Outputlength,amountNodesHL ))]
weights = [np.zeros((amountNodesHL,Inputlenght)),np.zeros((amountNodesHL,amountNodesHL)),np.zeros((Outputlength,amountNodesHL ))]
# weights ist die Matrix mit den Weights[Layer][Ziel][Quelle]
# Sie wird mit den Anzahlen der Nodes initialisiert (InputAmount, InputAmount+1,InputAmount+1,OutputAmount)
nodes = [np.zeros(Inputlenght),np.zeros(amountNodesHL),np.zeros(amountNodesHL),np.zeros(Outputlength)]
absnodes = [np.zeros(Inputlenght),np.zeros(amountNodesHL),np.zeros(amountNodesHL),np.zeros(Outputlength)]
# Nodes ist die Matrix mit den Nodewerten[Layer][Node] 
# Sie wird mit der Anzahl der Nodes initialisiert

print("filling weights with random..")
#init weights
for i in range(3):
    for x in range(len(weights[i])):
        for y in range(len(weights[i][x])):
            weights[i][x][y] = random.uniform(0, 1)
            '''print(weights[i][x][y])'''

    
    
print("weights randomly initliaized")
#print (weights)

# Looooooooooop
episodes = 500

# Variables
lear_rate   = 0.1
muta_rate   = 0.3
gamma       = 0.9
observation = env.reset()

if(nn.debug):
    print("Weights")
    print(weights[0])
    print ()
    print(weights[1])
    print ()
    print(weights[2])
    print ()

nn.Qvalue(weights,nodes,absnodes,observation)
'''
for e in range(episodes):
    observation = env.reset()
    done = False
    moves = 0
    while(done == False):
        moves+=1
        if(random.uniform() <= muta_rate):
            action = env.action_space.sample()
        else:
            action = nn.Qvalue(weights,observation)
        env.step(action)'''