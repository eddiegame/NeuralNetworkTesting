#-*- coding: utf-8 -*-
'''
Created on 01.09.2017
@author: steffl&flemmig 
'''

import gym
import random                     #Notwendig fpr die gernerierung von Zufallszahlen
import NeuralNetwork as nn        #Beinhaltet das Neurale Netz


if __name__ == '__main__':
    pass

# Das Environment vom openai gym wird geladen (hier FrozenLake)
print("loading Environment..")
env = gym.make('FrozenLake-v0')

# Die Spaces also die Anzahl verschiedener 
# Observations und Actions wird aus dem Environment geladen
print("loading Spaces..")
InputSum = env.observation_space
OutputSum = env.action_space


print("initializing weights and nodes matrix..")

# Die Weight Matix wird initialisiert
weights = nn.getZeroWeights()

# Die Nodematrix enthält die Nodewerte welche durch die Sigmoid Funktion normalisiert wurden
# nodes[Layer][Node]
nodes = nn.getZeroNodes()

# Die AbsNodes Matrix enthält alle Nodewerte vor der Normalisierung mit der Sigmoid Funktion
# absnodes[Layer][Node]
absnodes = nn.getZeroNodes()

# Die History Matrix enthält alle "gespielten" Actions, also Welche Action in welcher Observation wie oft vor kam
History = []




# Die Weight Matrix wird mit Zufallszahlen zwischen 0 und 1 gefüllt
# falls diese Später abgespeichert werden sollte muss sie hier eingelesen werden
print("filling weights with random..")
for i in range(3):
    for x in range(len(weights[i])):
        for y in range(len(weights[i][x])):
            weights[i][x][y] = random.uniform(0, 0.5)


    
print("weights randomly initliaized")

# Hier beginnt der main Loop

# Ein Debug print:
if(nn.debug):
    print("Weights")
    print(weights[0])
    print ()
    print(weights[1])
    print ()
    print(weights[2])
    print ()
    
# Training Loop, Ep gibt die Anzahl der zu durchlaufenden Episoden an
Ep = 100
nn.learn(env,Ep,weights, History)


