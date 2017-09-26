#-*- coding: utf-8 -*-
'''
Created on 01.09.2017
@author: steffl&flemmig 
Best 32 aus 2000 l= 0.05 f=0.7 1.5%
2.   61 aus 5000 l= 0.05 f=0.7 1.2%
3.   54 aus 5000 l= 0.1 f=0.7 1.08%
4.   35 aus 5000 l= 0.8 f=0.7
5.   50 aus 5000 l=0.05 f=0.7
'''

import gym
import random                     #Notwendig fpr die gernerierung von Zufallszahlen
import NeuralNetwork as nn        #Beinhaltet das Neurale Netz
import datetime

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

    
lear_rate       = 0.05
Zukunft         = 1
future_rate     = 0.7
Ep = 20000
file = "resultnn" + str(datetime.date.today()) + ".csv"
print("in File: " + file)
weightssave=weights
#for i in range(1,15):
    #future_rate     = 0.7
    #for j in range(1,15):
        #print("Durchlauf ",i,"/",j," von 20/30")
nn.learn(env,Ep,weights, History, lear_rate, Zukunft, future_rate, i, file)
        #weights=weightssave
        #future_rate= future_rate - 0.01
    #lear_rate=lear_rate-0.01
#print("ENDE")

#ToDo:
#weights davor und danach ausgeben lassen und selbst aussrechnen ob richtig verbessert wird
# Problem: solange muta_rate hoch is steigt die Win Anzahl, wenns low wird dann bleiben die Wins komplett aus!
# Idee: Weights werden nicht, oder nur schlecht angepasst! oder falsch angepasst
#die besten weiterlaufen lassen
#Loss beispiel berechnen per hand
#mehrere Threads aufrufbar mit verschiedenen learn rates 
#gui etc, env besser sichtbarmachen!
