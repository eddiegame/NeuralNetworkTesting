#-*- coding: utf-8 -*-
'''
Created on 01.09.2017
@author: steffl&flemmig du spast asdf dfg
'''

import gym
import numpy as np 				#Notwendig für die Matritzenrechnung
import random 					#Notwendig fpr die gernerierung von Zufallszahlen
import NeuralNetwork as nn		#Beinhaltet das Neurale Netz

if __name__ == '__main__':
    pass

# Das Environment vom openai gym wird geladen (hier FrozenLake)
print("loading Environment..")
env = gym.make('FrozenLake-v0')

# Die Spaces also die Anzahl verschiedener 
# Observations und Actions wird aus dem Environment geladen
print("loading Spaces..")
InputSum = env.observation_space
Inputlenght = 16				#Anzahl der Inputs (Observation)
OutputSum = env.action_space
Outputlength = 4				#Anzahl der Output (Actions)


print("initializing weights and nodes matrix..")
amountNodesHL = Inputlenght + 1
# Die Weight Matix wird initialisiert
# Weights[Layer][Ziel][Quelle]
# weights = [np.zeros((amountNodesHL,Inputlenght)),np.zeros((amountNodesHL,amountNodesHL)),np.zeros((Outputlength,amountNodesHL ))]
weights = [np.zeros((amountNodesHL,Inputlenght)),np.zeros((amountNodesHL,amountNodesHL)),np.zeros((Outputlength,amountNodesHL ))]

# Die Nodematrix enthält die Nodewerte welche durch die Sigmoid Funktion normalisiert wurden
# nodes[Layer][Node]
nodes = [np.zeros(Inputlenght),np.zeros(amountNodesHL),np.zeros(amountNodesHL),np.zeros(Outputlength)]

# Die AbsNodes Matrix enthält alle Nodewerte vor der Normalisierung mit der Sigmoid Funktion
# absnodes[Layer][Node]
absnodes = [np.zeros(Inputlenght),np.zeros(amountNodesHL),np.zeros(amountNodesHL),np.zeros(Outputlength)]

# Die History Matrix enthält alle "gespielten" Actions, also Welche Action in welcher Observation wie oft vor kam
history = []




# Die Weight Matrix wird mit Zufallszahlen zwischen 0 und 1 gefüllt
# falls diese Später abgespeichert werden sollte muss sie hier eingelesen werden
print("filling weights with random..")
for i in range(3):
    for x in range(len(weights[i])):
        for y in range(len(weights[i][x])):
            weights[i][x][y] = random.uniform(0, 1)
            '''print(weights[i][x][y])'''

    
print("weights randomly initliaized")

# Hier beginnt der main Loop
# Variablen:
episodes = 500
lear_rate   = 0.1
muta_rate   = 0.3
gamma       = 0.9
observation = env.reset()
done = False

# Ein Debug print:
if(nn.debug):
    print("Weights")
    print(weights[0])
    print ()
    print(weights[1])
    print ()
    print(weights[2])
    print ()
	
# Training Loop
for e in range(episodes):
	while(done == False):
		# Das Environemnt wird dargestellt
		env.render()
		# Die letzte observation wird gespeichert
		oldObs = observation
		# Das Neurale Netz ermittelt die Action mit dem höchsten Q-Wert für den aktuellen Stand
		NextAction = nn.Qvalue(weights,nodes,absnodes,observation)
		# Die beste Action wird ausgeführt
		observation, reward, done, info = env.step(NextAction)
		# Die durchgeführte Action wird gespeichert
		found = False
		
		for idx, entry in enumerate(history):
			# print("suche: " + str(oldObs) + "|" + str(NextAction) + "|" + str(observation) + " in history: " + entry[0] + "|" + entry[1] + "|" + entry[2])
			if oldObs==entry[0] and NextAction==entry[1] and observation==entry[2]:
				print(str(entry) + " already exist! -> count++")
				entry[3] = int(entry[3])+1
				print("idx: " + str(idx))
				history[idx] = entry
				print("hist bei idx" + str(history[idx]))
				found = True
		if found== False:
			newEntry = [oldObs,NextAction,observation,1]
			history.append(newEntry)
			sorted(history, key=lambda x: x[0])
		env.render() #warum doppelt?
    
# Die history wird ausgegeben
    print(history)
