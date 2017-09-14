#-*- coding: utf-8 -*-
'''
Created on 01.09.2017
@author: steffl&flemmig 
'''

import gym
import numpy as np                 #Notwendig für die Matritzenrechnung
import random                     #Notwendig fpr die gernerierung von Zufallszahlen
import NeuralNetwork as nn        #Beinhaltet das Neurale Netz
import math as m


# Wenn Done und Reward=0 -> Loch dann ist Reward -1
# Wenn Done und Reward=1 -> Loch dann ist Reward 1
# sonst 0
# Da GameHistorie übergeben wird wenn am Ende Loch wird alles mit reward -0,1 berechnet... -0,1 wird eine Variable sein
# Da GameHistorie übergeben wird wenn am Ende Win wird alles mit reward 0,1 berechnet... 0,1 wird eine Variable sein 

# 1.1 Loss für einzelne Aktionen in History berechnen und anpassen
# Idee von Andi
def loss(GameHist):
    for idxx, zug in enumerate(GameHist):
                if (idxx == len(GameHist)-1):
                    if(zug[len(zug)-2] == 1.0):
                        for idx, zuege in enumerate(GameHist):
                            if(idx != len(GameHist)-1):
                                if(zuege[len(zuege)-2] == 0):
                                    zuege[len(zuege)-2]=0.1 # ug[len(zug)-2]/10
                    else:
                        for idx, zuege in enumerate(GameHist):
                            if(idx != len(GameHist)-1 ):
                                if(zuege[len(zuege)-2] == 0):
                                    zuege[len(zuege)-2]=-0.1
                            else:
                                zuege[len(zuege)-2]=-1
    #print("Rewards angepasst")
    #nn.printHistory(GameHist)
# Idee von Eddi 
# 1.2 Gamma von i Berechnen = -L* Abl. von Aktivierungsfk(von Array tmp1) von Hidden Layers und Output
def ouputcalc(GameHist):  # absnodes für die Größe des Arrays
    for entry in GameHist:
        absnodes = entry[4]
    # Ziel: alle Absnodes zwischenspeicher... also großes Array mit 1 dim: länge der Game hist und ab der 2ten Dim abs nodes kopieren
    
	#LÄUFT [EDDIE]
    BpropNodes = [[np.zeros(nn.Inputlenght),np.zeros(nn.amountNodesHL),np.zeros(nn.amountNodesHL),np.zeros(nn.Outputlength)] for GH in range(len(GameHist))]
    for GameIndex in range(len(GameHist)):
        LayerIndex = 0
        while(LayerIndex < len(BpropNodes[GameIndex])):
            NodeIndex = 0
            while(NodeIndex < len(BpropNodes[GameIndex][LayerIndex])):
                BpropNodes[GameIndex][LayerIndex][NodeIndex] = absnodes[LayerIndex][NodeIndex] 
                NodeIndex+=1
            LayerIndex+=1
    #LÄUFT [EDDIE]
	
    for index, entry in enumerate(GameHist):
        for i in range(len(BpropNodes[index][2])):
            # wichtig: nur bei dem genommenen den Loss betrachen! alle anderen sind 0
            if(entry[1] == np.argmax((entry[len(entry)-1])[2][i])):
                BpropNodes[index][2][i] = entry[len(entry)-2] * arctanh((entry[len(entry)-1])[2][1])
    print("BpropNodes")
    print(BpropNodes)
    return None


# Ableitung von tanh
def arctanh(x): return 0.5*m.log2((1+x)/(1-x))
     
# 2.1 Werte der einzelnen Nodes Berechnen  anhand  der ausgehenden weights + des nächsten Nodes  -> in neue Matrix speichern tmp3

# 2.2 weights berechnen und änderung in Matrix schreiben

# 3 Backprop anwenden!