#-*- coding: utf-8 -*-
'''
Created on 01.09.2017
@author: steffl&flemmig 
'''

import numpy as np                 #Notwendig für die Matritzenrechnung
import NeuralNetwork as nn        #Beinhaltet das Neurale Netz
import math


# Wenn Done und Reward=0 -> Loch dann ist Reward -1
# Wenn Done und Reward=1 -> Loch dann ist Reward 1
# sonst 0
# Da GameHistorie übergeben wird wenn am Ende Loch wird alles mit reward -0,1 berechnet... -0,1 wird eine Variable sein
# Da GameHistorie übergeben wird wenn am Ende Win wird alles mit reward 0,1 berechnet... 0,1 wird eine Variable sein 

# 1.1 Loss für einzelne Aktionen in History berechnen und anpassen
# Idee von Andi
def loss(ZugReverse):
        #if(ZugReverse[len(ZugReverse)-1] == 1.0):
        ZugReverse[len(ZugReverse)-2] = 0.1 #0.5*(ZugReverse[len(ZugReverse)-2])
        return ZugReverse


    #print("Rewards angepasst")
    #nn.printHistory(GameHist)
# Idee von Eddi 
# 1.2 Gamma von i Berechnen = -L* Abl. von Aktivierungsfk(von Array tmp1) von Hidden Layers und Output
def createBpropNodes(GameHistlength): 
    BpropNodes = [[np.zeros(nn.Inputlenght),np.zeros(nn.AmountNodesHL),np.zeros(nn.AmountNodesHL),np.zeros(nn.Outputlength)] for g in range(GameHistlength)]
    return BpropNodes

def ouputcalc(BpropNodes ,ZugReverse, Index, weights):  
    for i in range(nn.Outputlength):
        print("Entry : ",ZugReverse[1],"  i : ",i)          
        if(ZugReverse[1] == i):    
            print("Index : ", Index)
            #print("Entry: ",ZugReverse[len(ZugReverse)-2], " mal ",arcsigmoid((ZugReverse[len(ZugReverse)-1])[3][i]))
            absnodes = NodesCalc(ZugReverse[0], weights, True)  
            BpropNodes[Index][3][i] = ZugReverse[len(ZugReverse)-2] * arcsigmoid(absnodes[3][i])
            break
    return BpropNodes


def sigmoid(x): return 1 / (1 + np.exp(-x))
def arcsigmoid(x): return sigmoid(1-sigmoid(x))
# Ableitung von tanh

def NodesCalc(observation, weights, absnodesBool):
    nodes = nn.getZeroNodes()
    absnodes = nn.getZeroNodes()  
    
    # 1. Einzelne Nodes berechnen Nodematrix erstellen
    # Inputs aus Observation in den InputLayer übernehmen
    nodes[0][observation]=1
    
    # Berechne die Node Werte durch die Addition der Produkte aus weight und Node (beginne in Layer 1)
    layer = 1
    while(layer < len(nodes)):
        # für jeden Node im Layer layer
        node = 0
        while(node < len(nodes[layer])):
            # erhöhe den wert in nodes[layer][node] um jedes Produkt der Weights und deren Nodes von denen sie kommen
            weight = 0
            while(weight < len(weights[layer-1][node])):
                # aktueller node += weight des Pfades (layer-1) * node im vorherigen Layer von dem der Pfad kommt
                absnodes[layer][node]+=(weights[layer-1][node][weight]*nodes[layer-1][weight])
                nodes[layer][node] = sigmoid(absnodes[layer][node])
                weight+=1
            node+=1
        layer+=1
    if(absnodesBool == True):
        return absnodes 
    else:
        return nodes
    
# 2.1 Werte der einzelnen Nodes Berechnen  anhand  der ausgehenden weights + des nächsten Nodes  -> in neue Matrix speichern tmp3

# 2.2 weights berechnen und änderung in Matrix schreiben

# 3 Backprop anwenden!