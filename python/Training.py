#-*- coding: utf-8 -*-
'''
Created on 01.09.2017
@author: steffl&flemmig 
'''

import numpy as np                 #Notwendig für die Matritzenrechnung
import NeuralNetwork as nn        #Beinhaltet das Neurale Netz
import math
import random as rm

# Wenn Done und Reward=0 -> Loch dann ist Reward -1
# Wenn Done und Reward=1 -> Loch dann ist Reward 1
# sonst 0
# Da GameHistorie übergeben wird wenn am Ende Loch wird alles mit reward -0,1 berechnet... -0,1 wird eine Variable sein
# Da GameHistorie übergeben wird wenn am Ende Win wird alles mit reward 0,1 berechnet... 0,1 wird eine Variable sein 

# 1.1 Loss für einzelne Aktionen in History berechnen und anpassen
# Idee von Andi
def loss(ZugReverse, futu_rate, weights, History):
        if(nn.debug==True):
            print("-------------------------------- loss ----------------------------")
        wslZustandCount=0
        wslZustand=ZugReverse[0]
        if(nn.debug==True):
            print("obs: ", ZugReverse[0], " genommene Aktion: ", ZugReverse[1])
        for Entry in History:
                if (ZugReverse[0]==Entry[0] and ZugReverse[1]==Entry[1]):
                    if Entry[3] > wslZustandCount:
                        wslZustand = Entry[2]
                        wslZustandCount = Entry[3]
        
        if(nn.debug==True):
            print("wslZustand: ", wslZustand)
        futuQvalue = NodesCalc(wslZustand, weights, False, True)
        if(nn.debug==True):
            print("futuQValue : ", futuQvalue)
        ZugReverse[4] = 0.5*(ZugReverse[3]+futu_rate*futuQvalue-ZugReverse[2])
        if(nn.debug==True):
            print("loss: ", ZugReverse[4])
        if(nn.debug==True):
            print("-------------------------------- Ende - loss ----------------------------")
        return ZugReverse


def createBpropNodes(GameHistlength): 
    BpropNodes = [[np.zeros(nn.Inputlenght),np.zeros(nn.AmountNodesHL),np.zeros(nn.AmountNodesHL),np.zeros(nn.Outputlength)] for g in range(GameHistlength)]
    return BpropNodes
	
def createBpropWeights(GameHistlength): 
    BpropNodes = [[np.zeros((nn.AmountNodesHL,nn.Inputlenght)),np.zeros((nn.AmountNodesHL,nn.AmountNodesHL)),np.zeros((nn.Outputlength,nn.AmountNodesHL ))] for g in range(GameHistlength)]
    return BpropNodes
# 1.2 Gamma von i Berechnen = -L* Abl. von Aktivierungsfk(von Array tmp1) von Hidden Layers und Output
def ouputcalc(BpropNodes ,ZugReverse, Index, weights):  
    for i in range(nn.Outputlength):
        if(nn.debug==True):
            print("Entry : ",ZugReverse[1],"  i : ",i)          
        if(ZugReverse[1] == i):    
            if(nn.debug==True):
                print("Index : ", Index)
            absnodes = NodesCalc(ZugReverse[0], weights, True, False) 

            BpropNodes[Index][3][i] = -1*ZugReverse[4] * arcsigmoid(absnodes[3][i])
            break
    return BpropNodes
# 2.1 Werte der einzelnen Nodes Berechnen  anhand  der ausgehenden weights + des nächsten Nodes 
def Bpropnodescalc(BpropNodes, ZugReverse,Index, weights):
    AnzahlHiddenLa = 2
    absnodes = NodesCalc(ZugReverse[0], weights, True, False) 
    SumWeights = 0
    if(nn.debug==True):
        print(" Nodes Calc --------------------------------------")
    for i in range(AnzahlHiddenLa,0,-1):
        if(nn.debug==True):
            print("\t außen i: ",i," --------------------------------------")
        for j in range(0, len(BpropNodes[Index][i])):
            if(nn.debug==True):
                print("\t\t innen j: ",j," --------------------------------------")  # immer 17 durchläufe
            for k in range(0, len(weights[i])): # erst 4 dann 17 durchläufe
                if(nn.debug==True):
                    print("\t\t\t innen k: ",k," --------------------------------------")
                SumWeights= SumWeights + weights[i][k][j-1]*BpropNodes[Index][i+1][k]
            BpropNodes[Index][i][j] = arcsigmoid(absnodes[i][j]) * SumWeights
            SumWeights = 0
    if(nn.debug==True):
        print("Nodes Calc- ENDE --------------------------------------")
    BpropNodes[Index][0][ZugReverse[0]]=1
    return BpropNodes
# 2.2 weights berechnen 
def weightscalc(BpropNodes, ZugReverse, Index, Bpropweights, weights):
    nodes = NodesCalc(ZugReverse[0], weights, False, False) 
    for i in range(0, len(Bpropweights[Index])):  # durchläuft die einzelnen 2D Weight Arrays
        if(nn.debug==True):
            print("i :",i)
        if(nn.debug==True):
            print("länge Array j ", len(Bpropweights[Index][i]))
        for j in range(0, len(Bpropweights[Index][i])): #durchläuft die einzelnen Rows (untereinander)
            if(nn.debug==True):
                print("\tj :",j)
            if(nn.debug==True):
                print("\tlänge Array k ", len(Bpropweights[Index][i][j]))
            for k in range(0, len(Bpropweights[Index][i][j])): #durchläuft Inhalte in den Rows
                if(nn.debug==True):
                    print("\t\tk :",k)
                tmp1 = nodes[i][k]
                tmp2 = BpropNodes[Index][i+1][j] 
                Bpropweights[Index][i][j][k] = tmp1 * tmp2 
                if(nn.debug==True):
                    print("\t\t\t Berechnung (",i,",",j,",",k,") = ",tmp1,"*",tmp2)
    return Bpropweights

# 3 Backprop anwenden!
def weigthchange(Bpropweights, weights, learn_rate, muta_rate, saveChangInweights, Index):
    for i in range(0, len(weights)): #geht weight arrays ab
        for j in range(0, len(weights[i])): #geht Rows ab
            for k in range(0,len(weights[i][j])): # einzelne Inhalte einer Row
                deltatmp = 0
                tmp1 = learn_rate*Bpropweights[Index][i][j][k]
                tmp2 = saveChangInweights[i][j][k]
                deltatmp = tmp1 + muta_rate*tmp2
                weights[i][j][k]=weights[i][j][k]+deltatmp
                saveChangInweights[i][j][k] = deltatmp
    return weights
def sigmoid(x): return 1 / (1 + np.exp(-x))
def arcsigmoid(x): return sigmoid(1-sigmoid(x))
# Ableitung von tanh

def NodesCalc(observation, weights, absnodesBool, QValueBool):
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
        if(QValueBool==True):
            return nodes[len(nodes)-1][np.argmax(nodes[len(nodes)-1])]
        else:
            return nodes
    




