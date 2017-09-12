'''
Created on 02.09.2017
@author: steffl&flemmig
''' 
debug = False

import numpy as np


Inputlenght = 16                #Anzahl der Inputs (Observation)
Outputlength = 4                #Anzahl der Output (Actions)
amountNodesHL = Inputlenght + 1 #Anzahl der Nodes im Hidden Layer

# Variablen des NN
lear_rate   = 0.1
muta_rate   = 0.3
gamma       = 0.9
Zukunft     = 1


#Gibt ein Array mit Nullen gefüllt zurück (Nodes)
def getZeroNodes():
    return [np.zeros(Inputlenght),np.zeros(amountNodesHL),np.zeros(amountNodesHL),np.zeros(Outputlength)]

#Gibt ein Array mit Nullen gefüllt zurück (Weights)
def getZeroWeights():
    return [np.zeros((amountNodesHL,Inputlenght)),np.zeros((amountNodesHL,amountNodesHL)),np.zeros((Outputlength,amountNodesHL ))]
    
#Gibt das Ergebnis der Sigmoid Funktion für X zurück
def sigmoid(x): return 1 / (1 + np.exp(-x))

#Funktion für das Training des NN
def train(environment, episodes, weights, history):
    env = environment
    #Durchlaufe alle Episoden die übergeben werden
    for e in range(episodes):
        Observation = env.reset()   #Environemnt zurücksetzen
        done = False
        move = 0
        print("## Episode: ",e,"##################################################################################")                
        while(done == False):       #Druchlaufe die Schleife, bis entweder das Ziel oder ein Hole erreicht wird
            # Das Environemnt wird dargestellt
            print("-- Move:",move,"----------------------------------------------------------------------------------")  
            env.render()
            # Die letzte observation wird gespeichert
            oldObservation = Observation
            
            # Das Neurale Netz ermittelt die Action mit dem höchsten Q-Wert für den aktuellen Stand
            print("\tCalculating Next Action by Qvalue...")
            NextAction = Qvalue(weights,Observation, history, False, Zukunft)
            print("\n\tChoosen Action: ", NextAction)
            
            # Die beste Action wird ausgeführt
            observation, reward, done, info = env.step(NextAction)
            move+=1
            print("\tAction Done - observation:",observation," reward:",reward," done:",done," info",info)
            
            # Die durchgeführte Action wird gespeichert
            saveAction(history,oldObservation,NextAction, observation)
            
    # Die history wird ausgegeben
    printHistory(history)
    
def saveAction(history, oldObs, NextAction, observation):
    print("History: ") 
    found = False
    for idx, entry in enumerate(history):
        if oldObs==entry[0] and NextAction==entry[1] and observation==entry[2]:
            entry[3] = int(entry[3])+1
            history[idx] = entry
            found = True
            print("\t Entry already exists: ",history[idx]," counter incremented")
    if found== False:
        newEntry = [oldObs,NextAction,observation,1]
        history.append(newEntry)
        sorted(history, key=lambda x: x[0])
        print("\t new Entry added: ",newEntry)
    print() 
    
def printHistory(history):
    print("Full History:")
    for entry in history:
        print("\t",entry) 
 
        
def Qvalue(weights, observation, history, recursion, future):
    # reset Nodes and Absnodes
    nodes = getZeroNodes()
    absnodes = getZeroNodes()
    tmp_recursion = recursion
    
    # 1. Einzelne Nodes berechnen Nodematrix erstellen
    # Inputs aus Observation in den InputLayer übernehmen
    nodes[0][observation]=1
    
    #print("\t\tCalculating Node values...")
    # Berechne die Nodewerte durch die Addition der Produkte aus weight und Node (beginne in Layer 1)
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
    #print("\t\t\t...done")
    
    #print("\t\tCalculating possible Futures ...")
    tmp_OutValues = np.zeros(len(nodes[3]))
    
    if(tmp_recursion == False):
        print("\t\tFuture start (MAIN)")
    else:
        print("\t\t\tFuture start: ",future)
        
    for QValueIndex in range(0, len(nodes[3])):
        if(tmp_recursion == False):
            print("\t\t\tIndex QValue: ",QValueIndex," Value: ",nodes[3][QValueIndex]," Recursion: ",tmp_recursion)
        else:
            print("\t\t\t\tIndex QValue: ",QValueIndex," Value: ",nodes[3][QValueIndex]," Recursion: ",tmp_recursion)
        if(tmp_recursion==False):
            tmp_OutValues[QValueIndex] =  nodes[3][QValueIndex]
        
    # 4 Q Values zwischenspeichern und Q-Value fkt rekursiv rufen
    gefunden = False
    if future > 0:
        future=future-1
      
        for QValueIndex in range(len(nodes[3])):
            # der wahrscheinlichste nächste Zustand 
            gefunden= False
            wslZustand = 0
            wslZustandCount = 0
            for entry in history:
                if (observation==entry[0] and QValueIndex==entry[1]):
                    if entry[3] > wslZustandCount:
                        wslZustand = entry[2]
                        wslZustandCount = entry[3]
                        # Rekursiv aufrufen von QValue, Rückgabe von QValue nötig! keine Aktion
                        rek = True
                        gefunden = True
                #print("---",wslZustand,"---",wslZustandCount,"-----")
            if(gefunden==True):
                print("\t\tOutputnode at ",QValueIndex," = ",nodes[3][QValueIndex])
                newQ = Qvalue(weights, wslZustand, history, rek, future)
                print("\t\tnew QValue for Action: ",QValueIndex," = ",tmp_OutValues[QValueIndex]," + ",newQ)
                tmp_OutValues[QValueIndex] = (tmp_OutValues[QValueIndex] +  newQ)/2
                print("\t\Future: ",future," Index QValue: ",QValueIndex," Value: ",nodes[3][QValueIndex])
    
    if(tmp_recursion == False):
        print("\t\tFuture end (MAIN)")
    else:
        print("\t\t\tFuture end: ",future)
    
    for QValueIndexEnd in range(0, len(nodes[3])):
        if(tmp_recursion == False):
            print("\t\t\tIndex QValue: ",QValueIndexEnd," Value: ",nodes[3][QValueIndexEnd])
        else:
            print("\t\t\t\tIndex QValue: ",QValueIndexEnd," Value: ",nodes[3][QValueIndexEnd])
        
        
    # 3. Outputs berechnen anhand von state und state + 1
    if(tmp_recursion == False):
        return np.argmax(tmp_OutValues)
    else:
        return nodes[3][np.argmax(nodes[3])]
    # 4. state + action in matrix speichern