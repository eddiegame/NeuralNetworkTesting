'''
Created on 02.09.2017
@author: steffl&flemmig und nun?
Bitte nicht die Dateien kaputt machen, danke
bitte lass es jetzt funktionieren!!!

Ist doch nicht so schwer oder?
meine Fresse hat des lange gedauert bis ich des mal auf die Reihe bekommen hab
''' 
debug = False

import numpy as np
import random as rm

Inputlenght = 16                #Anzahl der Inputs (Observation)
Outputlength = 4                #Anzahl der Output (Actions)
amountNodesHL = Inputlenght + 1

GameHist = []
Zug = 0

# Variablen:
lear_rate   = 0.1
muta_rate   = 0.3
gamma       = 0.9
Zukunft     = 1
future_rate = 0.3

def getZeroNodes():
    return [np.zeros(Inputlenght),np.zeros(amountNodesHL),np.zeros(amountNodesHL),np.zeros(Outputlength)]
def getZeroWeights():
    return [np.zeros((amountNodesHL,Inputlenght)),np.zeros((amountNodesHL,amountNodesHL)),np.zeros((Outputlength,amountNodesHL ))]
   
def sigmoid(x): return 1 / (1 + np.exp(-x))

def train(environment, episodes, weights, history):
    env = environment
    #Durchlaufe alle Episoden die übergeben werden
    for e in range(episodes):
        observation = env.reset()   #Environemnt zurücksetzen
        done = False
        move = 0
        newZug= None
        print("## Episode: ",e,"##################################################################################")                
        while(done == False):       #Druchlaufe die Schleife, bis entweder das Ziel oder ein Hole erreicht wird
            # Das Environemnt wird dargestellt
            print("-- Move:",move,"----------------------------------------------------------------------------------")  
            env.render()
            # Die letzte observation wird gespeichert
            oldObs = observation
            rek = False
            
            # Das Neurale Netz ermittelt die Action mit dem höchsten Q-Wert für den aktuellen Stand
            print("\tCalculating Next Action by Qvalue...")
            if(rm.uniform(0,1)>muta_rate):
                newZug = Qvalue(weights,observation, history, rek, Zukunft, future_rate, newZug)
            else: 
                newZug = [observation, rm.randint(0,3), 0]
                
            GameHist.append(newZug)
            for idxx, zug in enumerate(GameHist):
                if (idxx == len(GameHist)-1):
                    NextAction=int(zug[1])
            print("\n\tChoosen Action: ", NextAction)
            
            # Die beste Action wird ausgeführt
            observation, reward, done, info = env.step(NextAction)
            move+=1
            print("\tAction Done - observation:",observation," reward:",reward," done:",done," info",info)
            
            # Die durchgeführte Action wird gespeichert
            saveAction(history,oldObs,NextAction, observation)
            
    # Die history wird ausgegeben
    printHistory(history)
    printHistory(GameHist)
    
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
 

def Qvalue(weights, observation, history, rek, future, future_rate, newZug):
    # reset Nodes and Absnodes
    nodes = getZeroNodes()
    absnodes = getZeroNodes()
    oldObs =observation
    Rektmp = rek    
    
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
        
    tmpNodes3Values = np.zeros(len(nodes[3]))
    
    if(Rektmp == False):
        print("\t\tFuture start (MAIN)")
    else:
        print("\t\t\tFuture start: ",future)
    
    for QValuesss in range(0, len(nodes[3])):
        if(Rektmp == False):
            print("\t\t\tIndex QValue: ",QValuesss," Value: ",nodes[3][QValuesss]," Recursion: ",Rektmp)
        else:
            print("\t\t\t\tIndex QValue: ",QValuesss," Value: ",nodes[3][QValuesss]," Recursion: ",Rektmp)
        if(Rektmp==False):
            tmpNodes3Values[QValuesss] =  nodes[3][QValuesss]
        
    # 4 Q Values zwischenspeichern und Q-Value fkt rekursiv rufen
    gefunden = False
    QValues = 0
    if future > 0:
        future=future-1
      
        for QValues in range(len(nodes[3])):
            # der wahrscheinlichste nächste Zustand 
            gefunden= False
            wslZustand = 0
            wslZustandCount = 0
            for entry in history:
                if (observation==entry[0] and QValues==entry[1]):
                    if entry[3] > wslZustandCount:
                        wslZustand = entry[2]
                        wslZustandCount = entry[3]
                        # Rekursiv aufrufen von QValue, Rückgabe von QValue nötig! keine Aktion
                        rek = True
                        gefunden = True
            if(gefunden==True):
                print("\t\tOutputnode at ",QValues," = ",nodes[3][QValues])
                newQ = Qvalue(weights, wslZustand, history, rek, future, future_rate, newZug)
                tmpNodes3Values[QValues] = (tmpNodes3Values[QValues] +  future_rate*newQ)/2 # Future rate wegen Bellmann -> Zukunft ist nicht immer gleich # 
                print("\t\Future: ",future," Index QValue: ",QValues," Value: ",nodes[3][QValues])
            else:
                tmpNodes3Values[QValues] = tmpNodes3Values[QValues]/2   
    if(Rektmp == False):
        print("\t\tFuture end (MAIN)")
    else:
        print("\t\t\tFuture end: ",future)
    
    for QValuess in range(0, len(nodes[3])):
        if(Rektmp == False): 
            print("\t\t\tIndex QValue: ",QValuess," Value: ",tmpNodes3Values[QValuess])
        else:
            print("\t\t\t\tIndex QValue: ",QValuess," Value: ",tmpNodes3Values[QValuess])
    # 3. Outputs berechnen anhand von state und state + 1
    if(Rektmp == False):
        newZug = [oldObs, np.argmax(tmpNodes3Values), tmpNodes3Values[np.argmax(tmpNodes3Values)]]
        return newZug # np.argmax(tmpNodes3Values)
    else:
        return nodes[3][np.argmax(nodes[3])]
    # 4. state + action in matrix speichern
    
    
 
    
    
    
    
