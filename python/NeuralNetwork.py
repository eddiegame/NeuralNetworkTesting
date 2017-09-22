'''
Created on 02.09.2017
@author: steffl&flemmig
''' 
debug = False

import numpy as np
import random as rm
import math
import Training as tr

Inputlenght = 16                #Anzahl der Inputs (Observation) Hardgecoded
Outputlength = 4                #Anzahl der Output (Actions) Hardgecoded
AmountNodesHL = Inputlenght + 1
# GameHist speichert die einzelnen Züge eines Spiels bis Zustand=done
GameHist = []
Zug = 0

# Variablen: lear_rate gibt die Lerngeschwindigkeit an, muta_rate wie oft Zufallsaktionen genommen werden
# muta_rate_red um wie viel die muta_rate pro episode abnimmt, Zukunft wie viel Schritte in die Zukunft geschaut wird
# future_rate wie schwer die Werte aus der Zukunft mit ins Gewicht fallen
lear_rate       = 0.5
Zukunft         = 1
future_rate     = 0.3

def getZeroNodes():
    return [np.zeros(Inputlenght),np.zeros(AmountNodesHL),np.zeros(AmountNodesHL),np.zeros(Outputlength)]
def getZeroWeights():
    return [np.zeros((AmountNodesHL,Inputlenght)),np.zeros((AmountNodesHL,AmountNodesHL)),np.zeros((Outputlength,AmountNodesHL ))]
   
def sigmoid(x): return 1 / (1 + np.exp(-x))

def tanh(x): return math.tanh(x)

def learn(environment, episodes, weights, History):
    env = environment
    # saveChanges speichert alle veränderungen der Gewichte nach jedem Weightschange Vorgang
    saveChanges = getZeroWeights()
    muta_rate       = 0.7
    muta_rate_red   = 0.01
    #Durchlaufe alle Episoden die übergeben werden
    for e in range(episodes):
        observation = env.reset()   #Environemnt zurücksetzen
        done = False
        move = 0
        # newZug ist eine Liste, bei der von einem Zug obs, aktion, QValue, Reward und Loss für GameHist zwischengespeichert werden
        newZug= None
        if(debug==True):
            print("## Episode: ",e,"##################################################################################")    
        print("Episode:\t",e)
        GameHist.clear()            
        while(done == False):       #Druchlaufe die Schleife, bis entweder das Ziel oder ein Hole erreicht wird
            # Das Environemnt wird dargestellt
            if(debug==True):
                print("-- Move:",move,"----------------------------------------------------------------------------------")  
            # env.render()
            # Die letzte observation wird gespeichert
            LastObs = observation
            # rek wird benötigt da sonst die Durchläufe in der Zukunft die aktuellen überschreiben würden
            rek = False
            
            # Das Neurale Netz ermittelt die Action mit dem höchsten Q-Wert für den aktuellen Stand
            if(debug==True):
                print("\tCalculating Next Action by Qvalue...")
            if(rm.uniform(0,1)>muta_rate):
                newZug = getQvalue(weights,observation, History, rek, Zukunft, future_rate, newZug, False)
            else: 
                newZug = getQvalue(weights,observation, History, rek, 0, future_rate, newZug, True)
                if(debug==True):
                    print("\tRandom")
                
             
            GameHist.append(newZug)
            # nächste Aktion, die durch die QValue fkt bestimmt wurde, wird ausgegeben
            for Index, Zug in enumerate(GameHist):
                if (Index == len(GameHist)-1):
                    NextAction=int(Zug[1])
            if(debug==True):
                print("\n\tChoosen Action: ", NextAction)
            
            # Die beste Action wird ausgeführt
            observation, reward, done, info = env.step(NextAction)
            for Index, Zug in enumerate(GameHist):
                if (Index == len(GameHist)-1):
                    Zug[len(Zug)-2] = int(reward)
                    GameHist[Index] = Zug
            move+=1
            if(debug==True):
                print("\tAction Done - observation:",observation," reward:",reward," done:",done," info",info)
            
            # Die durchgeführte Action wird gespeichert
            saveAction(History,LastObs,NextAction, observation)
        # Arrays für Nodes und Weights für die Berechnungen im  Bprop werden init
        Bpropnodes = tr.createBpropNodes(len(GameHist))
        Bpropweights = tr.createBpropWeights(len(GameHist))
        print("\tReward: ",reward,"\tafter ",len(GameHist)," move(s)")
        if(debug==True):
            print("GameHist") 
            printHistory(GameHist)
            print("weights Anfang:")
            printHistory(weights)
        #für jeden Eintrag in GameHist(von hinten angefangen) wird das Netz trainiert.
        for Index, ZugReverse in enumerate(reversed(GameHist)):
            ZugReverse = tr.loss(ZugReverse, future_rate, weights, History)
            print("\t\tZug: ",Index,"\tError(Loss): ",ZugReverse[4])
            Bpropnodes = tr.ouputcalc(Bpropnodes, ZugReverse, Index, weights)
            Bpropnodes = tr.Bpropnodescalc(Bpropnodes, ZugReverse, Index, weights)
            Bpropweights = tr.weightscalc(Bpropnodes, ZugReverse, Index, Bpropweights, weights)
            weights = tr.weigthchange(Bpropweights, weights, lear_rate, muta_rate, saveChanges, Index)
        if(debug==True):
            print("weights Ende:")
            printHistory(weights)
            print("\tBpropNodes")
            printHistory(Bpropnodes)
            print("\tBpropWeights")
            printHistory(Bpropweights)

        # anpassen der muta_rate (Zufall wird weniger)
        muta_rate=muta_rate-muta_rate_red  
        
    # Die History wird ausgegeben
    if(debug==True):       
        printHistory(History)
    




# fügt die jeweiligen Aktionen mit den folgeaktionen in die History mitein
# History ist dafür da, das der Zustand als nächster angesehen wird der am wsl ist für jede Aktion
def saveAction(History, oldObs, NextAction, observation):
    if(debug==True):
        print("History: ") 
    found = False
    for Index, Entry in enumerate(History):
        if oldObs==Entry[0] and NextAction==Entry[1] and observation==Entry[2]:
            Entry[3] = int(Entry[3])+1
            History[Index] = Entry
            found = True
            if(debug==True):
                print("\t Entry already exists: ",History[Index]," counter incremented")
    if found==False:
        newEntry = [oldObs,NextAction,observation,1]
        History.append(newEntry)
        sorted(History, key=lambda x: x[0])
        if(debug==True):
            print("\t new Entry added: ",newEntry)
    if(debug==True):
        print() 
    
def printHistory(History):
    print("Full History:")
    for Index, Entry in enumerate(History):
        print("Index: ", Index)
        print("\t",Entry) 
        print("")
 

def getQvalue(weights, observation, History, rek, future, future_rate, newZug, rand):
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
        
    SavedNodes = getZeroNodes()[3]
    if(debug==True):
        if(Rektmp == False):
            print("\t\tFuture start (MAIN)")
        else:
            print("\t\t\tFuture start: ",future)
    
    for QValuesss in range(0, len(nodes[3])):
        if(Rektmp == False):
            if(debug==True):
                print("\t\t\tIndex QValue: ",QValuesss," Value: ",nodes[3][QValuesss]," Recursion: ",Rektmp)
            absnodescopy = np.full_like(absnodes, 0)
            np.copyto(absnodescopy,absnodes)
            SavedNodes[QValuesss] =  nodes[3][QValuesss]
        else:
            if(debug==True):
                print("\t\t\t\tIndex QValue: ",QValuesss," Value: ",nodes[3][QValuesss]," Recursion: ",Rektmp)

            
        
    # 4 Q Values zwischenspeichern und Q-Value fkt rekursiv rufen
    found = False
    Qvalue = 0
    if future > 0:
        future=future-1
      
        for Qvalue in range(len(nodes[3])):
            # der wahrscheinlichste nächste Zustand 
            found= False
            wslZustand = 0
            wslZustandCount = 0
            for Entry in History:
                if (observation==Entry[0] and Qvalue==Entry[1]):
                    if Entry[3] > wslZustandCount:
                        wslZustand = Entry[2]
                        wslZustandCount = Entry[3]
                        # Rekursiv aufrufen von QValue, Rückgabe von QValue nötig! keine Aktion
                        rek = True
                        found = True
            if(found==True):
                if(debug==True):
                    print("\t\tOutputnode at ",Qvalue," = ",nodes[3][Qvalue])
                newQvalue = getQvalue(weights, wslZustand, History, rek, future, future_rate, newZug, False)
                SavedNodes[Qvalue] = (SavedNodes[Qvalue] +  future_rate*newQvalue)/2 # Future rate wegen Bellmann -> Zukunft ist nicht immer gleich # 
                if(debug==True):
                    print("\t\Future: ",future," Index QValue: ",Qvalue," Value: ",nodes[3][Qvalue])
            else:
                SavedNodes[Qvalue] = SavedNodes[Qvalue]/2   
    if(debug==True):
        if(Rektmp == False):
            print("\t\tFuture end (MAIN)")
        else:
            print("\t\t\tFuture end: ",future)
    
    for Qvalue in range(len(nodes[3])):
        if(debug==True):
            if(Rektmp == False): 
                print("\t\t\tIndex QValue: ",Qvalue," Value: ",SavedNodes[Qvalue])
            else:
                print("\t\t\t\tIndex QValue: ",Qvalue," Value: ",SavedNodes[Qvalue])
    # 3. Outputs berechnen anhand von state und state + 1
    if(Rektmp == False):
        if(rand==False):
            newZug = [oldObs, np.argmax(SavedNodes), SavedNodes[np.argmax(SavedNodes)], 0, 0] # Obs, Aktion, QValue, Reward, Loss 
        else:
            randAction = rm.randint(0,Outputlength-1)
            newZug = [oldObs, randAction, SavedNodes[randAction], 0, 0]
        return newZug # np.argmax(SavedNodes)
    else:
        return nodes[3][np.argmax(nodes[3])]
    # 4. state + action in matrix speichern
    
    
 
    
    
    
    
