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


Inputlenght = 16                #Anzahl der Inputs (Observation)
Outputlength = 4                #Anzahl der Output (Actions)
amountNodesHL = Inputlenght + 1

def getZeroNodes():
    return [np.zeros(Inputlenght),np.zeros(amountNodesHL),np.zeros(amountNodesHL),np.zeros(Outputlength)]
def getZeroWeights():
    return [np.zeros((amountNodesHL,Inputlenght)),np.zeros((amountNodesHL,amountNodesHL)),np.zeros((Outputlength,amountNodesHL ))]
   

def sigmoid(x): return 1 / (1 + np.exp(-x))

def Qvalue(weights, nodes, absnodes, observation, history, rek, future):
    # reset Nodes and Absnodes
    nodes = getZeroNodes()
    absnodes = getZeroNodes()
    # calculate Qvalue return the proper action
    if(debug):
        print("WeightMatrix")
        print(weights)
        print
        print("NodeMatrix")
        print(nodes)
        print
    Rektmp = rek
    # 1. Einzelne Nodes berechnen Nodematrix erstellen
    # Inputs aus Observation in den InputLayer übernehmen
    nodes[0][observation]=1
    if(debug):
        print("Inputs in der NodeMatrix")
        print(nodes)
        print
    # Berechne die Node Werte durch die Addition der Produkte aus weight und Node (beginne in Layer 1)
    
    # für jeden Layer (beginne mit 1)
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
        
    if(debug):
        print("Node Werte berechnet:")
        print(nodes)
        print
    # 2. Aktivierungsfunktion auf hidden und output Layer anwenden
    
    # print(nodes)
    tmpNodes3Values = np.zeros(len(nodes[3]))
    print()
    for QValuesss in range(0, len(nodes[3])):
        print("ANFANG Zukunft: " + str(future) + " Index QValue: " + str(QValuesss) + " Wert: " + str(nodes[3][QValuesss]))
        print(Rektmp)
        if(Rektmp==False):
            tmpNodes3Values[QValuesss] =  nodes[3][QValuesss]
        
    # 4 Q Values zwischenspeichern und Q-Value fkt rekursiv rufen
    gefunden = False
    QValues = 0
    if future > 0:
        print("--------------------- Future --------------------------")
        future=future-1
      
        for QValues in range(len(nodes[3])):
            print("++++++++++++++++++++ ", QValues,"++++++++++++++++++++++++++++++++")
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
                print("---",wslZustand,"---",wslZustandCount,"-----")
            if(gefunden==True):
                print("nodes 3 bei " + str(QValues) + " = " + str(nodes[3][QValues]))
                newQ = Qvalue(weights, nodes, absnodes, wslZustand, history, rek, future)
                print("neuer QWert von Aktion: " + str(QValues) + ": " + str(tmpNodes3Values[QValues]) + "+" + str(newQ))
                tmpNodes3Values[QValues] = (tmpNodes3Values[QValues] +  newQ)/2
                print("Zukunft: " + str(future) + " Index QValue: " + str(QValues) + " Wert: " + str(nodes[3][QValues]))
            print("++++++++++++++++++++ ", QValues," End++++++++++++++++++++++++++++++++")
        print("--------------- Future End --------------------------")  
    # Wiedergabe der QValues am Ende 
    for QValuess in range(0, len(nodes[3])):
        print("ENDE: Zukunft: " + str(future) + " Index QValue: " + str(QValuess) + " Wert: " + str(nodes[3][QValuess]))
    if(Rektmp == False): 
        for QValuess in range(0, len(nodes[3])):
            print("ENDE-main: Zukunft: " + str(future) + " Index QValue: " + str(QValuess) + " Wert: " + str(tmpNodes3Values[QValuess])) 
    # 3. Outputs berechnen anhand von state und state + 1
    if(Rektmp == False):
        return np.argmax(tmpNodes3Values)
    else:
        return nodes[3][np.argmax(nodes[3])]
    # 4. state + action in matrix speichern
    
    
 
    
    
    
    
