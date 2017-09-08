'''
Created on 02.09.2017
@author: steffl&flemmig und nun?
''' 
debug = True

import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))

def Qvalue(weights, nodes, absnodes, observation):
    # calculate Qvalue return the proper action
    if(debug):
        print("WeightMatrix")
        print(weights)
        print
        print("NodeMatrix")
        print(nodes)
        print
    
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
    
    print(nodes)
    
    print()
    # 3. Outputs berechnen anhand von state und state + 1
    
    # 4. state + action in matrix speichern
    
    
