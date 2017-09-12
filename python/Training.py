#-*- coding: utf-8 -*-
'''
Created on 01.09.2017
@author: steffl&flemmig 
'''

import gym
import numpy as np                 #Notwendig für die Matritzenrechnung
import random                     #Notwendig fpr die gernerierung von Zufallszahlen
import NeuralNetwork as nn        #Beinhaltet das Neurale Netz


# Wenn Done und Reward=0 -> Loch dann ist Reward -1
# Wenn Done und Reward=1 -> Loch dann ist Reward 1
# sonst 0
# Da GameHistorie übergeben wird wenn am Ende Loch wird alles mit reward -0,1 berechnet... -0,1 wird eine Variable sein
# Da GameHistorie übergeben wird wenn am Ende Win wird alles mit reward 0,1 berechnet... 0,1 wird eine Variable sein 

# 1.1 Loss für einzelne Aktionen in History berechnen und anpassen

# 1.2 Gamma von i Berechnen = -L* Abl. von Aktivierungsfk(von Array tmp1) von Hidden Layers und Output

# 2.1 Werte der einzelnen Nodes Berechnen  anhand  der ausgehenden weights + des nächsten Nodes  -> in neue Matrix speichern tmp3

# 2.2 weights berechnen und änderung in Matrix schreiben

# 3 Backprop anwenden!