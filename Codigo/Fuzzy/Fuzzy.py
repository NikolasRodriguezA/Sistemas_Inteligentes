#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 22:26:03 2022

@author: sheldor
"""

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


df = pd.read_csv('../Datos/Breast Cancer Prediction.csv')

def barplot_visualization(x, df):
    fig = plt.Figure(figsize = (12, 6))
    fig = px.bar(x = df[x].value_counts().index, y = df[x].value_counts(), color = df[x].value_counts().index, height = 600)
    fig.show()
    
df