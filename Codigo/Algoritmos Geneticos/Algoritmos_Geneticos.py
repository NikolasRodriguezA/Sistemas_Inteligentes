"""
Created on Fri Apr 15 19:58:08 2022

@author: sheldor
"""

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from tensorflow.keras.layers import Dense, Input, Conv2D, Conv2DTranspose,UpSampling2D, Flatten, Reshape, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint 
from tensorflow.keras.utils import plot_model

from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import Recall

from sklearn.datasets import make_classification

# Definimos la cantidad de features a utilizar
n_features = 9

# Seteamos verbose en false
verbose = 0

# Generaramos el dataset
df = pd.read_csv('../Datos/Breast Cancer Prediction.csv')

X = pd.DataFrame()
X[['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 
   'Normal Nucleoli', 'Mitoses']] = df[['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 
                                        'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']]

X2, y2 = make_classification(n_samples=1000, n_features=n_features, n_classes=2, n_informative=4, 
                           n_redundant=1, n_repeated=2, random_state=1)
                                        
y = pd.DataFrame()
y[['Class']] = df[['Class']]

normalized_X=(X-X.mean())/X.std()

y = y['Class'].apply(lambda x: 1 if x==4 else 0 )

X_numpy = normalized_X.to_numpy()

Y_numpy = y.to_numpy()
 


X_train, X_rem, y_train, y_rem = train_test_split(X_numpy,Y_numpy, train_size=0.7)

X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)

print(f'Datos en Train: {X_train.shape[0]}')
print(f'Datos en Valid: {X_valid.shape[0]}')
print(f'Datos en Test: {X_test.shape[0]}')

# Instanciamos el modelo1 (reg Logistic)
model2 = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')

# Modelo 2 Red Neuronal

input_layer = Input(shape=(9,))
layer = Dense(16, activation='relu')(input_layer)
layer = Dropout(0.2)(layer)
out_layer=Dense(1,activation='sigmoid')(layer)
model = Model(inputs=input_layer,outputs=out_layer)

METRICS = [
    Recall(name='recall'),
    BinaryAccuracy(name='accuracy'),
    Precision(name='precision')
]

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=METRICS)


def calculate_fitness(model, x, y):
    cv_set = np.repeat(-1.0, x.shape[0])
    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if x_train.shape[0] != y_train.shape[0]:
            raise Exception()
        model.fit(x_train, y_train)
        predicted_y = model.predict(x_test)
        cv_set[test_index] = predicted_y
    return f1_score(y,cv_set)


def evaluate(individual):
    np_ind = np.asarray(individual)
    if np.sum(np_ind) == 0:
        fitness = 0.0
    else:
        feature_idx = np.where(np_ind == 1)[0]
        fitness = calculate_fitness(
            model2, X_numpy[:, feature_idx], Y_numpy
        )
        if verbose:
            print("Individuo: {}  Fitness Score: {} ".format(individual, fitness))

    return (fitness,)

creator.create("FeatureSelect", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FeatureSelect)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)

toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_features)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint) # Crossover
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1) # Mutacion
toolbox.register("select", tools.selTournament, tournsize=3) # Selecion
toolbox.register("evaluate", evaluate) # Evaluacion

N_POP = 100 # Tamaño de la población
CXPB = 0.5 # Probabilidad de crossover
MUTPB = 0.2 # Probabilidad de mutación
NGEN = 30 # Cantidad de generaciones

print(
    "Tamaño población: {}\nProbabilidad de crossover: {}\nProbabilida de mutación: {}\nGeneraciones totales: {}".format(
        N_POP, CXPB, MUTPB, NGEN
    )
)

# Función para generar salidas con estadisticas de cada generacion
def build_stats(gen, pop, fits):
    record = {}
    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5
    
    record['gen'] = gen + 1
    record['min'] = min(fits)
    record['max'] = max(fits)
    record['avg'] = mean
    record['std'] = std
    
    print("  Min {}  Max {}  Avg {}  Std {}".format(min(fits), max(fits), mean, std))
    
    return record

# Inicializamos a la poblacion
pop = toolbox.population(N_POP)

print("Evaluamos a los individuos inicializados.......")
fitnesses = list(map(toolbox.evaluate, pop))

# Asignamos a los inviduos el score del paso anterior
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

fitness_in_generation = {} # Variable auxiliar para generar el reporte
stats_records = [] # Variable auxiliar para generar el reporte

print("-- GENERACIÓN 0 --")
stats_records.append(build_stats(-1, pop, fitnesses[0]))

for g in range(NGEN):
    print("-- GENERACIÓN {} --".format(g + 1))
    # Seleccionamos a la siguiente generacion de individuos
    offspring = toolbox.select(pop, len(pop))
    
    # Clonamos a los invidiuos seleccionados
    offspring = list(map(toolbox.clone, offspring))

    # Aplicamos crossover y mutacion a los inviduos seleccionados
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values
            
    
    # Evaluamos a los individuos con una fitness invalida
    weak_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, weak_ind))
    for ind, fit in zip(weak_ind, fitnesses):
        ind.fitness.values = fit
    print("Individuos evaluados: {}".format(len(weak_ind)))

    # Reemplazamos a la poblacion completamente por los nuevos descendientes
    pop[:] = offspring
    
    # Mostramos las salidas de la estadisticas de la generacion actual
    fits = [ind.fitness.values[0] for ind in pop]
    
    stats_records.append(build_stats(g, pop, fits))
    

plt.figure(figsize=(10,8))
front = np.array([(c['gen'], c['avg']) for c in stats_records])
plt.plot(front[:,0][1:], front[:,1][1:], "-bo")
plt.title('Evolución F1')
plt.axis("tight")
plt.legend()
plt.xlabel("Generación")
plt.ylabel("F1")
plt.grid()
plt.show()

best_solution = tools.selBest(pop, 1)[0]
print(
    "El mejor individuo es: \n{}, con un F1 Score de {}".format(best_solution, best_solution.fitness.values)
)
