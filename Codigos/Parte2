import pandas as pd
import numpy as np
import cv2
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import os
from distutils.dir_util import copy_tree, remove_tree

from keras.utils.np_utils import to_categorical
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.utils import plot_model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV

from google.colab import drive
drive.mount('/content/drive')

train_dir = "/content/drive/MyDrive/Colab Notebooks/Data_trabalho_final/Alzheimer_Image/train/"
test_dir = "/content/drive/MyDrive/Colab Notebooks/Data_trabalho_final/Alzheimer_Image/test/"

def visualize(direction):
    list_dir=os.listdir(direction)
    plt.figure(figsize=(14,8))
    for i in range(1,7):
        plt.subplot(2,3,i)
        img= plt.imread(os.path.join(direction,list_dir[i]))
        plt.imshow(img,cmap='gray')
        plt.axis('off')
    plt.tight_layout()

ModerateDemented_dir= '/content/drive/MyDrive/Colab Notebooks/Data_trabalho_final/Alzheimer_Image/test/MildDemented'
visualize(ModerateDemented_dir)

data = []

for dirtrain in os.listdir(train_dir):
    print(dirtrain)
    for tr in os.listdir(train_dir + dirtrain):
        img = cv2.imread(train_dir + dirtrain + "/" + tr)
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(32, 32, 1)

        data.append([img, dirtrain])

for dirtest in  os.listdir(test_dir):
    print(dirtest)
    for ts in os.listdir(test_dir + dirtest):
        img = cv2.imread(test_dir + dirtest + "/" + ts)
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(32, 32, 1)

        data.append([img, dirtest])

import random
random.seed(20)
random.shuffle(data)

print(data[0][1])
print(data[3][1])

x, y = [], []
for e in data:
    x.append(e[0])
    y.append(e[1])

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(x[i])
  plt.title(y[i])
  plt.axis("off")

X = np.array(x)
Y = np.asarray(y)
hot = LabelEncoder()
y_encoded = hot.fit_transform(Y)
Y=to_categorical(y_encoded)
X=X/255
print(X.shape)
print(Y.shape)

random_seed = 2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=random_seed)

<hr></hr>

# **CNN**

<hr></hr>

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(32, 32, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

plot_model(model, to_file='model.png',show_shapes= True , show_layer_names=True)

model.compile(loss='mse', optimizer = "rmsprop", metrics = ['accuracy'])

history = model.fit(X_train, Y_train, validation_split=0.2, epochs = 10)

fig, ax = plt.subplots(1, 2, figsize=(12,6), facecolor="khaki")
ax[0].set_facecolor('palegoldenrod')  
ax[0].set_title('Loss', fontweight="bold")
ax[0].set_xlabel("Epoch", size=14)
ax[0].plot(history.epoch, history.history["loss"], label="Train Loss", color="navy")
ax[0].plot(history.epoch, history.history["val_loss"], label="Validation Loss", color="crimson", linestyle="dashed")
ax[0].legend()
ax[1].set_facecolor('palegoldenrod')
ax[1].set_title('Accuracy', fontweight="bold")
ax[1].set_xlabel("Epoch", size=14)
ax[1].plot(history.epoch, history.history["accuracy"], label="Train Acc.", color="navy")
ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation Acc.", color="crimson", linestyle="dashed")
ax[1].legend()

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1]*100)

<hr></hr>

# **Random Forest**

<hr></hr>

lenofimage = len(x)

X = np.array(x).reshape(lenofimage,-1)

X.shape

X = X/255.0
X[1]

y=np.array(y)
y.shape

from sklearn.model_selection import RandomizedSearchCV
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=77,stratify=y)

def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))

    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)

    pos_ind = (y_true!=0)|(y_pred!=0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]

    return 100 * np.mean(smap)

n_estimators = list(range(5,50))
max_features = ['sqrt']
max_depth = [int(x) for x in np.linspace(10, 60, num = 12)]
min_samples_split = list(range(2,11))
min_samples_leaf = list(range(1,10))
bootstrap = [True, False]

forest_params = [{'n_estimators': n_estimators,

    'max_features': max_features,

    'max_depth': max_depth,

    'min_samples_split': min_samples_split,

    'min_samples_leaf': min_samples_leaf,

    'bootstrap': bootstrap}]

cv = KFold(n_splits=5, shuffle=True, random_state=42)
rfc = RandomForestClassifier()

forest = RandomizedSearchCV(rfc, forest_params, cv = cv, verbose = -1)
forest = forest.fit(x_train, y_train)

test_predictions = forest.predict(x_test)

precision = accuracy_score(test_predictions, y_test) * 100
print("Accuracy with RandomForest: {0:.2f}".format(precision))

<hr></hr>

# **SVM**

<hr></hr>

# Importando a biblioteca do SVM
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svc = SVC(kernel='linear',gamma=1,C=1)
svc.fit(x_train, y_train)

pred_svm = svc.predict(x_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,pred_svm)*100
print("Accuracy on unknown data is",acc)

from sklearn.metrics import classification_report
print("Accuracy on unknown data is",classification_report(y_test,pred_svm))

<hr></hr>

# ***Comparação dos modelos***

<hr></hr>

# Dados fictícios de acurácia para diferentes modelos
acuracia = {'CNN': score[1]*100,
            'Random Forest': precision,
            'SVM': acc}

# Criar um DataFrame para armazenar os resultados
df = pd.DataFrame.from_dict(acuracia, orient='index', columns=['Acurácia'])

# Classificar os modelos em ordem decrescente de acurácia
df = df.sort_values(by='Acurácia', ascending=False)

# Imprimir a tabela de resultados
print(df)

# Dados fictícios de acurácia para diferentes modelos
acuracia = {'CNN': score[1]*100,
            'Random Forest': precision,
            'SVM': acc}

# Criar um DataFrame para armazenar os resultados
df = pd.DataFrame.from_dict(acuracia, orient='index', columns=['Acurácia'])

# Classificar os modelos em ordem decrescente de acurácia
df = df.sort_values(by='Acurácia', ascending=False)

# Plotar o gráfico de barras
df.plot(kind='bar', legend=False)
plt.xlabel('Modelos')
plt.ylabel('Acurácia')
plt.title('Resultados de Acurácia dos Modelos')
plt.show()
