import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV

from google.colab import drive
drive.mount('/content/drive')

dtypes = { 'Group': 'category', 'M/F': 'category'}

data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Data_trabalho_final/alzheimer.csv', dtype=dtypes)

Attributes:
It consists of 10 attributes which are describes as follows :

- Group  - It is a group of Converted (Previously Normal but developed dimentia later), Demented and Nondemented (Normal Pateints)

**Demographics Info**

- M.F - Gender
- Age - Age in years
- EDUC - Years of education
- SES - Socioeconomic status as assessed by the Hollingshead Index of Social Position and classified into categories from 1 (highest status) to 5 (lowest status)

**Clinical Info**

- MMSE - Mini-Mental State Examination score (range is from 0 = worst to 30 = best)
- CDR - Clinical Dementia Rating (0 = no dementia, 0.5 = very mild AD, 1 = mild AD, 2 = moderate AD)

**Derived anatomic volumes**

- eTIV - Estimated total intracranial volume, mm3
- nWBV - Normalized whole-brain volume
- ASF - Atlas scaling factor (unitless). Computed scaling factor that transforms native-space brain and skull to the atlas target

#Imprimindo as informações dos primeiros pacientes
data.head()

<hr></hr>

# **Análise dos dados**

<hr></hr>

data.info()

data.describe()

<hr></hr>

# ***Limpeza dos dados***

<hr></hr>

# Verificando valores faltantes
data.isna().sum()

# Verificando valores duplicados
data.duplicated().sum()

# Retirando atributos desnecessários
data = data.drop(['CDR'], axis=1) # remove a coluna de CDR pois mostra qual o resultado do exame
data.head()

# Podemos notar que há valores faltantes para SES e MMSE, e não há valores duplicados
# Com isso, devemos analisar as dados gerais desses dois atributos para poder preencher esses dados
print(data['SES'].describe())

def variacao(var):
    fig = plt.figure(figsize=(16,12))
    cmap=plt.cm.Blues
    cmap1=plt.cm.coolwarm_r
    ax1 = fig.add_subplot(212)

    ax1=sns.distplot(data[[var]],hist=False)
    ax1.set_title('Distribution of '+ var)
    plt.show()


variacao('SES')

Como podemos ver e sabemos pela definicação de cada atributo, o atributo SES recebe apenas números inteiros e está bem distribuido entre as classes. Logo, substituir pela média ou mediana seria uma boa escolha.

# Substituindo informação faltante pela mediana
data['SES'].fillna((data['SES'].median()), inplace=True)

data['MMSE'].describe()
variacao('MMSE')

Já o atributo MMSE, não está bem distribuido entre as classes, se concentrando nos valores entre 25 e 31, e só recebe números inteiros também. Então podemos substituir pela média ou mediana também.

# Substituindo informação faltante pela média
data['MMSE'].fillna((data['MMSE'].mean()), inplace=True)

data.isna().sum()

# Total de linhas e colunas
print("Total de linhas e colunas (Rows,Columns) : ",data.shape)

#Plota gráficos de relacionamento entre os dados, no caso, há uma comparação entre as espécies e seus tamanhos
sns.pairplot(data, hue="Group", corner=True, kind='reg')

# Histograma com as diferenças entre a quantidade de pacientes em cada grupo
sns.set_style("whitegrid")
palette=sns.color_palette("terrain")
sns.countplot(x='Group', data=data,palette=palette)
print(palette[2])

# Tirando os dados dos pacientes diagnosticados com Converted, pois queremos predizer apenas se a pessoa tem ou não Alzheimer
data = data[data["Group"] != "Converted"]

# Refazendo o index para não ficar "buracos" nos dados
data = data.reset_index(drop=True)

# Como o que queremos como resultado é o diagnóstico dos pacientes, devemos usar táticas de classificação para esse conjunto de dados, para isso é necessário tornar numérico os atributos que sejam categóricos
print(data.Group.value_counts())
print(data.Group.unique())


# Conversão (Group)
# 0 = Nondemented
# 1 = Demented
label = LabelEncoder()
label_group = label.fit_transform(data.Group)
print(label_group)
data['Group'] = label_group
data.head()

data.shape

# Conversão (M/F)
# 0 = F
# 1 = M
label_sex = label.fit_transform(data['M/F'])
print(label_sex)
data['M/F'] = label_sex
data.head()

correlation = data.corr()

# Plotando tabela de correlação
plot = sns.heatmap(correlation, annot = True, fmt=".1f", linewidths=.6)
plot

<hr></hr>

# ***Árvore de Decisão***

<hr></hr>

y = data['Group']
x = data.drop('Group',axis=1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/4, random_state=42)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")

    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")

params = {
    "criterion":("gini", "entropy"),
    "splitter":("best", "random"),
    "max_depth":(list(range(1, 10))),
    "min_samples_split":[2, 3, 4],
    "min_samples_leaf":list(range(1, 20)),
}


tree_clf = DecisionTreeClassifier(random_state=42)
tree_cv = GridSearchCV(
    tree_clf,
    params,
    scoring="f1",
    n_jobs=-1,
    verbose=1,
    cv=5
)

tree_cv.fit(x_train, y_train)
best_params = tree_cv.best_params_
print(f"Best paramters: {best_params})")

tree_cv.best_score_, tree_cv.best_params_

results = pd.DataFrame(tree_cv.cv_results_)
acc_tree = list(results['mean_test_score'])

tree_clf = DecisionTreeClassifier(**best_params)
tree_clf.fit(x_train, y_train)
print_score(tree_clf, x_train, y_train, x_test, y_test, train=False)

pred_tree = tree_clf.predict(x_test)

<hr></hr>

# ***KNN***

<hr></hr>

# importing sklearn StandardScaler class which is for Standardization
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(data.drop(['Group'],axis = 1),),
        columns=['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF'])

y = data['Group']

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=1/4,random_state=42, stratify=y)

X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

#Importando biblioteca de KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

knn_pipe = Pipeline(
    [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_jobs=-1))]
)

knn_params = {"knn__n_neighbors": range(1, 10)}

knn_grid = GridSearchCV(knn_pipe, knn_params, cv=5, n_jobs=-1, verbose=True)

knn_grid.fit(X_train, Y_train)

knn_grid.best_params_, knn_grid.best_score_

knn_grid.best_score_, knn_grid.best_params_

results = pd.DataFrame(knn_grid.cv_results_)
acc_knn = list(results['mean_test_score'])

knn_clf = KNeighborsClassifier(knn_grid.best_params_['knn__n_neighbors'])
knn_clf.fit(X_train, Y_train)
print_score(knn_clf, X_train, Y_train, X_test, Y_test, train=False)

pred_knn = knn_clf.predict(X_test)

<hr></hr>

# ***SVM***

<hr></hr>

# Melhorando os hiperparâmetros
params_grid = [{'kernel': ['rbf'], 'gamma': [
            1,
            0.1,
            0.01,
            0.001,
            0.0001,
        ], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()

# encoding train labels
encoder.fit(Y_train)
Y_train_svm = encoder.transform(Y_train)

# encoding test labels
encoder.fit(Y_test)
Y_test_svm = encoder.transform(Y_test)

#Total Number of Continous and Categorical features in the training set
num_cols = X_train._get_numeric_data().columns
print("Number of numeric features:",num_cols.size)
#list(set(X_train.columns) - set(num_cols))


names_of_predictors = list(X_train.columns.values)

Y_test_svm

X_test

# Importando a biblioteca do SVM
from sklearn.svm import SVC

svm_model = GridSearchCV(SVC(), params_grid, cv=5)
svm_model.fit(X_train, Y_train_svm)

svm_model.best_score_, svm_model.best_params_

results = pd.DataFrame(svm_model.cv_results_)
acc_svm = list(results['mean_test_score'])

svm_clf = SVC(C=svm_model.best_params_['C'], gamma=svm_model.best_params_['gamma'], kernel=svm_model.best_params_['kernel'])
svm_clf.fit(X_train, Y_train_svm)
print_score(svm_clf, X_train, Y_train_svm, X_test, Y_test_svm, train=False)

pred_svm = svm_clf.predict(X_test)

<hr></hr>

# ***Comparação dos modelos***

<hr></hr>

# Dados fictícios de acurácia para diferentes modelos
acuracia = {'Árvore de Decisão': accuracy_score(y_test, pred_tree) * 100,
            'KNN': accuracy_score(Y_test, pred_knn) * 100,
            'SVM': accuracy_score(Y_test_svm, pred_svm) * 100}

# Criar um DataFrame para armazenar os resultados
df = pd.DataFrame.from_dict(acuracia, orient='index', columns=['Acurácia'])

# Classificar os modelos em ordem decrescente de acurácia
df = df.sort_values(by='Acurácia', ascending=False)

# Imprimir a tabela de resultados
print(df)

# Dados de acurácia para diferentes modelos
acuracia = {'Árvore de Decisão': accuracy_score(y_test, pred_tree) * 100,
            'KNN': accuracy_score(Y_test, pred_knn) * 100,
            'SVM': accuracy_score(Y_test_svm, pred_svm) * 100}

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

# Teste de Diferença Estatística entre os modelos (ANOVA)
from scipy.stats import f_oneway

# Métricas de desempenho dos diferentes métodos
metodo1 = acc_tree
metodo2 = acc_knn
metodo3 = acc_svm

# Aplicando o teste ANOVA
stat, p_value = f_oneway(metodo1, metodo2, metodo3)

# Interpretação do resultado
alpha = 0.05  # Nível de significância
if p_value < alpha:
    print("Há diferença significativa entre os métodos.")
else:
    print("Não há diferença significativa entre os métodos.")

<hr></hr>

# ***Regras de associação***

<hr></hr>

dtypes = { 'Group': 'category', 'M/F': 'category'}

data_ra = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Data_trabalho_final/alzheimer.csv', dtype=dtypes)

data_ra.describe()

# Retirando atributos desnecessários
data_ra = data_ra.drop(['Group'], axis=1) # remove a coluna de Groups para fazer as regras com os grupos mais específicos de CDR
data_ra.head()

# Substituindo informação faltante pela média
data_ra['SES'].fillna((data_ra['SES'].median()), inplace=True)

# Substituindo informação faltante pela média
data_ra['MMSE'].fillna((data_ra['MMSE'].mean()), inplace=True)

data_ra.isna().sum()

data_ra.Age.value_counts()

bins = [59, 71, 77, 88, 99]
data_ra.Age = pd.cut(data_ra['Age'].to_numpy(), bins = bins, labels=['60-71 years', '71-78 years', '78-88 years', '88-98 years'])

data_ra.EDUC.value_counts()

bins = [5, 12, 13, 16, 25]
data_ra.EDUC = pd.cut(data_ra['EDUC'].to_numpy(), bins = bins, labels=['6-12 EDUC', '12-13 EDUC', '13-16 EDUC', '16-24 EDUC'])

bins = [0, 2, 3, 6]
data_ra.SES = pd.cut(data_ra['SES'].to_numpy(), bins = bins, labels=['0-2 SES', '2-3 SES', '3-5 SES'])

data_ra.MMSE.value_counts()

bins = [3, 27, 29, 31]
data_ra.MMSE = pd.cut(data_ra['MMSE'].to_numpy(), bins = bins, labels=['4-27 MMSE', '27-29 MMSE', '29-30 MMSE'])

data_ra.CDR.value_counts()

bins = [-0.1, 0.4, 0.9, 1.4, 2.1]
data_ra.CDR = pd.cut(data_ra['CDR'].to_numpy(), bins = bins, labels=['Nondemented', 'Very_mild_demented', 'Mild_demented', 'Moderate_demented'])

bins = [1104, 1357, 1470, 1597, 2005]
data_ra.eTIV = pd.cut(data_ra['eTIV'].to_numpy(), bins = bins, labels=['1105-1357 eTIV', '1357-1470 eTIV', '1470-1597 eTIV', '1597-2004 eTIV'])

bins = [0.642, 0.729, 0.756, 0.838]
data_ra.nWBV = pd.cut(data_ra['nWBV'].to_numpy(), bins = bins, labels=['0.644-0.729 nWBV', '0.729-0.756 nWBV', '0.756-0.837 nWBV'])

bins = [0.874, 1.194, 1.293, 1.588]
data_ra.ASF = pd.cut(data_ra['ASF'].to_numpy(), bins = bins, labels=['0.876-1.194 ASF', '1.194-1.293 ASF', '1.293-1.587 ASF'])

data_ra

CDR = data_ra['CDR']
nondemented = data_ra[CDR == 'Nondemented']
very_demented = data_ra[CDR == 'Very_mild_demented']
mild_demented = data_ra[CDR == 'Mild_demented']
moderate_demented = data_ra[CDR == 'Moderate_demented']

nondemented.shape, very_demented.shape, mild_demented.shape, moderate_demented.shape

!pip install apyori
from apyori import apriori

nondemented_trans = nondemented.values.tolist()

very_trans = very_demented.values.tolist()

mild_trans = mild_demented.values.tolist()

moderate_trans = moderate_demented.values.tolist()

regras_nondemented = list(apriori(nondemented_trans, min_support=0.1, min_confidence=0.8, min_lift=1.5))

regras_very = list(apriori(very_trans, min_support=0.1, min_confidence=0.8, min_lift=1.5))

regras_mild = list(apriori(mild_trans, min_support=0.1, min_confidence=0.8, min_lift=1.5))

regras_moderate = list(apriori(moderate_trans, min_support=0.1, min_confidence=0.8, min_lift=1.5))

def regras_itemset(itemset, nivel):
  for i, rule in enumerate(itemset.ordered_statistics):
    nome = ', '.join(rule.items_add)
    if(len(rule.items_add) == 1 and nome == nivel):
        item = ', '.join(itemset.items)
        print(f'itemset: {item}, Support: {str(itemset.support)}')
        antecedent = ', '.join(rule.items_base)
        consequent = ', '.join(rule.items_add)
        print(f'\tRegra{str(i+1)}: {antecedent} => {consequent}')
        print(f'\tConfidence: {str(rule.confidence)}, Lift: {str(rule.lift)}')
        print()

for itemset in regras_nondemented:
  regras_itemset(itemset, "Nondemented")

for itemset in regras_very:
  regras_itemset(itemset, "Very_mild_demented")

for itemset in regras_mild:
  regras_itemset(itemset, "Mild_demented")

for itemset in regras_moderate:
  regras_itemset(itemset, "Moderate_demented")
