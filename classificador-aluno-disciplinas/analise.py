import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras 


seed = 123
np.random.seed(seed)  # for reproducibility


xl = pd.ExcelFile("data.xlsx")
df = xl.parse(xl.sheet_names[0])
df.head()

#%%
disciplinas = np.unique(df.disciplina)

disciplinas_dict = {}
for i,it in enumerate(disciplinas):
    disciplinas_dict[i]=it
    disciplinas_dict[it]=i

#dict = { 0:1, 1:2, 10:3, 11:4, 20:5, 21:6, 30:7, 31:8, 40:9, 41:10, }
aluno_map_dict = {  }
c = 0
for p in range(0,300,10):
    aluno_map_dict[p] = c
    aluno_map_dict[p+1] = c+1
    c += 2


#%%




def converte_periodo_cod(cod_inicio,cod_fim):
    def converte_periodo(ano_inicial,semestre_inicial,ano_final,semestre_final):
        anos = ano_final-ano_inicial    
        return 2*anos + semestre_final-semestre_inicial
        
    ano_inicial      = int(str(cod_inicio)[0:4])
    ano_final        = int(str(cod_fim)[0:4])
    semestre_inicial = int(str(cod_inicio)[4:])
    semestre_final   = int(str(cod_fim)[4:])
    
    periodos_totais = converte_periodo(ano_inicial,semestre_inicial,ano_final,semestre_final)
    
    print(ano_inicial,ano_final,semestre_inicial,semestre_final,periodos_totais)
    return periodos_totais

converte_periodo_cod('20061','20061')
converte_periodo_cod('20061','20062')
converte_periodo_cod('20061','20071')
converte_periodo_cod('20061','20072')
converte_periodo_cod('20061','20081')
converte_periodo_cod('20061','20082')

#%%

def create_aluno_array(id_aluno,max_periodos = 8):

    aluno = {}

    aluno_df = df[df.aluno == id_aluno]

    aluno_concluiu = aluno_df.concluiu.values[0]
    aluno_ano_inicio = np.min(aluno_df.periodo.values)
    aluno_periodo = [converte_periodo_cod(aluno_ano_inicio,p)   for p in aluno_df.periodo.values]

    aluno_periodo = np.array(aluno_periodo)
    aluno_disciplinas = aluno_df.disciplina.values
    aluno_notas = aluno_df.nota.values

    aluno['aluno_concluiu'] = aluno_concluiu

    print(aluno_concluiu)
    print(aluno_disciplinas)
    print(aluno_notas)
    print(aluno_periodo)

    aluno_dict = {}
    for i in range(max_periodos):
        ids = np.where(aluno_periodo == i)[0]
        print(ids)
        aluno_dict[i] = [[aluno_disciplinas[ii], aluno_notas[ii]] for ii in ids]

    print(aluno_dict)
    aluno['periodos'] = aluno_dict
    
    
    def create_aluno_matrix(aluno_dict, disciplinas, disciplinas_dict, max_periodos=8):
        aluno_matrix = np.zeros((max_periodos, len(disciplinas))) - 1
        for per in range(max_periodos):
            for it in aluno_dict['periodos'][per]:
                aluno_matrix[per, disciplinas_dict[it[0]]] = it[1]  # matrix recebe nota na posicao correta
    
        return aluno_matrix

    aluno_matrix = create_aluno_matrix(aluno, disciplinas, disciplinas_dict, max_periodos=8)
    aluno['matrix'] = aluno_matrix
    
    return aluno

#%%
alunos = np.unique(df.aluno)
len(alunos)

X = []
Y = []

for aluno in alunos:
    aluno_dict = create_aluno_array(aluno)
    X.append(aluno_dict['matrix'].ravel())    
    Y.append(aluno_dict['aluno_concluiu'])

X = np.array(X)
Y = np.array(Y)


plt.matshow(X)


print('numero alunos', Y.shape[0])

print('numero alunos aprovados', np.sum(Y))


from sklearn.decomposition import pca
pca = PCA(n_components=200)
pca.fit(X)

print(pca.explained_variance_ratio_)  
X = pca.transform(X)

plt.matshow(X)


from matplotlib.colors import ListedColormap
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

plt.scatter(X[:,0],X[:,1], c=Y,cmap=cm_bright)


#%%

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
#%%


encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)



# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(50, input_dim=X.shape[1], kernel_initializer='normal', activation='sigmoid'))
    model.add(Dropout(0.3))
    model.add(Dense(50, input_dim=X.shape[1], kernel_initializer='normal', activation='sigmoid'))
    model.add(Dropout(0.3))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=10, verbose=1)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


model = create_baseline()
model.summary()

class_weight = {0 : 1., 1: 10.}

model.fit(X_train,Y_train,validation_split=0.1,epochs=406,class_weight=class_weight)

yhat = model.predict(X_test)

plt.scatter(Y_test,yhat)
plt.plot(Y_test)
plt.plot(yhat)



from matplotlib.colors import ListedColormap
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

plt.scatter(X_test[:,0],X_test[:,1], c=Y_test,cmap=cm_bright)
plt.scatter(X_test[:,0],X_test[:,1], s=5,c=yhat.ravel(),cmap=cm_bright)









#%% http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

# Build a classification task using 3 informative features
#X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
#                           n_redundant=2, n_repeated=0, n_classes=8,
#                           n_clusters_per_class=1, random_state=0)


X = X
y = Y

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy',verbose=2)
rfecv.fit(X[:,0:25], y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()












