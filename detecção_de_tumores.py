#--------------------------------- Importação das bibliotecas
# computação numerica, para manipulação de arrays e matrizes
import numpy as np 
# carregar nosso dataset BREAST CANCER WISCONSIN
from sklearn.datasets import load_breast_cancer
# Separar os dados do conjunto para treinamento e teste
from sklearn.model_selection import train_test_split
# Classificador da árvore de decisão
from sklearn.tree import DecisionTreeClassifier
# Métrica para avaliar a taxa de acerto do modelo
from sklearn.metrics import accuracy_score
# Construir a árvore
from sklearn.tree import plot_tree
# Criar a imagem da árvore
import matplotlib.pyplot as plt
# Construir e criar a matriz de confusão
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#--------------------------------- Carregar dados do dataset
data = load_breast_cancer() 
m_data = data.data
labels = data.target
#--------------------------------- Dados para treinamento e teste
m_data_train, m_data_test, labels_train, labels_test = train_test_split(m_data, labels, test_size=0.3, random_state=42) #controlar aleatoriedade e deixar o algoritmo consistente

#--------------------------------- Treinando o nosso classificador
my_classifier = DecisionTreeClassifier(random_state=42)
my_classifier.fit(m_data_train, labels_train)

#--------------------------------- Métrica para taxa de acerto
labels_predict = my_classifier.predict(m_data_test)
accuracy = accuracy_score(labels_test, labels_predict)
print("Acurácia do modelo:", accuracy)
#--------------------------------- Visualizar a árvore de decisão e matriz de confusão
plt.figure(figsize=(20,10))
plot_tree(my_classifier, feature_names=data.feature_names, class_names=data.target_names, filled=True, fontsize=5,proportion=True)
plt.show()
# Gerar a matriz de confusão
cm = confusion_matrix(labels_test, labels_predict)
# Plotar a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=my_classifier.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão')
plt.show()
