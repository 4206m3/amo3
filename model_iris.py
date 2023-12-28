import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print('Загружаем набор данных Ирисы...')
iris = datasets.load_iris()


print('Обучаем модель логистической регрессии...')
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.3, random_state = 0)
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)
print('accuracy of model: ',model.score(X_test, y_test))

print('Получим предсказание модели на тестовом примере [1,1,1,1]:')
res = model.predict([[1,1,1,1]])
print(iris.target_names[res][0])