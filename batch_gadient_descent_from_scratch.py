
from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X,y = load_diabetes(return_X_y=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

X_train.shape

class GD:
  def __init__(self,learning_rate, epochs) :
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.intercept = None
    self.coef = None
    self.loss_history = []

  def fit(self,X_train,y_train):
    self.intercept= 0
    self.coef = np.ones(X_train.shape[1])

    for i in range(self.epochs):
      y_hat = self.intercept + np.dot(X_train, self.coef)
      self.loss_history.append(np.mean((y_train - y_hat) ** 2))
      intercept_der = -2*np.mean(y_train - y_hat)
      self.intercept = self.intercept - self.learning_rate*(intercept_der)
      coef_der= -2*np.dot((y_train-y_hat),X_train)/X_train.shape[0]
      self.coef = self.coef - self.learning_rate*(coef_der)
    print(self.intercept,self.coef)

  def predict(self,X_test):
    return self.intercept + np.dot(X_test, self.coef)



if __name__ == "__main__":
    obj1 = GD(0.1, 100)
    obj1.fit(X_train, y_train)
    # obj1.predict(X_test)

