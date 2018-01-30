from svm import SVM
from sklearn.datasets import load_svmlight_file
from sklearn import metrics

learner = SVM(
    trade_off=1.0,
    gamma=0.1,
    batch_size=10,
    rf_dim=400,
    learning_rate=1e-3,
    num_epochs=20,
)

x_train, y_train = load_svmlight_file('svmguide1.txt')
x_test, y_test = load_svmlight_file('svmguide1_t.txt')
x_train = x_train.toarray()
x_test = x_test.toarray()

y_train[y_train == 0] = -1
y_test[y_test == 0] = -1

learner.fit(x_train, y_train)

y_test_predict = learner.predict(x_test)
test_acc = metrics.accuracy_score(y_test, y_test_predict)
print('Test accuracy:', test_acc)

