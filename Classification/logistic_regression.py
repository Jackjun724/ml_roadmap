import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_data():
    iris = load_iris()
    X = iris.data
    y = np.array([(0 if i > 1 else 1) for i in iris.target])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    y_train = np.reshape(y_train, (-1, 1))

    return X_train, X_test, y_train, y_test


def neural_network():
    """
    具备神经网络的模型
    参数的定义参看 Andrew Ng 的深度学习课程 ：
        A: Cost Function
        C: 未经过Sigmoid计算的Cost Function
        W、B: weight、bias
    """

    X_train, X_test, y_train, y_test = load_data()

    speed_rate = 0.01
    # 数据集数量, 数据集宽度
    m = X_train.shape[0]

    # 参数初始化
    W_1 = np.random.randn(2, 4)
    W_2 = np.random.randn(1, 2)
    B_1 = np.zeros((1, 2))
    B_2 = np.zeros((1, 1))

    epochs = 200000
    acc_print = epochs / 20
    for i in range(epochs):
        # 正向传递 双层神经网络
        # 第一隐层
        Z_1 = np.dot(X_train, W_1.T) + B_1
        A_1 = sigmoid(Z_1)

        # 输出层
        Z_2 = np.dot(A_1, W_2.T) + B_2
        A_2 = sigmoid(Z_2)

        # Back prop , gradient descent
        dZ_2 = A_2 - y_train
        dW_2 = np.sum(dZ_2 * A_1, axis=0, keepdims=True) / m
        dB_2 = np.sum(dZ_2, keepdims=True) / m

        dA_1 = dZ_2 * W_2
        dG = A_1 * (1 - A_1)
        dZ_1 = dA_1 * dG
        dW_1 = np.dot(dZ_1.T, X_train) / m
        dB_1 = np.sum(dZ_1, axis=0, keepdims=True) / m

        # 调整
        W_2 = W_2 - dW_2 * speed_rate
        B_2 = B_2 - dB_2 * speed_rate
        W_1 = W_1 - dW_1 * speed_rate
        B_1 = B_1 - dB_1 * speed_rate
        if i % acc_print == 0:
            test_result = sigmoid(np.dot(sigmoid(np.dot(X_test, W_1.T) + B_1), W_2.T) + B_2).T[0]
            test_result = np.array([1 if i > 0.5 else 0 for i in test_result])
            train_result = sigmoid(np.dot(sigmoid(np.dot(X_train, W_1.T) + B_1), W_2.T) + B_2).T[0]
            train_result = np.array([1 if i > 0.5 else 0 for i in train_result])
            print('------------------------------------------')
            print(f'{i} : Test Accuracy  :' + str(accuracy_score(y_test, test_result)))
            print(f'{i} : Train Accuracy :' + str(accuracy_score(y_train, train_result)))
            print('------------------------------------------')

    # np.set_printoptions(precision=3, suppress=True)
    test_result = sigmoid(np.dot(sigmoid(np.dot(X_test, W_1.T) + B_1), W_2.T) + B_2).T[0]
    test_result = np.array([1 if i > 0.5 else 0 for i in test_result])
    print('Accuracy：' + str(accuracy_score(y_test, test_result)))


def normal():
    """
    普通单层机器学习的Logistic Regression，不包含神经网络
    """
    # load iris dataset
    X_train, X_test, y_train, y_test = load_data()
    # define learning rate (alpha)
    speed_rate = 0.1

    # 数据集数量, 数据集宽度
    m = X_train.shape[0]
    d = X_train.shape[1]

    W = np.zeros((d, 1))
    B = 0

    # train
    epochs = 200000
    acc_print = epochs / 20
    for i in range(epochs):
        cost_func = np.dot(X_train, W) + B
        Y_hat = sigmoid(cost_func)
        dZ = Y_hat - y_train
        dw = np.reshape(np.sum(dZ * X_train, axis=0) / np.array([m]), (-1, 1))
        db = np.sum(dZ) / m
        W = W - speed_rate * dw
        B = B - speed_rate * db
        if i % acc_print == 0:
            test_result = sigmoid(np.dot(X_test, W) + B).T[0]
            test_result = np.array([1 if i > 0.5 else 0 for i in test_result])
            train_result = sigmoid(np.dot(X_train, W) + B).T[0]
            train_result = np.array([1 if i > 0.5 else 0 for i in train_result])
            print('------------------------------------------')
            print(f'{i} : Test Accuracy  :' + str(accuracy_score(y_test, test_result)))
            print(f'{i} : Train Accuracy :' + str(accuracy_score(y_train, train_result)))
            print('------------------------------------------')

    np.set_printoptions(precision=3, suppress=True)
    test_result = sigmoid(np.dot(X_test, W) + B).T[0]
    test_result = np.array([1 if i > 0.5 else 0 for i in test_result])
    print('Accuracy：' + str(accuracy_score(y_test, test_result)))


if __name__ == '__main__':
    # 普通机器学习模型
    normal()
    # 神经网络模型
    # neural_network()

