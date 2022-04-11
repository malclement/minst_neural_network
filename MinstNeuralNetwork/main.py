from matplotlib import pyplot as plt

from MinstNeuralNetwork.DNN import DeepNeuralNetwork, x_train, x_val, y_val, y_train, epoch_list, accuracy_list
from MinstNeuralNetwork.tools import network_visualisation

# dnn = DeepNeuralNetwork(sizes=[784, 265, 140, 10])
# dnn.train(x_train, y_train, x_val, y_val)


dnn_1 = DeepNeuralNetwork(sizes=[784, 128, 64, 10])
dnn_1.train(x_train, y_train, x_val, y_val)
# network_visualisation(epoch_list, accuracy_list)

"""
dnn_1.complete_image_test("~/Desktop/mes_chiffres/1.jpg", 1)
dnn_1.complete_image_test("~/Desktop/mes_chiffres/2.jpg", 2)
dnn_1.complete_image_test("~/Desktop/mes_chiffres/3.jpg", 3)
dnn_1.complete_image_test("~/Desktop/mes_chiffres/4.jpg", 4)
dnn_1.complete_image_test("~/Desktop/mes_chiffres/5.jpg", 5)
dnn_1.complete_image_test("~/Desktop/mes_chiffres/6.jpg", 6)
dnn_1.complete_image_test("~/Desktop/mes_chiffres/7.jpg", 7)
dnn_1.complete_image_test("~/Desktop/mes_chiffres/8.jpg", 8)
dnn_1.complete_image_test("~/Desktop/mes_chiffres/9.jpg", 9)
"""
dnn_1.test_on_personal_image("~/Desktop/img_27.jpg", 9)

dnn_1.test_on_personal_image("~/Desktop/mes_chiffres/1.jpg", 1)
dnn_1.test_on_personal_image("~/Desktop/mes_chiffres/2.jpg", 2)
dnn_1.test_on_personal_image("~/Desktop/mes_chiffres/3.jpg", 3)
dnn_1.test_on_personal_image("~/Desktop/mes_chiffres/4.jpg", 4)
dnn_1.test_on_personal_image("~/Desktop/mes_chiffres/5.jpg", 5)
dnn_1.test_on_personal_image("~/Desktop/mes_chiffres/6.jpg", 6)
dnn_1.test_on_personal_image("~/Desktop/mes_chiffres/7.jpg", 7)
dnn_1.test_on_personal_image("~/Desktop/mes_chiffres/8.jpg", 8)
dnn_1.test_on_personal_image("~/Desktop/mes_chiffres/9.jpg", 9)
"""
dnn2.train(x_train, y_train, x_val, y_val)
plt.plot(epoch_list, accuracy_list)
dnn3.train(x_train, y_train, x_val, y_val)
plt.plot(epoch_list, accuracy_list)
dnn4.train(x_train, y_train, x_val, y_val)
plt.plot(epoch_list, accuracy_list)
"""

#network = DrawNN([10, 12, 12, 5])
#network.draw()

