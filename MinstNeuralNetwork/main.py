from MinstNeuralNetwork.DNN import DeepNeuralNetwork, x_train, x_val, y_val, y_train

# dnn = DeepNeuralNetwork(sizes=[784, 265, 140, 10])
# dnn.train(x_train, y_train, x_val, y_val)


dnn_1 = DeepNeuralNetwork(sizes=[784, 397, 10, 10])
dnn_1.train(x_train, y_train, x_val, y_val)
"""
dnn_1.test_on_personal_image("~/Desktop/mes_chiffres/1.jpg", 1)
dnn_1.test_on_personal_image("~/Desktop/mes_chiffres/2.jpg", 2)
dnn_1.test_on_personal_image("~/Desktop/mes_chiffres/3.jpg", 3)
dnn_1.test_on_personal_image("~/Desktop/mes_chiffres/4.jpg", 4)
dnn_1.test_on_personal_image("~/Desktop/mes_chiffres/5.jpg", 5)
dnn_1.test_on_personal_image("~/Desktop/mes_chiffres/6.jpg", 6)
dnn_1.test_on_personal_image("~/Desktop/mes_chiffres/7.jpg", 7)
dnn_1.test_on_personal_image("~/Desktop/mes_chiffres/8.jpg", 8)
dnn_1.test_on_personal_image("~/Desktop/mes_chiffres/9.jpg", 9)
dnn_1.test_on_personal_image("~/Desktop/img_27.jpg", 9)"""
"""
plt.plot(epoch_list, accuracy_list)
"""
"""
dnn2.train(x_train, y_train, x_val, y_val)
plt.plot(epoch_list, accuracy_list)
dnn3.train(x_train, y_train, x_val, y_val)
plt.plot(epoch_list, accuracy_list)
dnn4.train(x_train, y_train, x_val, y_val)
plt.plot(epoch_list, accuracy_list)
"""
"""
plt.xlabel('epoch')  # Add an x-label to the axes.
plt.ylabel('accuracy in %')  # Add a y-label to the axes.
plt.grid()
plt.show()
"""
"""
network = DrawNN([784, 397, 10, 10])
network.draw()
"""
