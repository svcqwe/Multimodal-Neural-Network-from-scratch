import numpy as np
from layers import DenseLayer, ReLU, Sigmoid, Dropout, BatchNormalization, Conv_2D, MinPooling2D, Flatten, Concatenate, GRU
import datetime
import json
from psycopg2 import connect
import matplotlib.pyplot as plt
from PIL import Image 

class NeuralNetwork:

    def __init__(self, num_features:int):

        self.conv_1 = Conv_2D(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.min_pooling_1 = MinPooling2D()
        self.conv_2 = Conv_2D(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.min_pooling_2 = MinPooling2D()
        self.flatten_conv = Flatten(mode="cnn")

        self.gru_1 = GRU(input_size=num_features, hidden_size=num_features+8)
        self.gru_2 = GRU(input_size=num_features+8, hidden_size=num_features+4)
        self.flatten_rnn = Flatten(mode='last')

        self.concat = Concatenate()

        self.dense_1 = DenseLayer(input_units=36895, output_units=512, init_weigts='he', l1_reg=0.0001, l2_reg=0.001)
        self.batchnorm_1 = BatchNormalization(num_features=512)
        self.activation_1 = ReLU()
        self.dropout_1 = Dropout(p=0.2)

        self.dense_2 = DenseLayer(input_units=512, output_units=256, init_weigts='he', l1_reg=0.00001, l2_reg=0.0001)
        self.batchnorm_2 = BatchNormalization(num_features=256)
        self.activation_2 = ReLU()
        self.dropout_2 = Dropout(p=0.4)

        self.dense_3 = DenseLayer(input_units=256, output_units=128, init_weigts='he', l1_reg=0.000001, l2_reg=0.00001)
        self.batchnorm_3 = BatchNormalization(num_features=128)
        self.activation_3 = ReLU()
        self.dropout_3 = Dropout(p=0.3)

        self.dense_4 = DenseLayer(input_units=128, output_units=32, init_weigts='he', l1_reg=0, l2_reg=0)
        self.batchnorm_4 = BatchNormalization(num_features=32)
        self.activation_4 = ReLU()
        self.dropout_4 = Dropout(p=0.15)

        self.dense_5 = DenseLayer(input_units=32, output_units=1, init_weigts='xavier', l1_reg=0, l2_reg=0)
        self.activation_5 = Sigmoid()

    def forward(self, X_img, X_vectors, train:bool=True):

        conv_output_1 = self.conv_1.forward(X_img)
        min_pooling_output_1 = self.min_pooling_1.forward(conv_output_1)
        conv_output_2 = self.conv_2.forward(min_pooling_output_1)
        min_pooling_output_2 = self.min_pooling_2.forward(conv_output_2)
        flattened_conv = self.flatten_conv.forward(min_pooling_output_2)

        hidden_output_rnn_1, h_current = self.gru_1.forward(X_vectors)
        hidden_output_rnn_2, _ = self.gru_2.forward(hidden_output_rnn_1, h_current)
        flattened_rnn = self.flatten_rnn.forward(hidden_output_rnn_2)

        concat_output = self.concat.forward([flattened_conv, flattened_rnn])

        hidden_output_0 = self.dense_1.forward(concat_output)
        hidden_output_1 = self.batchnorm_1.forward(hidden_output_0, training=train)
        hidden_output_2 = self.activation_1.forward(hidden_output_1)
        hidden_output_3 = self.dropout_1.forward(hidden_output_2, train=train)

        hidden_output_4 = self.dense_2.forward(hidden_output_3)
        hidden_output_5 = self.batchnorm_2.forward(hidden_output_4, training=train)
        hidden_output_6 = self.activation_2.forward(hidden_output_5)
        hidden_output_7 = self.dropout_2.forward(hidden_output_6, train=train)

        hidden_output_8 = self.dense_3.forward(hidden_output_7)
        hidden_output_9 = self.batchnorm_3.forward(hidden_output_8, training=train)
        hidden_output_10 = self.activation_3.forward(hidden_output_9)
        hidden_output_11 = self.dropout_3.forward(hidden_output_10, train=train)

        hidden_output_12 = self.dense_4.forward(hidden_output_11)
        hidden_output_13 = self.batchnorm_4.forward(hidden_output_12, training=train)
        hidden_output_14 = self.activation_4.forward(hidden_output_13)
        hidden_output_15 = self.dropout_4.forward(hidden_output_14, train=train)

        hidden_output_16 = self.dense_5.forward(hidden_output_15)
        output = self.activation_5.forward(hidden_output_16)

        return output

    def backward(self, grad):
        grad = self.activation_5.backward(grad)
        grad = self.dense_5.backward(grad)

        grad = self.dropout_4.backward(grad)
        grad = self.activation_4.backward(grad)
        grad = self.batchnorm_4.backward(grad)
        grad = self.dense_4.backward(grad)

        grad = self.dropout_3.backward(grad)
        grad = self.activation_3.backward(grad)
        grad = self.batchnorm_3.backward(grad)
        grad = self.dense_3.backward(grad)

        grad = self.dropout_2.backward(grad)
        grad = self.activation_2.backward(grad)
        grad = self.batchnorm_2.backward(grad)
        grad = self.dense_2.backward(grad)

        grad = self.dropout_1.backward(grad)
        grad = self.activation_1.backward(grad)
        grad = self.batchnorm_1.backward(grad)
        grad = self.dense_1.backward(grad)

        grad_concat = self.concat.backward(grad)

        grad = self.flatten_rnn.backward(grad_concat[1])
        grad = self.gru_2.backward(grad)
        grad = self.gru_1.backward(grad)

        grad = self.flatten_conv.backward(grad_concat[0])
        grad = self.min_pooling_2.backward(grad)
        grad = self.conv_2.backward(grad)
        grad = self.min_pooling_1.backward(grad)
        grad = self.conv_1.backward(grad)

    def SGD(self, learning_rate):
        self.dense_5.update(learning_rate)
        self.dense_4.update(learning_rate)
        self.dense_3.update(learning_rate)
        self.dense_2.update(learning_rate)
        self.dense_1.update(learning_rate)

        self.batchnorm_4.update(learning_rate)
        self.batchnorm_3.update(learning_rate)
        self.batchnorm_2.update(learning_rate)
        self.batchnorm_1.update(learning_rate)

        self.gru_2.update(learning_rate)
        self.gru_1.update(learning_rate)

        self.conv_2.update(learning_rate)
        self.conv_1.update(learning_rate)

    def binary_cross_entropy(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def compute_loss(self, y_pred, y_true):
        main_loss = self.binary_cross_entropy(y_pred, y_true)
        reg_loss = self.dense_1.get_regularization_loss() + self.dense_2.get_regularization_loss() + self.dense_3.get_regularization_loss()

        return main_loss + reg_loss
    
    def predict(self, X_img, X_vectors):
        return self.forward(X_img, X_vectors, train=False)


    def feature_importance(self, X_img, X_vectors, iters:int=50, epsilon=1e-8, sigma=0.01, top_print:int=3, max_povtors:int=1):

        print("начало FI")
        grad_img_all = []
        grad_vectors_all = []
        for _ in range(iters):
            output = self.forward(X_img + np.random.normal(0, sigma, X_img.shape), X_vectors + np.random.normal(0, sigma, X_vectors.shape), train=False)
            grad_output = np.ones_like(output)
            self.backward(grad_output)

            grad_img_all.append(self.conv_1.backward_output.copy())           #Последний градиент для изображеений
            grad_vectors_all.append(self.gru_1.backward_output.copy())    #Последний градиент для числовых данных
        
        
        grad_img = np.mean(np.abs(grad_img_all), axis=0)
        grad_vectors = np.mean(np.abs(grad_vectors_all), axis=0)

        grad_img_norm = grad_img / (np.sum(grad_img) + epsilon)    
        grad_vectors_norm = np.max((grad_vectors / (np.sum(grad_vectors) + epsilon)).reshape(X_vectors.shape[0], X_vectors.shape[-1]), axis=0)

        FI_img = 100 * grad_img_norm / (np.sum(grad_img_norm) + epsilon)
        FI_num = 100 * grad_vectors_norm / (np.sum(grad_vectors_norm) + epsilon)

        #print(f"\nShape Градиентов изображения: \n{FI_img.shape}\n")
        #print(f"\nГрадиенты изображения: \n{FI_img}\n")

        original_paths = ['C:/ZERODEFECT/WorkFlow/Frames/frame_0.jpg']

        # Загрузка оригинального изображения (как NumPy-массив)
        original = np.array(Image.open(original_paths[0]).convert('L'))  # 'L' для grayscale; уберите, если RGB

        # Градиентная тепловая карта
        grad_map = FI_img[0, 0, :, :]

        # Нормализация градиентов для [0, 1]
        if grad_map.max() - grad_map.min() != 0:
            norm_map = (grad_map - grad_map.min()) / (grad_map.max() - grad_map.min())
        else:
            norm_map = grad_map

        plt.figure(figsize=(6, 6))
        plt.imshow(original, cmap='gray')
        plt.imshow(norm_map, cmap='viridis', alpha=0.5) 
        plt.colorbar()
        plt.title(f'Тепловая карта')
        plt.axis('off')
        plt.show()

        try:
            conn = connect(
                host="000.000.0.000",
                dbname="process_1",
                user="user",
                password="admin",
                port="5432"
            )
            with conn.cursor() as cursor:
                cursor.execute("""INSERT INTO importance_levels (il_parameter_1, il_parameter_2, il_parameter_3, il_parameter_4, il_parameter_5, 
                                                                 il_parameter_6, il_parameter_7, il_parameter_8, il_parameter_9, il_parameter_10,
                                                                 il_parameter_11, 
                                                                 il_parameter_12, il_parameter_13,
                                                                 il_parameter_14,
                                                                 il_parameter_15, il_parameter_16, il_parameter_17,
                                                                 il_parameter_18, il_parameter_19, il_parameter_20,
                                                                 il_parameter_21, il_parameter_22, il_parameter_23) 
                               
                                  VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", 
                                  (FI_num[0], FI_num[1], FI_num[2], FI_num[3], FI_num[4], 
                                   FI_num[5], FI_num[6], FI_num[7], FI_num[8], FI_num[9],
                                   FI_num[10], 
                                   FI_num[11], FI_num[12], 
                                   FI_num[13], 
                                   FI_num[14], FI_num[15], FI_num[16], 
                                   FI_num[17], FI_num[18], np.mean(FI_num[22:27]),
                                   FI_num[19], FI_num[20], FI_num[21]))
                conn.commit()

        except Exception as e: 
            print(f"Ошибка в FI: {e}")
        finally: 
            conn.close()




    def train(self, X_img, X_vectors, y, epochs:int=2, learning_rate:float=0.001, batch_size:int=16, verbose:bool=True):
        losses = []
        for epoch in range(epochs):
            epoch_losses = []
            start_index = 0
            end_index = batch_size
            print('\n')
            start = datetime.datetime.now()
            for i in range(len(y) // batch_size):
    
                y_pred = self.forward(X_img[start_index:end_index], X_vectors[start_index:end_index])
    
                loss = self.compute_loss(y_pred, y[start_index:end_index])
                epoch_losses.append(loss)
    
                grad = (y_pred - y[start_index:end_index]) / y[start_index:end_index].shape[0]
    
                self.backward(grad)
                self.SGD(learning_rate)
                
                start_index += batch_size
                end_index += batch_size
                
                print(f"-------{i+1}/{len(y) // batch_size}-------", end='\r')
  
            losses.append(np.mean(epoch_losses))
            finish = datetime.datetime.now()

            print(f"\nEpoch {epoch}, Loss: {np.mean(epoch_losses):.4f}-------{finish-start}")

        return losses
    
    def save(self, path:str = "Updated_params.json"):
        dense_1 = (self.dense_1.weights.tolist(), self.dense_1.biases.tolist())
        dense_2 = (self.dense_2.weights.tolist(), self.dense_2.biases.tolist())
        dense_3 = (self.dense_3.weights.tolist(), self.dense_3.biases.tolist())
        dense_4 = (self.dense_4.weights.tolist(), self.dense_4.biases.tolist())
        dense_5 = (self.dense_5.weights.tolist(), self.dense_5.biases.tolist())

        batchnorm_1 = (self.batchnorm_1.gamma.tolist(), self.batchnorm_1.beta.tolist())
        batchnorm_2 = (self.batchnorm_2.gamma.tolist(), self.batchnorm_2.beta.tolist())
        batchnorm_3 = (self.batchnorm_3.gamma.tolist(), self.batchnorm_3.beta.tolist())
        batchnorm_4 = (self.batchnorm_4.gamma.tolist(), self.batchnorm_4.beta.tolist())

        gru_1 = (self.gru_1.Wz.tolist(), self.gru_1.Uz.tolist(), self.gru_1.bz.tolist(),
                 self.gru_1.Wr.tolist(), self.gru_1.Ur.tolist(), self.gru_1.br.tolist(),
                 self.gru_1.Wh.tolist(), self.gru_1.Uh.tolist(), self.gru_1.bh.tolist())
        gru_2 = (self.gru_2.Wz.tolist(), self.gru_2.Uz.tolist(), self.gru_2.bz.tolist(),
                 self.gru_2.Wr.tolist(), self.gru_2.Ur.tolist(), self.gru_2.br.tolist(),
                 self.gru_2.Wh.tolist(), self.gru_2.Uh.tolist(), self.gru_2.bh.tolist())
        
        conv_1 = (self.conv_1.weights.tolist(), self.conv_1.bias.tolist())
        conv_2 = (self.conv_2.weights.tolist(), self.conv_2.bias.tolist())

        params = {

            "dense_1":{
                "weights": dense_1[0],
                "biases": dense_1[1]
            },
            "dense_2":{
                "weights": dense_2[0],
                "biases": dense_2[1]
            },
            "dense_3":{
                "weights": dense_3[0],
                "biases": dense_3[1]
            },
            "dense_4":{
                "weights": dense_4[0],
                "biases": dense_4[1]
            },
            "dense_5":{
                "weights": dense_5[0],
                "biases": dense_5[1]
            },

            "batchnorm_1":{
                "gamma": batchnorm_1[0],
                "beta": batchnorm_1[1]
            },
            "batchnorm_2":{
                "gamma": batchnorm_2[0],
                "beta": batchnorm_2[1]
            },
            "batchnorm_3":{
                "gamma": batchnorm_3[0],
                "beta": batchnorm_3[1]
            },
            "batchnorm_4":{
                "gamma": batchnorm_4[0],
                "beta": batchnorm_4[1]
            },

            "gru_1":{
                "Wz": gru_1[0],
                "Uz": gru_1[1],
                "bz": gru_1[2],
                "Wr": gru_1[3],
                "Ur": gru_1[4],
                "br": gru_1[5],
                "Wh": gru_1[6],
                "Uh": gru_1[7],
                "bh": gru_1[8]
            },
            "gru_2":{
                "Wz": gru_2[0],
                "Uz": gru_2[1],
                "bz": gru_2[2],
                "Wr": gru_2[3],
                "Ur": gru_2[4],
                "br": gru_2[5],
                "Wh": gru_2[6],
                "Uh": gru_2[7],
                "bh": gru_2[8]
            },

            "conv_1":{
                "weights": conv_1[0],
                "biases": conv_1[1],
            },
            "conv_2":{
                "weights": conv_2[0],
                "biases": conv_2[1],
            }
        }


        with open(path, "a") as f:
            f.write(json.dumps(params, indent=4))
        return

    def load_params(self, path:str = "Updated_params.json"):

        with open(path, "r") as file:
            data = json.load(file)

        self.dense_1.weights, self.dense_1.biases = np.array(data['dense_1']['weights']), np.array(data['dense_1']['biases']) 
        self.dense_2.weights, self.dense_2.biases = np.array(data['dense_2']['weights']), np.array(data['dense_2']['biases']) 
        self.dense_3.weights, self.dense_3.biases = np.array(data['dense_3']['weights']), np.array(data['dense_3']['biases']) 
        self.dense_4.weights, self.dense_4.biases = np.array(data['dense_4']['weights']), np.array(data['dense_4']['biases']) 
        self.dense_5.weights, self.dense_5.biases = np.array(data['dense_5']['weights']), np.array(data['dense_5']['biases']) 

        self.batchnorm_1.gamma, self.batchnorm_1.beta = np.array(data['batchnorm_1']['gamma']), np.array(data['batchnorm_1']['beta']) 
        self.batchnorm_2.gamma, self.batchnorm_2.beta = np.array(data['batchnorm_2']['gamma']), np.array(data['batchnorm_2']['beta']) 
        self.batchnorm_3.gamma, self.batchnorm_3.beta = np.array(data['batchnorm_3']['gamma']), np.array(data['batchnorm_3']['beta']) 
        self.batchnorm_4.gamma, self.batchnorm_4.beta = np.array(data['batchnorm_4']['gamma']), np.array(data['batchnorm_4']['beta']) 

        self.gru_1.Wz, self.gru_1.Uz, self.gru_1.bz = np.array(data['gru_1']["Wz"]), np.array(data['gru_1']["Uz"]), np.array(data['gru_1']["bz"])
        self.gru_1.Wr, self.gru_1.Ur, self.gru_1.br = np.array(data['gru_1']["Wr"]), np.array(data['gru_1']["Ur"]), np.array(data['gru_1']["br"])
        self.gru_1.Wh, self.gru_1.Uh, self.gru_1.bh = np.array(data['gru_1']["Wh"]), np.array(data['gru_1']["Uh"]), np.array(data['gru_1']["bh"])

        self.gru_2.Wz, self.gru_2.Uz, self.gru_2.bz = np.array(data['gru_2']["Wz"]), np.array(data['gru_2']["Uz"]), np.array(data['gru_2']["bz"])
        self.gru_2.Wr, self.gru_2.Ur, self.gru_2.br = np.array(data['gru_2']["Wr"]), np.array(data['gru_2']["Ur"]), np.array(data['gru_2']["br"])
        self.gru_2.Wh, self.gru_2.Uh, self.gru_2.bh = np.array(data['gru_2']["Wh"]), np.array(data['gru_2']["Uh"]), np.array(data['gru_2']["bh"])

        self.conv_1.weights, self.conv_1.bias = np.array(data['conv_1']['weights']), np.array(data['conv_1']['biases'])
        self.conv_2.weights, self.conv_2.bias = np.array(data['conv_2']['weights']), np.array(data['conv_2']['biases'])