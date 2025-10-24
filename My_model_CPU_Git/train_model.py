from Model import NeuralNetwork
import numpy as np
from PIL import Image
import random as r
from scaler import MinMaxScaler
from psycopg2 import connect
from io import BytesIO
from PIL import Image
import datetime


def create_dataset():
    syrie_one_hot = {
        "один": [1, 0, 0, 0, 0],
        "два": [0, 1, 0, 0, 0],
        "один + два": [0, 0, 1, 0, 0],
        "три": [0, 0, 0, 1, 0],
        "четыре": [0, 0, 0, 0, 1]
    }

    train_data = [[] for _ in range(1000)]

    for i in range(1000):
        train_data[i].append([r.randrange(150, 153), r.randrange(158, 161), r.randrange(166, 169), r.randrange(174, 177), r.randrange(182, 185),
                             r.randrange(140, 143), r.randrange(140, 143), r.randrange(140, 143), r.randrange(140, 143), r.randrange(140, 143), 
                             r.randrange(50, 53),       
                             r.randrange(7, 9),         
                             r.randrange(50, 53),       
                             r.randrange(30, 35),       
                             r.randrange(2, 5),         
                             r.randrange(2, 4),         
                             r.randrange(1, 3),         
                             r.uniform(0.05, 0.1),      
                             r.randrange(40, 50)])      

        if(i < 200):                                   
            train_data[i][0].extend([10, 20, 4, syrie_one_hot["один"][0], syrie_one_hot["один"][1], syrie_one_hot["один"][2], syrie_one_hot["один"][3], syrie_one_hot["один"][4]])
        elif((i >= 200) and (i < 400)):
            train_data[i][0].extend([18, 16, 2.5, syrie_one_hot["два"][0], syrie_one_hot["два"][1], syrie_one_hot["два"][2], syrie_one_hot["два"][3], syrie_one_hot["два"][4]])
        elif((i >= 400) and (i < 600)):
            train_data[i][0].extend([20, 14, 2, syrie_one_hot["один + два"][0], syrie_one_hot["один + два"][1], syrie_one_hot["один + два"][2], syrie_one_hot["один + два"][3], syrie_one_hot["один + два"][4]])
        elif((i >= 600) and (i < 800)):
            train_data[i][0].extend([25, 20, 3.5, syrie_one_hot["три"][0], syrie_one_hot["три"][1], syrie_one_hot["три"][2], syrie_one_hot["три"][3], syrie_one_hot["три"][4]])
        else:
            train_data[i][0].extend([29, 17, 4.3, syrie_one_hot["четыре"][0], syrie_one_hot["четыре"][1], syrie_one_hot["четыре"][2], syrie_one_hot["четыре"][3], syrie_one_hot["четыре"][4]])

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_last = scaler.fit_transform(np.array(train_data))

    camera = np.zeros((1000, 384, 384))
    for i in range(100):
        with Image.open(f"C:/Frames/frame_{i}.jpg") as img:
            gray_img = np.array(img.convert('L')).astype("float")   # или сразу в этой строке /255.0
            for j in range(384):
              for k in range(384):
                gray_img[j][k] /= 255.0
        camera[i] = gray_img

    y = []
    for i in range(1000):
        y.append(r.randrange(0, 2))
    y = np.array(y).reshape(-1, 1)

    scaler.save("scaler_numeric.json")

    return camera, train_last, y

#Расскометировать код ниже, если необходимо более 100 примеров с изображениями
#camera_train, numeric_train, y = create_dataset()
#
#for _ in range(9):
#    for i in range(100):
#        camera_train[i+100] = camera_train[i]
#
#camera_train = camera_train.reshape(1000, 1, 384, 384)

def load_dataset_from_db():
    try:
        start = datetime.datetime.now()
        conn = connect(
            host="000.000.0.000",
            dbname="process_1",
            user="user",
            password="admin",
            port="5432"
        )
        end = datetime.datetime.now()
        print(f"Время подключения к БД: {end-start}")
            
        first_result = None
        second_result = None
        third_result = None
        camera_result = None

        start = datetime.datetime.now()

        with conn.cursor() as cursor:
            cursor.execute("""SELECT id_process, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5, parameter_6, parameter_7, parameter_8, parameter_9, parameter_10, 
                                         parameter_11, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17
                                  FROM first_processes ORDER BY id_process DESC LIMIT 100;""")
            first_result = cursor.fetchall()
            conn.commit()

        with conn.cursor() as cursor:
            cursor.execute("""SELECT parameter_18, parameter_19, parameter_20 FROM second_processes ORDER BY id_process DESC LIMIT 100;""")
            second_result = cursor.fetchall()
            conn.commit()

        with conn.cursor() as cursor:
            cursor.execute("""SELECT frame FROM camera ORDER BY id_process DESC LIMIT 100;""")
            camera_result = cursor.fetchall()
            conn.commit()

        with conn.cursor() as cursor:
            cursor.execute("""SELECT parameter_21, parameter_22, parameter_23 FROM third_processes ORDER BY id_process DESC LIMIT 100;""")
            third_result = cursor.fetchall()
            conn.commit()

        with conn.cursor() as cursor:
            cursor.execute("""SELECT bad_flag FROM y_true ORDER BY id_process DESC LIMIT 100;""")
            y = cursor.fetchall()
            conn.commit()
        
        end = datetime.datetime.now()
        print(f"Время высасывания данных: {end-start}")

        start = datetime.datetime.now()

        X_numeric = np.zeros((100, 27))
        X_img = np.zeros((100, 384, 384))

        for ss in range(100):
            class_ = [0, 0, 0, 0, 0]
            for i in range(len(second_result[ss][2])):
                if(len(second_result[ss][2]) == 1):
                    class_[second_result[ss][2][i]-1] = 1
                else:
                    class_[second_result[ss][2][i][0]-1] = 1

            X_numeric[ss] = np.array([first_result[ss][1], first_result[ss][2], first_result[ss][3], first_result[ss][4], first_result[ss][5],
                first_result[ss][6], first_result[ss][7], first_result[ss][8], first_result[ss][9], first_result[ss][10],
                first_result[ss][11],
                first_result[ss][12], first_result[ss][13],
                first_result[ss][14],
                first_result[ss][15], first_result[ss][16], first_result[ss][17],
                int(second_result[ss][0]), second_result[ss][1],
                third_result[ss][0], third_result[ss][1], third_result[ss][2],
                class_[0], class_[1], class_[2], class_[3], class_[4]])
            
            X_img[ss] = np.array(Image.open(BytesIO(camera_result[ss][0]))) / 255.0

        scaler = MinMaxScaler(feature_range=(0, 1))
        X_numeric = scaler.fit_transform(X_numeric)
        scaler.save()
        
        y = np.array(y).astype(int)

        end = datetime.datetime.now()
        print(f"Время преобразования данных перед подачей в нейронку: {end-start}")

        return X_img.reshape(100, 1, 384, 384), X_numeric.reshape(100, 1, 27), y

    except Exception as e: 
        conn.close()
        print(f"Err: {e}")
        return
    
    finally:
        conn.close()


camera_train, numeric_train, y = load_dataset_from_db()

model = NeuralNetwork(num_features=27)
losses = model.train(camera_train, numeric_train, y, epochs=2, learning_rate=0.01, batch_size=10, verbose=True)
model.save("Saved_model.json")


