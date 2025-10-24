from Model import NeuralNetwork
from scaler import MinMaxScaler
from psycopg2 import connect
import numpy as np
from PIL import Image
from io import BytesIO
from time import sleep


def process_1(path:str=""):
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.load_scaler(f"{path}/scaler_params.json")

        model = NeuralNetwork(num_features=27)
        model.load_params(f"{path}/Saved_model.json")
        
        conn = connect(
            host="000.000.0.000",
            dbname="process_1",
            user="usre",
            password="admin",
            port="5432"
        )
            
        while True:    
            first_result = None
            second_result = None
            third_result = None
            camera_result = None

            with conn.cursor() as cursor:
                cursor.execute("""SELECT id_process, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5, parameter_6, parameter_7, parameter_8, parameter_9, parameter_10, 
                                         parameter_11, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17
                                  FROM first_processes ORDER BY id_process DESC LIMIT 10;""")
                first_result = cursor.fetchall()
                conn.commit()

            with conn.cursor() as cursor:
                cursor.execute("""SELECT parameter_18, parameter_19, parameter_20 FROM second_processes ORDER BY id_process DESC LIMIT 10;""")
                second_result = cursor.fetchall()
                conn.commit()

            with conn.cursor() as cursor:
                cursor.execute("""SELECT frame FROM camera ORDER BY id_process DESC LIMIT 10;""")
                camera_result = cursor.fetchall()
                conn.commit()

            with conn.cursor() as cursor:
                cursor.execute("""SELECT parameter_21, parameter_22, parameter_23 FROM third_processes ORDER BY id_process DESC LIMIT 10;""")
                third_result = cursor.fetchall()
                conn.commit()


            X_numeric = np.zeros((10, 27))
            X_img = np.zeros((10, 384, 384))

            for ss in range(10):
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

            X_numeric = scaler.transform(X_numeric).reshape(10, 1, 27)

            prediction = model.predict(X_img.reshape(10, 1, 384, 384), X_numeric)
            print(f"\nid_process: {first_result[0][0]}")
            print(f"ПРЕДСКАЗАНИЕ СДЕЛАНО: {prediction[0][0]*100}")

            with conn.cursor() as cursor:
                cursor.execute("""INSERT INTO prediction (id_process, predict) 
                                  VALUES (%s, %s)""", (first_result[0][0], round((prediction[0][0]*100).astype("float64"), 2)))
                conn.commit()

            model.feature_importance(X_img=np.array(X_img).reshape(10, 1, 384, 384), X_vectors=X_numeric, iters=2)
            sleep(20)

    except Exception as e:
        print(f"Err: {e}")
    finally:
        conn.close()

process_1("C:/WorkFlow_path")