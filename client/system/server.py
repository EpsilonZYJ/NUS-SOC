import paho.mqtt.client as mqtt
from tensorflow.keras.models import load_model
import numpy as np
import json
from PIL import Image
from password_load import load_password
import time
import matplotlib.pyplot as plt
import os
import threading
from queue import Queue

classes = ["daisy", "dandelion", "roses", "sunflowers", "tulips"] # used for test
cat_classes = ['pallas', 'persian', 'ragdoll', 'singapura', 'sphynx']  # used for cat classification
models = ['efficientnet', 'xception', 'test']  # available models for cat classification

model_dict = {
    "efficientnet": {
        "path": "model/cats_efficientnetb0-Noise-Brightness-V1.keras",
        "input_size": 224,
        "scale": 255.0,  # scale factor for EfficientNet
        "classes": cat_classes
    },
    "xception": {
        "path": "model/cat_classifier_xception.h5",
        "input_size": 299,
        "scale": 255.0,  # scale factor for Xception
        "classes": cat_classes
    },
    "test": {
        "path": "model/flowers.keras",
        "input_size": 224,
        "scale": 1.0,  # scale factor for test model
        "classes": classes
    },
    "insection": {
        "path": "model/cats_insection.keras",
        "input_size": 224,
        "scale": 1.0,  # scale factor for insection model
        "classes": cat_classes
    }
}
        
FILENAME = 'flowers.keras'  # used for test
EFFICIENTNET_FILENAME = 'modelpara/cat_classifier_efficientnet.h5'
XCEPTION_FILENAME = 'modelpara/cat_classifier_xception.h5'  # used for cat classification
IMAGE_FILEPATH = 'result_image.json' # information about the classified images will be saved here

class MQTTInferenceServer:
    """
    A simple MQTT inference server that listens for image classification requests.
    """
    def __init__(self, hostname, password_path="mqtt.pwd", model_name='test'):
        # self.model_name = model_name
        if model_name not in model_dict:
            raise ValueError(f"Model '{model_name}' is not supported. Available models: {list(model_dict.keys())}")
        # self.classes = model_dict[model_name]['classes']
        print("Loading model from ", model_dict[model_name]['path'])
        try:
            # self.model = load_model(model_dict[model_name])
            # self.model_params = model_dict[model_name]
            self.init_model(model_name)
            print("Model loaded successfully.")
        except Exception as e:
            print("Error loading model:", e)
            try:
                # self.model = load_model(model_dict[model_name]['path'], compile=False)
                # self.model_params = model_dict[model_name]
                self.init_model(model_name, compile=False)
                print("Model loaded without compilation.")
            except Exception as e2:
                print("Error loading model without compilation:", e2)
                raise e2
        print("Done.")
        print("Connecting to broker")
        self.client = mqtt.Client()
        username, password = load_password(password_path)
        self.client.username_pw_set(username, password)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(hostname, 1883, 60)

        # 添加图像保存队列和线程
        self.image_queue = Queue()
        self.plot_thread = threading.Thread(target=self._plot_worker, daemon=True)
        self.plot_thread.start()

        print("Done")
    
    # def load_image_for_models(self, img_data):
    #     input_size = self.model_params['input_size']
    #     img_data = Image.resize(input_size, input_size)  # Resize the image
    #     img_data = np.array(img_data) * self.model_params['scale']  # Scale the image data
    #     final = np.expand_dims(img_data, axis=0)  # Add batch dimension
    #     return final

    def load_image_for_models(self, img_data):
        """
        根据模型类型处理图像数据
        """
        # 获取模型参数
        input_size = self.model_params['input_size']
        scale = self.model_params['scale']
        
        # 将接收的数据转换为 numpy 数组
        img_array = np.array(img_data)
        
        # 处理数组维度
        if img_array.ndim == 4 and img_array.shape[0] == 1:
            img_array = img_array[0]  # 移除批次维度
        
        # 确保数据在正确范围内
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
        
        # 转换为 PIL Image
        img_pil = Image.fromarray(img_array)
        img_resized = img_pil.resize((input_size, input_size), Image.LANCZOS)
        # 转换回 numpy 数组
        img_final = np.array(img_resized)
        
        # 根据模型应用不同的缩放
        if scale == 1.0:
            # 对于 test 模型，归一化到 0-1
            img_final = img_final.astype(np.float32) / 255.0
        else:
            # 对于 EfficientNet 和 Xception，保持 0-255 范围
            img_final = img_final.astype(np.float32)
        
        # 添加批次维度
        return np.expand_dims(img_final, axis=0)


    def classify(self, image):
        result = self.model.predict(image)
        themax = np.argmax(result)
        print("Result: ", result)
        return (self.classes[themax], result[0][themax], themax)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected.")
            self.client.subscribe("Group19/IMAGE/classify")
        else:
            print("Failed to connect. Error code: ", rc)

    def classify_result(self, filename, data):
        print("Start classifying.")
        label, prob, index = self.classify(data)
        print("Done.")

        return {
            "filename": filename,
            "prediction": label,
            "score": float(prob),
            "index": str(index)
        }
    
    def init_model(self, model_name, compile=True):
        try:
            self.client.loop_stop()
            self.model_name = model_name
            self.classes = model_dict[model_name]['classes']
            self.model = load_model(model_dict[model_name]['path'], compile=compile)
            self.model_params = model_dict[model_name]
            self.client.loop_start()    
            print(f"Model {model_name} initialized")
        except Exception as e:
            print(f"Error loading model from {model_name}: {e}")
        # if os.path.exists(modelpath):
        #     try:
        #         self.client.loop_stop()
        #         self.model = load_model(modelpath)
        #         self.client.loop_start()
        #         print(f"Model loaded from {modelpath}")
        #     except Exception as e:
        #         print(f"Error loading model from {modelpath}: {e}")

    def on_message(self, client, userdata, msg):
        recv_dict = json.loads(msg.payload)
        # img_data = np.array(recv_dict["data"])
        img_data = self.load_image_for_models(recv_dict["data"])
        result = self.classify_result(recv_dict["filename"], img_data)
        # print(recv_dict["data"])
        self.plot_image(img_data, recv_dict, result)
        print("Sending results.")
        self.client.publish("Group19/IMAGE/predict", json.dumps(result))
    
    # def plot_image(self, img_data, recv_dict, result):
    #     if img_data.ndim == 4 and img_data.shape[0] == 1:
    #         img_display = img_data[0]
    #     else:
    #         img_display = img_data

    #     if img_display.max() <= 1.0:
    #         img_display = (img_display * 255).astype(np.uint8)
    #     else:
    #         img_display = img_display.astype(np.uint8)
    #     # plot the image
    #     plt.figure(figsize=(8, 6))
    #     plt.imshow(img_display)
    #     plt.title(f"File: {recv_dict['filename']}\nPrediction: {result['prediction']} (Score: {result['score']:.3f})\nModel: {self.model_name}")
    #     plt.axis('off')
    #     # Create results directory if it doesn't exist
    #     results_dir = "results"
    #     if not os.path.exists(results_dir):
    #         os.makedirs(results_dir)
    #     # Save the plot to the results directory
    #     # output_filename = f"result_{recv_dict['filename'].split('/')[-1]}"
    #     output_filename = f"{results_dir}/result_{recv_dict['filename'].split('/')[-1]}"
    #     plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    #     plt.close() 
    #     print(f"Result saved to {output_filename}")

    #     image_info = {
    #         "filename": recv_dict["filename"],
    #         "prediction": result["prediction"],
    #         "score": result["score"],
    #         "index": result["index"],
    #         "image_path": output_filename
    #     }
    #     # Save the image information to a JSON file
    #     try:
    #         if os.path.exists(IMAGE_FILEPATH):
    #             with open(IMAGE_FILEPATH, 'r', encoding='utf-8') as f:
    #                 data = json.load(f)
    #         else:
    #             data = []
    #     except (json.JSONDecodeError, FileNotFoundError):
    #         data = []

    #     data.append(image_info)
    #     with open(IMAGE_FILEPATH, 'w', encoding='utf-8') as f:
    #         json.dump(data, f, indent=4, ensure_ascii=False)
    #         f.write('\n')

    def _plot_worker(self):
        """
        在单独线程中处理图像绘制，避免GUI线程问题
        """
        import matplotlib
        matplotlib.use('Agg')  # 确保使用非交互式后端
        import matplotlib.pyplot as plt
        
        while True:
            try:
                item = self.image_queue.get()
                if item is None:  # 退出信号
                    break
                
                img_data, recv_dict, result = item
                self._plot_image_worker(img_data, recv_dict, result, plt)
                self.image_queue.task_done()
            except Exception as e:
                print(f"Error in plot worker: {e}")
    
    def _plot_image_worker(self, img_data, recv_dict, result, plt):
        """
        实际的图像绘制工作
        """
        if img_data.ndim == 4 and img_data.shape[0] == 1:
            img_display = img_data[0]
        else:
            img_display = img_data

        if img_display.max() <= 1.0:
            img_display = (img_display * 255).astype(np.uint8)
        else:
            img_display = img_display.astype(np.uint8)

        plt.figure(figsize=(8, 6))
        plt.imshow(img_display)
        plt.title(f"File: {recv_dict['filename']}\nPrediction: {result['prediction']} (Score: {result['score']:.3f})\nModel: {self.model_name}")
        plt.axis('off')
        
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        output_filename = f"{results_dir}/result_{recv_dict['filename'].split('/')[-1]}"
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Result saved to {output_filename}")

        # 保存图像信息
        image_info = {
            "filename": recv_dict["filename"],
            "prediction": result["prediction"],
            "score": result["score"],
            "index": result["index"],
            "image_path": output_filename
        }
        self._save_image_info(image_info)

    def _save_image_info(self, image_info):
        """
        保存图像信息到JSON文件
        """
        try:
            if os.path.exists(IMAGE_FILEPATH):
                with open(IMAGE_FILEPATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = []
        except (json.JSONDecodeError, FileNotFoundError):
            data = []

        data.append(image_info)
        
        with open(IMAGE_FILEPATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.write('\n')

    def plot_image(self, img_data, recv_dict, result):
        """
        将绘制任务添加到队列
        """
        self.image_queue.put((img_data, recv_dict, result))

    def load_image_info(self):
        # path of the image information JSON file
        return IMAGE_FILEPATH

    def start_server(self):
        self.client.loop_start()
        try:
            while True:
                time.sleep(1)  # Keep the server running
        except KeyboardInterrupt:
            print("Server stopped by user.")
            self.client.loop_stop()
            self.client.disconnect()

if __name__ == "__main__":
    # server = MQTTInferenceServer(hostname="192.168.43.58", classes=cat_classes, model_path=EFFICIENTNET_FILENAME)
    server = MQTTInferenceServer(hostname="127.0.0.1", model_name='efficientnet')  # Change to 'efficientnet' or 'test' as needed
    server.start_server()