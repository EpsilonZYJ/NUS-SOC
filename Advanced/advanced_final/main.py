import argparse
import paho.mqtt.client as mqtt
import json
import run_multi_model as multi_model_decision
# import multi_model_decision1 as multi_model_decision
import time
import threading

class InstructionGiver:
    def __init__(self, hostname: str, password_path: str="mqtt.pwd") -> None:
        self.hostname = hostname
        self.client = mqtt.Client()
        username, password = self._load_password(password_path)
        
        # 状态管理
        self.current_state = "normal"
        self.last_published_state = None
        self.last_published_direction = None
        self.state_lock = threading.Lock()
        self.direction_lock = threading.Lock()
        
        try:
            if username and password:
                self.client.username_pw_set(username, password)
            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message
            self.client.connect(hostname, 1883, 60)
            self.send_status: bool = False
        except Exception as e:
            print(f"Error setting up MQTT client: {e}")
            raise e
    
    def _load_password(self, path="mqtt.pwd"):
        try:
            with open(path, "r") as f:
                line = f.readline().strip()
                username, password = line.split()
                return username, password
        except Exception as e:
            print(f"Error loading password: {e}")
            print("将使用匿名连接")
            return None, None

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT broker")
            self.client.subscribe("Group19/CONTROL")
            self.send_status = True
        else:
            print(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        recv_dict = json.loads(msg.payload)
        if recv_dict["status"] == "found":
            self.send_status = False
        else:
            self.send_status = True

    def state_change_callback(self, new_state):
        """状态变化回调函数"""
        with self.state_lock:
            self.current_state = new_state
            self.publish_state(new_state)

    def direction_change_callback(self, direction_info):
        """方向变化回调函数"""
        with self.direction_lock:
            direction = direction_info['direction']
            target_id = direction_info.get('target_id')
            
            # 避免重复发送相同的方向指令
            # current_direction_key = f"{direction}_{target_id}"
            # if self.last_published_direction != current_direction_key:
            self.publish_direction(direction_info)
                # self.last_published_direction = current_direction_key

    def publish_state(self, state):
        """发布状态到 MQTT"""
        if self.last_published_state != state:
            state_message = {
                "timestamp": time.time(),
                "state": state,
                "device_id": "camera_detector"
            }
            
            try:
                self.client.publish("Group19/CONTROL/state", json.dumps(state_message))
                self.last_published_state = state
                print(f"状态已发布到 MQTT: {state}")
            except Exception as e:
                print(f"发布状态失败: {e}")

    def publish_direction(self, direction_info):
        """发布方向指令到 MQTT"""
        try:
            direction_message = {
                "timestamp": direction_info['timestamp'],
                "direction": direction_info['direction'],
                "target_id": direction_info.get('target_id'),
                "state": direction_info['state'],
                "device_id": "camera_detector"
            }
            
            self.client.publish("Group19/CONTROL/direction", json.dumps(direction_message))
            print(f"方向指令已发布到 MQTT: {direction_info['direction']} (目标ID: {direction_info.get('target_id')})")
            
        except Exception as e:
            print(f"发布方向指令失败: {e}")

    # def publish(self, instruction: str) -> None:
    #     """发布指令到 MQTT (保持原有接口)"""
    #     if self.send_status:
    #         instruction_dict = {
    #             "instruction": instruction,
    #             "timestamp": time.time(),
    #             "current_state": self.current_state
    #         }
    #         self.client.publish("Group19/CONTROL/direction", json.dumps(instruction_dict))

    def start_periodic_state_publish(self):
        """启动定期状态发布线程"""
        def publish_thread():
            while True:
                with self.state_lock:
                    if self.current_state:
                        self.publish_state(self.current_state)
                time.sleep(5)  # 每5秒发布一次状态
        
        thread = threading.Thread(target=publish_thread, daemon=True)
        thread.start()
    
    def start_server(self, opt) -> None:
        """启动服务器"""
        self.client.loop_start()
        
        # 启动定期状态发布
        self.start_periodic_state_publish()
        
        try:
            # 创建检测器，传入状态和方向回调函数
            detector = multi_model_decision.MultiModelCameraDetector(
                device=opt.device,
                state_callback=self.state_change_callback,
                direction_callback=self.direction_change_callback  # 新增方向回调
            )
            
            # 启动检测
            detector.run_detection(
                camera=opt.camera,
                conf_thres=opt.conf_thres,
                iou_thres=opt.iou_thres
            )
            
        except KeyboardInterrupt:
            print("Server stopped by user.")
        finally:
            self.client.loop_stop()
            self.client.disconnect()
    
    def stop_server(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID (默认: 0), Pi摄像头ID: 1')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU阈值')
    parser.add_argument('--device', default='', help='设备选择 (cpu, 0, 1, 2, 3)')
    parser.add_argument('--mqtt-host', default="192.168.43.8", help='MQTT 服务器地址')
    
    opt = parser.parse_args()
    
    print(f"启动 MQTT 客户端，连接到: {opt.mqtt_host}")
    server = InstructionGiver(hostname=opt.mqtt_host)
    server.start_server(opt)

if __name__ == '__main__':
    main()