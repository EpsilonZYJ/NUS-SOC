import os
import cv2
import time
import torch
import argparse
from pathlib import Path
from numpy import random
from random import randint
import torch.backends.cudnn as cudnn
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import math

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, \
                time_synchronized, TracedModel
from utils.download_weights import download

from autorun import DirectionTracker

# For SORT tracking
# import skimage
from sort import *

# 模型配置字典
model_dict = {
    'smoke': 'trained_model/smoke.pt',
    'litter': 'trained_model/new_trash_best.pt', 
    'fall': 'trained_model/fall.pt',
    'fight': 'trained_model/fight.pt',
}
FLASK_URL = 'http://192.168.43.8:5000/video'

def draw_boxes(img, bbox, identities=None, categories=None, names=None, model_name=None, offset=(0, 0)):
    """绘制边界框和标签"""
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        
        # 为不同模型设置不同颜色
        model_colors = {
            'smoke': (0, 255, 0),    # 绿色
            'litter': (255, 0, 0),   # 蓝色
            'fall': (0, 0, 255),     # 红色
            'fight': (255, 255, 0)   # 青色
        }
        color = model_colors.get(model_name, (255, 0, 20))
        
        # 创建标签
        if model_name in ['smoke', 'litter']:
            # 追踪模型显示ID
            label = f"{model_name}:{names[cat]}:{id}"
        else:
            # 普通检测模型只显示类别
            label = f"{model_name}:{names[cat]}"
        
        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签背景
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        
        # 绘制标签文字
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, [255, 255, 255], 1)
    
    return img

class FlaskCameraStream:
    """从Flask应用获取摄像头流的类"""
    
    def __init__(self, url='http://localhost:5000/video'):
        self.url = url
        self.stream = None
        self.connected = False
        
    def connect(self):
        """连接到Flask视频流"""
        try:
            # 创建视频捕获对象
            self.stream = cv2.VideoCapture(self.url)
            if self.stream.isOpened():
                self.connected = True
                print(f"成功连接到Flask摄像头流: {self.url}")
                return True
            else:
                print(f"无法连接到Flask摄像头流: {self.url}")
                return False
        except Exception as e:
            print(f"连接Flask摄像头流时出错: {e}")
            return False
    
    def read(self):
        """读取一帧"""
        if not self.connected or self.stream is None:
            return False, None
        
        try:
            ret, frame = self.stream.read()
            if not ret:
                # 尝试重新连接
                self.connected = False
                print("连接断开，尝试重新连接...")
                time.sleep(1)
                self.connect()
                return False, None
            return True, frame
        except Exception as e:
            print(f"读取帧时出错: {e}")
            return False, None
    
    def release(self):
        """释放资源"""
        if self.stream is not None:
            self.stream.release()
        self.connected = False

class MultiModelCameraDetector:
    
    def __init__(self, device='', state_callback=None, direction_callback=None):
        self.device = select_device(device)
        self.models = {}
        self.trackers = {}
        self.load_all_models()
        self.flask_camera = FlaskCameraStream(FLASK_URL)
        self.model_detect = {}
        self.results = {}
        
        # 添加状态和方向回调函数
        self.state_callback = state_callback
        self.direction_callback = direction_callback  # 新增方向回调
        self.current_state = "normal"  # 将状态变量作为实例变量
        
        # 为追踪模型创建SORT追踪器
        self.trackers['smoke'] = Sort(max_age=5, min_hits=2, iou_threshold=0.2)
        self.trackers['litter'] = Sort(max_age=5, min_hits=2, iou_threshold=0.2)
        
        # 为追踪模型生成随机颜色
        self.rand_color_list = []
        amount_rand_color_prime = 5003
        for i in range(0, amount_rand_color_prime):
            r = randint(0, 255)
            g = randint(0, 255)
            b = randint(0, 255)
            rand_color = (r, g, b)
            self.rand_color_list.append(rand_color)
        
        # 性能统计
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.avg_fps = 0
        
        # 垃圾归属判断器
        self.litter_checker = LitterChecker(proximity_threshold=150)
    
    def load_all_models(self):
        """预加载所有模型"""
        print("正在预加载所有模型...")
        start_time = time.time()
        
        for model_name, weight_path in model_dict.items():
            try:
                print(f"  加载模型: {model_name}")
                # 使用 attempt_load 加载 YOLO 模型
                model = attempt_load(weight_path, map_location=self.device)
                
                if self.device.type != 'cpu':
                    model.half()
                
                # 预热模型
                model(torch.zeros(1, 3, 640, 640).to(self.device).type_as(next(model.parameters())))
                
                self.models[model_name] = model
                print(f"  {model_name} 加载完成")
                
            except Exception as e:
                print(f"  {model_name} 加载失败: {str(e)}")
    
        load_time = time.time() - start_time
        print(f"所有模型加载完成，用时: {load_time:.2f}s")
    
    def detect_with_tracking(self, model_name, frame, conf_thres=0.5, iou_thres=0.45):
        """使用追踪功能的检测（用于smoke和litter）"""
        try:
            model = self.models[model_name]
            tracker = self.trackers[model_name]
            
            # 获取类别名称
            names = model.module.names if hasattr(model, 'module') else model.names
            
            # 使用letterbox预处理图像，保持宽高比
            img, ratio, pad = letterbox(frame, 640, stride=32, auto=True)
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.device.type != 'cpu' else img.float()
            img /= 255.0
            img = img.unsqueeze(0)
            
            # 推理
            with torch.no_grad():
                pred = model(img, augment=False)[0]
            
            # 应用NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
            
            # 处理检测结果
            dets_to_sort = np.empty((0, 6))
            
            for i, det in enumerate(pred):
                if len(det):
                    # 缩放边界框到原始图像尺寸，传入正确的ratio和pad信息
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape, ratio_pad=(ratio, pad)).round()
                    
                    # 准备追踪数据
                    for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                        dets_to_sort = np.vstack((dets_to_sort, 
                                    np.array([x1, y1, x2, y2, conf, detclass])))
            
            # 运行SORT追踪
            tracked_dets = tracker.update(dets_to_sort)
            tracks = tracker.getTrackers()
            
            # 绘制追踪轨迹
            for track in tracks:
                # 绘制彩色追踪轨迹
                [cv2.line(frame, (int(track.centroidarr[i][0]),
                                int(track.centroidarr[i][1])), 
                                (int(track.centroidarr[i+1][0]),
                                int(track.centroidarr[i+1][1])),
                                self.rand_color_list[track.id % len(self.rand_color_list)], thickness=2) 
                                for i, _ in enumerate(track.centroidarr) 
                                if i < len(track.centroidarr)-1]
            
            # 绘制边界框
            detections_info = []
            if len(tracked_dets) > 0:
                bbox_xyxy = tracked_dets[:, :4]
                identities = tracked_dets[:, 8]
                categories = tracked_dets[:, 4]
                draw_boxes(frame, bbox_xyxy, identities, categories, names, model_name)
                for i,det in enumerate(tracked_dets):
                    x1, y1, x2, y2 = det[:4]      # 坐标
                    class_id = int(det[4])         # 类别索引
                    conf = det[5]                  # 置信度
                    track_id = int(det[8])         # 追踪ID
                    class_name = names[class_id]   # 类别名称
                    # conf = torch.sigmoid(conf) 
                    conf = torch.sigmoid(torch.tensor(conf)).item()
                    
                    detections_info.append({
                        'bbox': [x1, y1, x2, y2],
                        'class_name': class_name,
                        'class_id': class_id,
                        'confidence': conf,
                        'track_id': track_id
                    })
            
            return {
                'model_name': model_name,
                'detections': len(tracked_dets),
                'detections_info': detections_info,
                'error': None
            }
            
        except Exception as e:
            return {
                'model_name': model_name,
                'detections': 0,
                'error': str(e)
            }
    
    def detect_without_tracking(self, model_name, frame, conf_thres=0.5, iou_thres=0.45):
        """不使用追踪功能的检测（用于fall和fight）"""
        try:
            model = self.models[model_name]
            
            # 获取类别名称
            names = model.module.names if hasattr(model, 'module') else model.names
            
            # 使用letterbox预处理图像，保持宽高比
            img, ratio, pad = letterbox(frame, 640, stride=32, auto=True)
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.device.type != 'cpu' else img.float()
            img /= 255.0
            img = img.unsqueeze(0)
            
            # 推理
            with torch.no_grad():
                pred = model(img, augment=False)[0]
            
            # 应用NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
            detections_info = []
            # 处理检测结果
            for i, det in enumerate(pred):
                if len(det):
                    # 缩放边界框到原始图像尺寸，传入正确的ratio和pad信息
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape, ratio_pad=(ratio, pad)).round()
                    
                    # 处理每个检测结果
                    for *xyxy, conf, cls in reversed(det):
                        if int(cls) < len(names):
                            class_name = names[int(cls)]
                            class_id = int(cls)  # 添加这行
                            # conf = torch.sigmoid(conf) 
                            conf = torch.sigmoid(torch.tensor(conf)).item()
                            # 绘制边界框
                            x1, y1, x2, y2 = [int(x) for x in xyxy]

                            detections_info.append({
                                'bbox': [x1, y1, x2, y2],
                                'class_name': class_name,
                                'class_id': class_id,  # 添加这行
                                'confidence': conf
                            })

                            # 为不同模型设置不同颜色
                            model_colors = {
                                'fall': (0, 0, 255),     # 红色
                                'fight': (255, 255, 0)   # 青色
                            }
                            color = model_colors.get(model_name, (128, 128, 128))
            
                            # 绘制边界框
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # 创建标签
                            label = f"{model_name}:{class_name} {conf:.2f}"
                            
                            # 绘制标签背景
                            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                            
                            # 绘制标签文字
                            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.6, [255, 255, 255], 1)
        
            return {
                'model_name': model_name,
                'detections': len([d for d in pred[0] if len(d) > 0]) if len(pred) > 0 else 0,
                'detections_info':detections_info,
                'error': None
            }
            
        except Exception as e:
            return {
                'model_name': model_name,
                'detections': 0,
                'detections_info': [],  # 确保返回空列表
                'error': str(e)
            }
    
    def analyze_litter_ownership(self, frame):
        """分析垃圾归属，找到离垃圾最近的鞋（人）"""
        try:
            # 获取smoke模型检测到的鞋（人）
            litter_detections = self.model_detect.get('litter', [])
            person_boxes = []
            for detection in litter_detections:
                if detection['class_name'] == 'shoe':  # 鞋代表人
                    person_boxes.append({
                        'bbox': detection['bbox'],
                        'track_id': detection['track_id']
                    })
            
            # 获取litter模型检测到的垃圾
            # litter_detections = self.model_detect.get('litter', [])
            litter_boxes = []
            for detection in litter_detections:
                if detection['class_name'] != 'shoe':  # 排除鞋，只保留垃圾
                    litter_boxes.append({
                        'bbox': detection['bbox'],
                        'track_id': detection['track_id'],
                        'class_name': detection['class_name']
                    })
            
            # 如果没有检测到鞋子，输出None
            if not person_boxes:
                print("镜头中未出现鞋子，鞋子返回: None")
                # 如果有垃圾但没有鞋子，在垃圾框上显示无归属信息
                for litter in litter_boxes:
                    x1, y1, x2, y2 = litter['bbox']
                    cv2.putText(frame, f"Owner: None", 
                               (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                return False
            
            # 如果有垃圾和人员，进行归属判断
            if litter_boxes and person_boxes:
                associations = self.litter_checker.find_littering_person(litter_boxes, person_boxes)
                
                # 在图像上绘制归属关系
                for association in associations:
                    litter_id = association['litter_id']
                    person_id = association['person_id']
                    distance = association['distance']
                    
                    # 找到对应的边界框
                    litter_box = None
                    person_box = None
                    
                    for litter in litter_boxes:
                        if litter['track_id'] == litter_id:
                            litter_box = litter['bbox']
                            break
                    
                    for person in person_boxes:
                        if person['track_id'] == person_id:
                            person_box = person['bbox']
                            break
                    
                    if litter_box and person_box:
                        # 计算中心点
                        litter_center = self.litter_checker.calculate_center(litter_box)
                        person_center = self.litter_checker.calculate_center(person_box)
                        
                        # 绘制连接线
                        cv2.line(frame, 
                                (int(litter_center[0]), int(litter_center[1])),
                                (int(person_center[0]), int(person_center[1])),
                                (0, 255, 255), 2)  # 黄色连接线
                        
                        # 在连接线中点显示距离信息
                        mid_x = int((litter_center[0] + person_center[0]) / 2)
                        mid_y = int((litter_center[1] + person_center[1]) / 2)
                        cv2.putText(frame, f"ID{person_id}->ID{litter_id}: {distance:.1f}px", 
                                   (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        # 在垃圾框上显示归属信息
                        x1, y1, x2, y2 = litter_box
                        cv2.putText(frame, f"Owner: ID{person_id}", 
                                   (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        # 找到对应的垃圾信息
                        litter_info = None
                        for litter in litter_boxes:
                            if litter['track_id'] == litter_id:
                                litter_info = litter
                                break
                        
                        if litter_info:
                            print(f"垃圾归属判断: 垃圾ID{litter_id}({litter_info['class_name']}) 属于 人员ID{person_id}, 距离: {distance:.1f}像素")
                return True
            
        except Exception as e:
            print(f"垃圾归属判断出错: {str(e)}")

    def get_results(self):
        return 
    
    def draw_stats(self, frame, model_stats, total_detections):
        """在帧上绘制统计信息"""
        # 计算FPS
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # 每30帧更新一次FPS
            current_time = time.time()
            self.avg_fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
        
        # 绘制FPS
        cv2.putText(frame, f"FPS: {self.avg_fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 绘制总检测数
        cv2.putText(frame, f"Total Detections: {total_detections}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 绘制各模型检测数
        y_offset = 90
        for model_name, count in model_stats.items():
            color = (0, 255, 0) if count > 0 else (128, 128, 128)
            cv2.putText(frame, f"{model_name}: {count}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
    
    def run_detection(self, camera=0, conf_thres=0.5, iou_thres=0.45):
        """运行摄像头实时检测"""
        if camera == 0:
            print("启动本地摄像头实时检测...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print(f"错误: 无法打开本地摄像头")
                return
            get_frame = lambda: cap.read()
            release = lambda: cap.release()
            window_name = 'Multi-Model Detection'
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
        elif camera == 1:
            print("启动树莓派(Flask)摄像头实时检测...")
            
            # 多次尝试连接
            max_retries = 3
            for attempt in range(max_retries):
                print(f"连接尝试 {attempt + 1}/{max_retries}")
                if self.flask_camera.connect():
                    break
                else:
                    print(f"连接失败，等待 3 秒后重试...")
                    time.sleep(3)
            else:
                print("无法连接到Flask摄像头流，切换到本地摄像头")
                return self.run_detection(camera=0, conf_thres=conf_thres, iou_thres=iou_thres)
            
            get_frame = lambda: self.flask_camera.read()
            release = lambda: self.flask_camera.release()
            window_name = 'Multi-Model Detection (Flask Camera)'
            
        else:
            print("错误: 未知摄像头类型")
            return

        print("按 'q' 键退出")
        print("按 's' 键保存当前帧")
        print("按 'r' 键重置状态")

        # 状态变量
        frame_interval = 6  # 检测间隔（帧）
        frame_count = 0
        decision_interval = 4
        decision_count = 0
        history_decision = [None] * decision_interval
        
        # 添加方向检测的时间间隔控制
        direction_interval = 0.5  # 2秒间隔
        last_direction_time = time.time()
        
        # 错误处理变量
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        try:
            while True:
                ret, frame = get_frame()
                if not ret:
                    consecutive_errors += 1
                    print(f"错误: 无法读取摄像头帧 ({consecutive_errors}/{max_consecutive_errors})")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print("连续错误次数过多，退出检测")
                        break
                        
                    # 如果是 Flask 摄像头，尝试重新连接
                    if camera == 1:
                        print("尝试重新连接Flask摄像头...")
                        self.flask_camera.release()
                        time.sleep(2)
                        if self.flask_camera.connect():
                            print("重新连接成功")
                            consecutive_errors = 0
                            continue
                        else:
                            print("重新连接失败")
                            
                    time.sleep(0.1)
                    continue
                
                # 重置错误计数
                consecutive_errors = 0
                
                frame_count += 1
                if frame_count % frame_interval != 0:
                    continue

                # 验证帧的有效性
                if frame is None or frame.size == 0:
                    print("警告: 接收到空帧")
                    continue
                    
                # 调整帧大小（如果需要）
                height, width = frame.shape[:2]
                if height > 1080 or width > 1920:
                    scale_factor = min(1920/width, 1080/height)
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    frame = cv2.resize(frame, (new_width, new_height))

                # 根据状态选择运行的模型
                if self.current_state == "normal":
                    tracking_models = ['smoke', 'litter']
                    tracking_models = ['litter']
                    normal_models = ['fall', 'fight']
                elif self.current_state == "litter":
                    tracking_models = ['litter']
                    normal_models = []

                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {}
                    for model_name in tracking_models:
                        future = executor.submit(self.detect_with_tracking, model_name, frame, conf_thres, iou_thres)
                        futures[future] = model_name

                model_stats = {}
                total_detections = 0
                for future in futures:
                    try:
                        result = future.result(timeout=2)
                        model_name = result['model_name']
                        detections = result['detections']
                        detections_info = result['detections_info']
                        self.model_detect[model_name] = detections_info
                        model_stats[model_name] = detections
                        total_detections += detections
                        if 'error' in result and result['error']:
                            print(f"模型 {model_name} 处理失败: {result['error']}")
                    except Exception as e:
                        model_name = futures[future]
                        print(f"模型 {model_name} 处理失败: {str(e)}")
                        model_stats[model_name] = 0

                self.draw_stats(frame, model_stats, total_detections)
                cv2.imshow(window_name, frame)
                
                # 决策逻辑
                output_label, output_conf, output_bbox, output_id = self.decide_output(frame)
                if output_label is None:
                    print("None detected!")
                elif self.current_state == 'litter':
                    # 方向检测的时间间隔控制
                    current_time = time.time()
                    if current_time - last_direction_time >= direction_interval:
                        target_id = output_id
                        direction = self.decide_direction(self.trackers['litter'], target_id)
                        if direction:
                            print(f"当前方向: {direction}")
                            # 更新最后方向检测时间
                            last_direction_time = current_time
                        else:
                            print("未检测到方向，但仍然更新时间")
                            last_direction_time = current_time
                    else:
                        # 显示距离下次方向检测的时间
                        remaining_time = direction_interval - (current_time - last_direction_time)
                        print(f"方向检测冷却中，还需等待 {remaining_time:.1f} 秒")
                else:
                    last_output = history_decision[(decision_count+decision_interval-1)%decision_interval]
                    if last_output is None:
                        history_decision[decision_count] = output_label
                    else:
                        if last_output == output_label:
                            if decision_count == decision_interval - 1:
                                self.results = {
                                    'frame': frame,
                                    'label': output_label,
                                    'bbox': output_bbox,
                                    'ID': output_id
                                }
                                print("*************************************************************")
                                print(f"Detection of this frame: {output_label}; Confidence: {output_conf}; BBOX: {output_bbox}; ID: {output_id}")
                                print(f"decision_count: {decision_count};")
                                print("*************************************************************")
                                
                                # 如果检测到 litter，切换到 litter 状态
                                if output_label == "litter":
                                    self.notify_state_change("litter")  # 通知状态变化
                                    target_id = output_id
                                    # 重置方向检测时间，立即进行第一次方向检测
                                    last_direction_time = time.time() - direction_interval
                                    
                            history_decision[decision_count] = output_label
                        else:
                            history_decision = [None] * decision_interval
                            decision_count = 0
                    decision_count = (decision_count + 1) % decision_interval

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户退出")
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    save_path = f"{'flask_' if camera==1 else ''}camera_detection_{timestamp}.jpg"
                    cv2.imwrite(save_path, frame)
                    print(f"已保存当前帧到: {save_path}")
                elif key == ord('r'):
                    self.notify_state_change("normal")
                    print("手动重置到正常状态")
                    # 重置方向检测时间
                    last_direction_time = time.time()

        except KeyboardInterrupt:
            print("用户中断检测")
        except Exception as e:
            print(f"检测过程中出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            release()
            cv2.destroyAllWindows()
            print("摄像头检测结束")

    def notify_state_change(self, new_state):
        """通知状态变化"""
        if self.current_state != new_state:
            self.current_state = new_state
            if self.state_callback:
                self.state_callback(new_state)
            print(f"状态变化: {new_state}")
    
    def notify_direction_change(self, direction, target_id=None):
        """通知方向变化"""
        if self.direction_callback:
            direction_info = {
                'direction': direction,
                'target_id': target_id,
                'timestamp': time.time(),
                'state': self.current_state
            }
            self.direction_callback(direction_info)
            print(f"方向指令: {direction} (目标ID: {target_id})")
    
    def get_current_state(self):
        """获取当前状态"""
        return self.current_state

    def decide_direction(self, tracker, target_id):
        """
        跟踪特定 ID 的物体，并返回前进方向
        :param tracker: SORT 跟踪器对象
        :param target_id: 要跟踪的目标 ID
        """
        # 获取所有正在跟踪的目标
        tracks = tracker.getTrackers()
        
        # 遍历所有目标，找到指定 ID 的目标
        for track in tracks:
            # print(f"trackids: {track.id}")
            if track.id + 1 == target_id:  # 修正匹配逻辑
                # 获取目标的边界框
                bbox = track.get_state()[:, :4]  # [x1, y1, x2, y2]
                bbox = bbox.flatten().tolist()  # 转换为一维列表
                bbox = [int(coord) for coord in bbox]  # 确保所有坐标为整数
                print(f"目标 ID {target_id} 的边界框: {bbox}")
                
                # 将边界框传入 AutoRunner
                AutoRunner = DirectionTracker(target_id, "litter", self.results['frame'])
                direction = AutoRunner.updateAction(bbox)
                print(f"目标 ID {target_id} 的方向: {direction}")
                
                # 如果有方向变化，通知 MQTT
                # if direction and direction != "no_action":
                if direction: 
                    self.notify_direction_change(direction, target_id)
                
                return direction
        
        print(f"未找到目标 ID {target_id}")
        return None


    def decide_output(self,frame):
        # print(self.model_detect)
        model_conf = {}
        model_item_id = {}
        model_item_bbox = {}
        for model_name, _ in model_dict.items():
            detection = self.model_detect.get(model_name, [])
            if model_name == 'smoke':
                for item_info in detection: # 检测到烟就表示smoke
                    if item_info['class_name'] == 'cigarette':
                        if model_name not in model_conf or model_conf[model_name] < item_info['confidence']:
                            model_item_id[model_name] = item_info['track_id']
                            model_item_bbox[model_name] = item_info['bbox']
                        model_conf[model_name] = max(item_info['confidence'], model_conf.get(model_name, 0))
            elif model_name == 'litter':
                for item_info in detection:
                    # if self.analyze_litter_ownership(frame) and item_info['class_name'] != 'shoe':
                    if item_info['class_name'] != 'shoe':
                        if model_name not in model_conf or model_conf[model_name] < item_info['confidence']:
                            model_item_id[model_name] = item_info['track_id']
                            model_item_bbox[model_name] = item_info['bbox']
                        model_conf[model_name] = max(item_info['confidence'], model_conf.get(model_name, 0))
            elif model_name == 'fall':
                # 检测到有人摔倒
                for item_info in detection:
                    model_item_id[model_name] = -1
                    if model_name not in model_conf or model_conf[model_name] < item_info['confidence']:
                        model_item_bbox[model_name] = item_info['bbox']
                    model_conf[model_name] = max(item_info['confidence'], model_conf.get(model_name, 0))
            elif model_name == 'fight':
                # 检测到有人打架
                for item_info in detection:
                    model_item_id[model_name] = -1
                    if model_name not in model_conf or model_conf[model_name] < item_info['confidence']:
                        model_item_bbox[model_name] = item_info['bbox']
                    model_conf[model_name] = max(item_info['confidence'], model_conf.get(model_name, 0))
        
        if not model_conf:
            print("未检测到任何模型的结果")
            return None, None, None, None  # 返回四个 None 值

        max_key = max(model_conf, key=model_conf.get)  # 找到最大值对应的键
        max_value = model_conf[max_key]  # 获取最大值
        print(f"最大的键是: {max_key}, 最大的值是: {max_value}")
        return max_key, max_value, model_item_bbox[max_key], model_item_id[max_key]


class LitterChecker:
    def __init__(self, proximity_threshold=100):
        """
        proximity_threshold: 距离阈值（像素），用于判断是否关联
        """
        self.proximity_threshold = proximity_threshold

    def calculate_center(self, bbox):
        """
        计算框的中心点
        bbox: [x1, y1, x2, y2]
        返回: (center_x, center_y)
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y

    def calculate_distance(self, point1, point2):
        """
        计算两点之间的欧几里得距离
        point1, point2: (x, y)
        返回: 距离
        """
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def find_littering_person(self, litter_boxes, person_boxes):
        """
        找到每个垃圾框最近的人员框
        litter_boxes: [{'bbox': [x1, y1, x2, y2], 'track_id': int, 'class_name': str}, ...]
        person_boxes: [{'bbox': [x1, y1, x2, y2], 'track_id': int}, ...]
        返回: [{'litter_id': int, 'person_id': int, 'distance': float}, ...]
        """
        associations = []
        
        # 如果没有人员，返回空列表
        if not person_boxes:
            return associations

        for litter in litter_boxes:
            litter_center = self.calculate_center(litter['bbox'])
            closest_person = None
            min_distance = float('inf')

            for person in person_boxes:
                person_center = self.calculate_center(person['bbox'])
                distance = self.calculate_distance(litter_center, person_center)

                if distance < min_distance and distance < self.proximity_threshold:
                    min_distance = distance
                    closest_person = person

            if closest_person:
                associations.append({
                    'litter_id': litter['track_id'],
                    'person_id': closest_person['track_id'],
                    'distance': min_distance
                })

        return associations
    


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID (默认: 0), Pi摄像头ID: 1')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU阈值')
    parser.add_argument('--device', default='', help='设备选择 (cpu, 0, 1, 2, 3)')
    
    opt = parser.parse_args()
    
    # 创建多模型摄像头检测器
    detector = MultiModelCameraDetector(device=opt.device)
    
    detector.run_detection(
        camera=opt.camera,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres
    )


if __name__ == '__main__':
    main()