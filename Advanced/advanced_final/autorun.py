from abc import abstractmethod
import cv2
import numpy as np
import time
import os

EPSILON = 100

class TrackedObject:
    def __init__(self, object_id, class_name):
        self.object_id = object_id
        self.class_name = class_name
        self.last_frame = None

    @abstractmethod
    def updateFrame(self, frame):
        """
        Update the video frame of the tracked object.
        """
        pass

    @abstractmethod
    def updateAction(self, bbox_pos: list[float]) -> str:
        """
        Send the action of the tracked object.
        """
        pass

    @abstractmethod
    def _calculate_data_deviation(self) -> int:
        """
        Calculate the core data of the tracked object
        to determine the action.
        """
        pass

class DirectionTracker(TrackedObject):
    def __init__(self, object_id, class_name, frame):
        super().__init__(object_id, class_name)
        self.frame = frame

    def updateFrame(self, frame):
        pass

    def _calculate_data_deviation(self, mid_point_x: float) -> float:
        height, width, _ = self.frame.shape
        width_mid = width // 2
        return mid_point_x - width_mid
    
    def updateAction(self, bbox_pos: list[float]) -> str:
        global EPSILON, DIRECTION_LIST
        try:
            if len(bbox_pos) != 4:
                raise ValueError("Bounding box position must be a list of 4 integers.")
            leftup_x, leftup_y, rightdown_x, rightdown_y = bbox_pos
            mid_point_x = (leftup_x + rightdown_x) // 2
            deviation = self._calculate_data_deviation(mid_point_x)
            if deviation > EPSILON:
                return "right"
            elif deviation < -EPSILON:
                return "left"
            else:
                return "no_action"
        except ValueError as e:
            print(f"Error in DirectionTracker, updateAction: {e}")
            return None
        