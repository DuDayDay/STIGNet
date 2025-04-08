from PySide2 import QtCore
import numpy as np
from PySide2.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from PySide2.QtCore import Qt

class PointCloudWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.point_cloud_frame = []  # 存储点云数据
        self.frame_index = 0  # 当前帧索引
        self.rotation_x = 0
        self.rotation_y = 0
        self.zoom = -10
        self.last_mouse_pos = None
        self.pan_offset_x = 0  # 用于平移的X轴偏移
        self.pan_offset_y = 0  # 用于平移的Y轴偏移
        self.is_panning = False  # 是否正在进行平移
        self.axis_length = 5.0  # 坐标轴长度，初始化

    def initializeGL(self):
        glClearColor(0, 0, 0, 1)
        glEnable(GL_DEPTH_TEST)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # 应用缩放和平移
        glTranslatef(self.pan_offset_x, self.pan_offset_y, self.zoom)
        # 应用旋转
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)

        # 绘制坐标轴和刻度
        self.draw_axes()

        # 绘制点云
        if self.point_cloud_frame:
            current_frame = self.point_cloud_frame[self.frame_index]
            self.center_point_cloud(current_frame)
            glBegin(GL_POINTS)
            for point in current_frame:
                glColor3f(1, 1, 1)
                glVertex3f(point[0], point[1], point[2])
            glEnd()

    def center_point_cloud(self, points, centroid=None):
        if centroid is None:
            if isinstance(points, np.ndarray):
                if points.size == 0:
                    return
                points = points.tolist()

            if points:
                centroid = [sum(coord) / len(points) for coord in zip(*points)]

        if centroid is not None:
            glTranslatef(-centroid[0], -centroid[1], 0)

    def draw_axes(self):
        """绘制自适应视口大小的 XYZ 坐标轴和刻度"""
        tick_interval = 0.5  # 刻度间隔
        num_ticks = int(self.axis_length / tick_interval)

        # 绘制 X 轴
        glColor3f(1, 0, 0)  # 红色
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(self.axis_length, 0, 0)
        glEnd()

        # X 轴刻度
        for i in range(1, num_ticks + 1):
            glBegin(GL_LINES)
            glVertex3f(i * tick_interval, -0.1, 0)
            glVertex3f(i * tick_interval, 0.1, 0)
            glEnd()

        # 绘制 Y 轴
        glColor3f(0, 1, 0)  # 绿色
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, self.axis_length, 0)
        glEnd()

        # Y 轴刻度
        for i in range(1, num_ticks + 1):
            glBegin(GL_LINES)
            glVertex3f(-0.1, i * tick_interval, 0)
            glVertex3f(0.1, i * tick_interval, 0)
            glEnd()

        # 绘制 Z 轴
        glColor3f(0, 0, 1)  # 蓝色
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, self.axis_length)
        glEnd()

        # Z 轴刻度
        for i in range(1, num_ticks + 1):
            glBegin(GL_LINES)
            glVertex3f(-0.1, 0, i * tick_interval)
            glVertex3f(0.1, 0, i * tick_interval)
            glEnd()

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.is_panning = True
        self.last_mouse_pos = (event.x(), event.y())

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is not None:
            dx = event.x() - self.last_mouse_pos[0]
            dy = event.y() - self.last_mouse_pos[1]

            if self.is_panning:  # 平移模式
                self.pan_offset_x += dx * 0.01  # 调整比例控制平移速度
                self.pan_offset_y -= dy * 0.01
            else:  # 旋转模式
                self.rotation_x += dy
                self.rotation_y += dx

            self.update()
        self.last_mouse_pos = (event.x(), event.y())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.is_panning = False
        self.last_mouse_pos = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.zoom += delta / 240.0
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        glViewport(0, 0, self.width(), self.height())
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width() / self.height(), 0.1, 100)
        glMatrixMode(GL_MODELVIEW)

        # 根据窗口尺寸调整坐标轴长度
        self.axis_length = 0.1 * min(self.width(), self.height()) / 100.0  # 取窗口大小的10%作为轴长度
