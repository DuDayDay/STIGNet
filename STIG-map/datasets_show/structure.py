import sys

from PySide2 import QtWidgets, QtCore
from PySide2.QtCore import Qt, QTimer, QUrl
from PySide2.QtUiTools import QUiLoader
from datasets_show.point_clouds_density import compute_density,create_plot_with_cluster_selection
from PySide2.QtCore import QStringListModel
import os
import re
import xml.etree.ElementTree as ET
import json
import open3d as o3d
from point_seg.point_seg_filter import extract_object_pointcloud_o3d
import numpy as np
from pointcloud.PointCloudPlayer import PointCloudPlayer_2
from pointcloud.pointshow import PointCloudWidget
from point_seg.point_clouds_process import align_point_cloud_to_maximize_z,align_point_cloud_to_maximize_x_keepZ
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtUiTools import QUiLoader
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from datasets_show.points_track import pixelate_point_cloud_fixed_ratio
import matplotlib.cm as cm
from datasets_show.point_sample import farthest_point_sampling, plot_sphere
from sympy.abc import q
from PySide2.QtWebEngineWidgets import QWebEngineView
from datasets_show.point_activate import compute_multiframe_activity_with_displacement,map_activity_to_plotly_colors
import plotly.graph_objs as go
from collections import defaultdict
def project_point_cloud(point_cloud):
    xy_projection = point_cloud[:, :2]
    xz_projection = point_cloud[:, [0, 2]]
    yz_projection = point_cloud[:, 1:]
    return xy_projection, xz_projection, yz_projection


def create_projection_image(projection_data, colors, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.scatter(projection_data[:, 0], projection_data[:, 1], s=5,color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    # 设置坐标轴的范围和比例
    ax.set_aspect('equal', 'box')  # 强制x和y比例相同
    min_val = min(projection_data.min(), -projection_data.max())
    max_val = max(projection_data.max(), -projection_data.min())
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # 将图像保存到缓冲区
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)

    img = QImage()
    img.loadFromData(buf.read())
    return img
class datasets_structure(QtWidgets. QMainWindow):

    def __init__(self):
        super(datasets_structure, self).__init__()
        loader = QUiLoader()
        self.ui = loader.load('ui/datasets.ui', self)
        self.setWindowTitle("数据集结构")
        self.setCentralWidget(self.ui)

        self.ui.pushButton.clicked.connect(self.load_datasets)
        self.ui.pushButton_2.clicked.connect(self.show_datasets)
        self.datasets_dict = ''

        self.listView = self.ui.listView
        self.model = QStringListModel(self)
        self.listView.setModel(self.model)
        self.listView.clicked.connect(self.load_dl)

        self.listView_point = self.ui.listView_2
        self.model_point = QStringListModel(self)
        self.listView_point.setModel(self.model_point)
        # self.listView_point.clicked.connect(self.image_frame)

        self.listView_label = self.ui.listView_3
        self.model_label = QStringListModel(self)
        self.listView_label.setModel(self.model_label)
        # self.listView_label.clicked.connect(self.image_frame)
        self.name_count_text = None
        self.seg_point_cloud = []
        self.player = None  # 初始化播放器
        self.point_cloud_widget = PointCloudWidget()  # 创建点云显示窗口
        self.point_cloud_widget.setParent(self.ui.openGLWidget)
        # self.point_cloud_widget.setGeometry(self.ui.openGLWidget.rect())  # 设置几何位置
        self.point_cloud_widget.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.player = PointCloudPlayer_2(self.point_cloud_widget, self)
        """投影显示"""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_projections)
        self.timer.start(100)  # 每 500 毫秒更新一次
        self.ui.pushButton_3.clicked.connect(self.density)
        self.x_y = []
        self.x_z = []
        self.y_z = []
        self.first = 0
        self.first_2 = 0

        self.plot_widget = self.ui.widget
        self.web_view = QWebEngineView(self.plot_widget)
        self.plot_widget.layout().addWidget(self.web_view)


    def find_xml(self, xml_path):
        # 解析 XML 内容
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # 找到 <class> 标签并提取 name 属性
        class_element = root.find("class")
        if class_element is not None:
            name_value = class_element.get("name")
            print("name 对应的值:", name_value)
        else:
            print("未找到 <class> 标签")
            name_value = None
        frame_element = root.find("frame")
        if frame_element is not None:
            num = frame_element.get("num")
            print("num 对应的值:", num)
        else:
            print("未找到 <class> 标签")
            num = None
        return name_value, num

    def load_datasets(self):
        options = QtWidgets.QFileDialog.Options()
        directory_name = QtWidgets.QFileDialog.getExistingDirectory(self, "选择文件夹", "", options=options)
        if directory_name:
            print(f"选择的文件夹: {directory_name}")
            self.ui.lineEdit.setText(directory_name)
            self.datasets_dict = directory_name
            self.load_folders_point(directory_name)
        else:
            print("未选择任何文件夹")

    def load_folders_point(self, folder_path):
        if os.path.isdir(folder_path):
            # 获取子文件夹
            subfolders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

            # 初始化用于存储子文件夹名称和 XML 文件内容的列表
            folder_and_xml_content = []

            # 初始化字典用于统计 name 出现的次数
            name_count = defaultdict(int)

            # 遍历每个子文件夹，读取所有 XML 文件内容
            for subfolder in subfolders:
                subfolder_path = os.path.join(folder_path, subfolder)

                # 查找子文件夹中的所有 XML 文件
                xml_files = [f for f in os.listdir(subfolder_path) if f.endswith('.xml')]

                if xml_files:
                    # 每个文件夹的内容显示初始化
                    xml_contents = f"{subfolder}: "

                    # 遍历并读取 XML 文件内容
                    for xml_file in xml_files:
                        xml_file_path = os.path.join(subfolder_path, xml_file)

                        try:
                            # 使用 find_xml 方法获取 XML 中的 name 和数量
                            name, num = self.find_xml(xml_file_path)

                            # 将当前 XML 文件的 name 添加到文件夹显示内容
                            xml_contents += f"\n  {xml_file}: {name}"

                            # 更新 name 计数
                            name_count[name] += 1

                        except ET.ParseError:
                            xml_contents += f"\n  {xml_file}: Invalid XML format"

                else:
                    xml_contents = f"{subfolder}: No XML file found"

                # 将文件夹名称和所有 XML 文件内容添加到列表
                folder_and_xml_content.append(xml_contents)

            # 更新模型，以显示子文件夹名称及其 XML 文件内容
            self.model.setStringList(folder_and_xml_content)

            # 打印 name 出现的次数统计
            print("Name Counts:")
            for name, count in name_count.items():
                print(f"{name}: {count}")
            name_count_text = "数据集整体动作统计:\n" + "\n".join(f"{name}: {count}" for name, count in name_count.items())
            self.ui.textEdit_3.setPlainText(name_count_text)
            self.name_count_text = name_count_text
        else:
            print('fail')
            self.model.setStringList([])  # 如果路径无效，清空列表

    def load_dl(self, index):
        self.first = 0
        self.first_2 = 0
        str_4 = ''
        self.seg_point_cloud = []
        self.point_cloud_widget.setVisible(False)
        self.reset_view()
        data = self.model.data(index)
        folder_name = data.split(": ", 1)[0]  # 提取 `subfolder`
        folder_path = os.path.join(self.datasets_dict, folder_name)
        points = os.path.join(folder_path, 'pcd')
        labels = os.path.join(folder_path, 'label')
        points_files = self.load_files(points)
        labels_files = self.load_files(labels)
        self.model_point.setStringList(points_files)
        self.model_label.setStringList(labels_files)
        files = zip(points_files, labels_files)
        for point, label in files:
            point_file = os.path.join(points, point)
            label_file = os.path.join(labels, label)
            points_seg = self.label_points(point_file, label_file)
            if self.ui.checkBox.isChecked():
                centroid = points_seg.mean(axis=0)
                # 将点云平移，使质心位于原点
                points_seg = points_seg - centroid
                str_4 = "进行中心化处理"
            else:
                str_4 = "没有进行中心化处理"
            if points_seg is not None:
                if self.first == 0:
                    rotation_matrix = align_point_cloud_to_maximize_z(points_seg)
                    self.first = 1
                aligned_point_cloud = np.dot(points_seg, rotation_matrix.T)
                # 将质心平移到原点
                centroid = aligned_point_cloud.mean(axis=0)
                aligned_point_cloud -= centroid
                points_seg = aligned_point_cloud
                if self.first_2 == 0:
                    rotation_matrix_2 = align_point_cloud_to_maximize_x_keepZ(points_seg)
                    self.first_2 = 1
                points_seg = np.dot(points_seg, rotation_matrix_2.T)
                points_seg[:, 2] *= -1
            self.seg_point_cloud.append(points_seg)
        # self.load_and_display_projections(self.seg_point_cloud[1])
        # self.point_sample()
        # self.point_activate()
        lengths = [len(sublist) for sublist in self.seg_point_cloud]
        # 计算平均值
        average_length = 0
        if lengths:  # 检查列表是否非空
            average_length = sum(lengths) / len(lengths)
            print("平均长度:", average_length)
        else:
            print("seg_point_cloud 为空，无法计算平均长度")
        xml = os.path.join(folder_path, 'point_clouds.xml')
        name_value, num = self.find_xml(xml)
        print(name_value, num)
        str_1 = f"该点云视频片段的动作类型为：{name_value}"
        str_2 = f"一共有{str(num)}帧点云图"
        str_3 = f"平均点云数为{average_length}"
        str_all = self.name_count_text + '\n' + '——————————————————' + '\n' + str_1 + '\n' + str_2 + '\n' + str_3 + '\n' + str_4
        self.ui.textEdit_3.setPlainText(str_all)

    def load_files(self, folder_path):
        if os.path.isdir(folder_path):
            # 获取文件夹中的所有图像文件名
            points_files = [f for f in os.listdir(folder_path)]
            # 按照文件名中的数字顺序排序
            points_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))  # 提取文件名中的数字进行排序
            # 更新模型以显示文件名
            print(points_files)
        else:
            points_files = [] # 如果不是有效的文件夹，清空列表

        return points_files

    def load_json(self, json_file_path):

        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def label_points(self, points_file, label_file):
        filtered_point_cloud = []
        json_data = self.load_json(label_file)
        pcd_file_path = points_file
            # 使用open3d加载PCD文件
        point_cloud = o3d.io.read_point_cloud(pcd_file_path)

            # 提取第一个对象的点云
        if json_data["objects"] != []:
            object_info = json_data["objects"][0]
            filtered_point_cloud = extract_object_pointcloud_o3d(point_cloud, object_info)
            filtered_point_cloud = np.asarray(filtered_point_cloud.points)
        return filtered_point_cloud

    def show_datasets(self):
        if self.seg_point_cloud is not None:
            self.point_cloud_widget.setVisible(True)
            self.point_cloud_widget.point_cloud_frame = self.seg_point_cloud
            self.point_cloud_widget.is_playing = True
            # self.player = PointCloudPlayer_2(self.point_cloud_widget, self)
            self.player.start(50)  # 开始播放，每 100 毫秒更新一次

    def calculate_density_colors(self, points, n_neighbors=10):
        """Calculates density-based color for each point using K-Nearest Neighbors."""
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(points)
        distances, _ = nbrs.kneighbors(points)

        # Inverse of average distance to represent density
        density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-5)
        density_normalized = (density - density.min()) / (density.max() - density.min())

        # Map density to color
        colors = cm.viridis(density_normalized)[:, :3]  # Get RGB values from colormap
        return colors

    def update_projections(self):

        if self.point_cloud_widget.frame_index < len(self.seg_point_cloud):
            print(self.point_cloud_widget.frame_index)
            current_frame = self.seg_point_cloud[self.point_cloud_widget.frame_index]
            xy_proj, xz_proj, yz_proj = project_point_cloud(current_frame)
            colors = self.calculate_density_colors(current_frame)
            self.x_z = xz_proj
            self.y_z = yz_proj
            self.x_y = xy_proj
            xy_image = create_projection_image(xy_proj,colors, "Projection on X-Y Plane", "X", "Y")
            xz_image = create_projection_image(xz_proj,colors, "Projection on X-Z Plane", "X", "Z")
            yz_image = create_projection_image(yz_proj,colors, "Projection on Y-Z Plane", "Y", "Z")

            self.ui.label_4.setPixmap(QPixmap.fromImage(xy_image))
            self.ui.label_5.setPixmap(QPixmap.fromImage(xz_image))
            self.ui.label_6.setPixmap(QPixmap.fromImage(yz_image))

            xy_size = self.ui.label_4.size()
            xz_size = self.ui.label_5.size()
            yz_size = self.ui.label_6.size()

            # 将投影图像缩放为 QLabel 的大小
            self.ui.label_4.setPixmap(
                QPixmap.fromImage(xy_image).scaled(xy_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.ui.label_5.setPixmap(
                QPixmap.fromImage(xz_image).scaled(xz_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.ui.label_6.setPixmap(
                QPixmap.fromImage(yz_image).scaled(yz_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def reset_view(self):
        # 清空 QLabel 中的图片
        self.ui.label_4.clear()
        self.ui.label_5.clear()
        self.ui.label_6.clear()

    def density(self):
        current_directory = os.path.abspath(os.getcwd())
        print("当前工作目录的绝对路径:", current_directory)
        if self.seg_point_cloud is not None:
            self.pix()
            all_points, density = compute_density(self.seg_point_cloud)

            # 绘制带颜色的点云
            create_plot_with_cluster_selection(all_points, density)
            plot_file = os.path.join(current_directory, "plot.html")
            self.web_view.setUrl(QUrl.fromLocalFile(plot_file))
            # 确保页面加载完成后再显示
            self.web_view.loadFinished.connect(self.on_load_finished)
        else:
            QtWidgets.QMessageBox.information(self, "fail", f"请先加载数据")

    def on_load_finished(self, ok):
        if ok:
            self.web_view.show()  # 页面加载成功后显示
        else:
            print("加载页面失败")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.web_view.setGeometry(0, 0, self.plot_widget.width(), self.plot_widget.height())
        self.web_view.resize(self.plot_widget.size())
    def pix(self):
        # 设置像素化的网格中每个像素的实际空间尺寸 (例如，1个像素表示 0.1x0.1 的空间)
        pixel_size = (0.01, 0.01)

        # 获取像素化图像
        # pixelated_image = pixelate_point_cloud_fixed_ratio(self.x_z, pixel_size)

        # 可视化像素化图像
        # plt.imshow(pixelated_image, cmap='gray', interpolation='nearest')
        # plt.title("Pixelated Point Cloud (Fixed Ratio)")
        # plt.xlabel("X Axis (pixels)")
        # plt.ylabel("Y Axis (pixels)")
        # plt.show()

    def point_sample(self):
        if self.seg_point_cloud is not None:
            points = self.seg_point_cloud[1]
            num_samples = 16  # 可以根据需要调整采样点数量
            sampled_points, sampled_indices = farthest_point_sampling(points, num_samples)

            fig = plt.figure(figsize=(10, 5))
            # 原始点云，显示采样点和球体
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=1, label='Original Points')
            ax1.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], c='red', s=10,
                        label='Sampled Points')
            ax1.set_title("原始点云（包含采样点）")
            # 绘制每个采样点的球体
            for center in sampled_points:
                plot_sphere(ax1, center, radius=0.1, color='red', alpha=0.3)
            ax1.legend()
            # 仅显示采样点
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], c='red', s=10)
            ax2.set_title("最远点采样后的点云")
            plt.show()

    def point_activate(self):
        point_clouds = self.seg_point_cloud
        all_points = np.vstack(point_clouds)

        # 计算多帧点云的活动频率，考虑位移
        voxel_size = 0.2
        activity_map = compute_multiframe_activity_with_displacement(point_clouds, voxel_size=voxel_size)

        # 准备颜色数据
        color_values = map_activity_to_plotly_colors(activity_map, all_points, voxel_size=voxel_size)

        # 创建 plotly 的散点图用于显示所有帧合并后的点云
        fig = go.Figure(data=[go.Scatter3d(
            x=all_points[:, 0],
            y=all_points[:, 1],
            z=all_points[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=color_values,  # 根据活动频率映射的颜色
                # colorscale='Viridis',  # 可选择其他颜色主题
                colorscale='Inferno',
                colorbar=dict(title='Activity Frequency'),
                opacity=0.8
            )
        )])

        # 更新布局
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            title="Multi-frame Point Cloud Activity Frequency (With Displacement)"
        )

        # 显示图像
        fig.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = datasets_structure()
    main_window.show()
    sys.exit(app.exec_())