"""
this data_reprocessing is used to fit for others models like P4DTransformer

"""
from utils.data_preprocess import read_data
import os
import numpy as np
import random
import xml.etree.ElementTree as ET
def parse_xml(xml_file):
    """
    解析 XML 文件并提取所需信息。
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    class_type = None
    frame_num = None

    # 提取 class 的 type 和 frame 的 num
    for elem in root:
        if elem.tag == "class" and "type" in elem.attrib:
            class_type = elem.attrib["type"]
        elif elem.tag == "frame" and "num" in elem.attrib:
            frame_num = elem.attrib["num"]

    return class_type, frame_num
def adjust_frames(points_clouds):
    k = len(points_clouds)
    if k == 16:
        return points_clouds  # 如果已经是16帧，直接返回

    # 生成0到k-1之间的16个均匀分布的索引
    indices = np.linspace(0, k - 1, num=16, endpoint=True)
    # 四舍五入取整，得到具体的帧索引
    indices = np.round(indices).astype(int)

    # 根据索引构造调整后的点云列表
    adjusted_clouds = [points_clouds[i] for i in indices]
    return adjusted_clouds

def adjust_boxes(seg_point_cloud,SAMPLE_NUM,output_dir):
    points_cloud = []
    for point in seg_point_cloud:
        points_num = point.shape[0]
        all_sam = np.arange(points_num)
        if points_num < SAMPLE_NUM:
            if points_num < SAMPLE_NUM / 2:
                needed = SAMPLE_NUM - 2 * points_num
                # 使用 random.choices 允许重复抽样以补充样本
                supplementary = random.choices(list(all_sam), k=needed)
                index = supplementary
                index.extend(list(all_sam) * 2)  # 添加两次原样本
            else:
                needed = SAMPLE_NUM - points_num
                supplementary = random.sample(list(all_sam), needed)
                index = supplementary
                index.extend(list(all_sam))  # 添加一次原样本
        else:
            index = random.sample(list(all_sam), SAMPLE_NUM)
        points = point[index, :]
        points_cloud.append(points)
    points_cloud = adjust_frames(points_cloud)
    points_cloud = np.array(points_cloud, dtype=object)
    np.savez_compressed(os.path.join(output_dir, os.path.basename(filename).split('.')[0] + '.npz'), point_clouds= points_cloud)
    print('Saved-' + filename)


path = 'raw-ahu'
save_path = 'save'
class_path = os.listdir(path)
counter = {}
SAMPLE_NUM = 256


for class_name in class_path:
    path_labels = os.listdir(os.path.join(path, class_name))
    for path_label in path_labels:
        point_path = os.path.join(path, class_name, path_label)
        points = os.path.join(point_path, 'pcd')
        labels = os.path.join(point_path, 'label')
        xml_file = os.path.join(point_path, 'point_clouds.xml')
        class_type, _ = parse_xml(xml_file)
        if class_name != 'data':
            num = counter.get(class_type, 0)
            if int(class_type) <=3:
                filename = 'a' + f"{int(class_type):02d}" + '_e' + f"{num:02d}" + '_sdepth'
            else:
                filename = 'a' + f"{int(class_type)-1:02d}" + '_e' + f"{num:02d}" + '_sdepth'
            counter[class_type] = num + 1
            seg_point_cloud, _ = read_data(points, labels)
            adjust_boxes(seg_point_cloud, SAMPLE_NUM, save_path)
        if class_name == 'data' and int(class_type) <= 3:
            if int(class_type) == 3:
                class_type = '9'
            num = counter.get(class_type, 0)
            filename = 'a' + f"{int(class_type):02d}" + '_e' + f"{num:02d}" + '_sdepth'
            counter[class_type] = num + 1
            seg_point_cloud, _ = read_data(points, labels)
            adjust_boxes(seg_point_cloud, SAMPLE_NUM, save_path)

