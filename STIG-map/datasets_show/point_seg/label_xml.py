import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

string_to_number = {
    "抡手": 0,
    "踢脚": 1,
    "下蹲": 2,
    "抬手": 3,
    "走动": 4,
    "低头": 5,
    "转身": 6
}
def xml(value,num):
    # 创建根元素
    root = ET.Element("data")
    # 添加子元素
    item1 = ET.SubElement(root, "class")
    item1.set("name", value)
    item1.set("type", str(string_to_number[value]))

    item2 = ET.SubElement(root, "frame")
    item2.set("num", num)
    item2.set("type", "pcd")

    # 将 ElementTree 转换为字符串
    rough_string = ET.tostring(root, encoding="utf-8")

    # 使用 minidom 进行格式化
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ")
    return pretty_xml