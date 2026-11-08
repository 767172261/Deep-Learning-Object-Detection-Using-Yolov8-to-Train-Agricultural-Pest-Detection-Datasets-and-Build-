农业害虫检测数据集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c65bfa6e8357444f8e6f3e591f8a8a15.png)

三类，分别包含：蝗虫（locust），毛毛虫（caterpillar），甲虫（beetle）。
4010张JPG图片，xml格式，4010个。三类害虫：locust，caterpillar，beetle。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ec87539a6dae4ffba867121760524cf8.png)

1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9c142a56e97c406196feb0c1ab6e4edd.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5e888e8db8b046f2812aaa9729ba6496.png)
使用YOLOv8进行农业害虫检测，YOLO系列模型在目标检测任务中的高效性和准确性。从数据准备到模型训练、评估及推理的全流程操作。

### 1. 环境准备

首先，确保安装了必要的依赖项。由于YOLOv8是较新的版本，请确保获取最新的安装指南。这里假设您已经设置好了Python环境：

```bash
pip install ultralytics
```

### 2. 数据转换

您的标签文件为VOC格式（XML），需要将其转换为YOLO格式（TXT）。以下是一个简单的Python脚本，用于将VOC XML标签转换为YOLO TXT格式：

```python
import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(xml_file, classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)
    
    yolo_labels = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in classes:
            continue
        
        class_id = classes.index(class_name)
        bbox = obj.find('bndbox')
        
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        x_center = (xmin + xmax) / 2.0 / image_width
        y_center = (ymin + ymax) / 2.0 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height
        
        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return yolo_labels

# 主函数
def main():
    dataset_dir = './path/to/dataset'
    output_dir = './path/to/output'
    classes = ['locust', 'caterpillar', 'beetle']
    
    os.makedirs(output_dir, exist_ok=True)
    
    for xml_file in os.listdir(dataset_dir):
        if not xml_file.endswith('.xml'):
            continue
        
        yolo_labels = convert_voc_to_yolo(os.path.join(dataset_dir, xml_file), classes)
        
        txt_file = xml_file.replace('.xml', '.txt')
        with open(os.path.join(output_dir, txt_file), 'w') as f:
            for label in yolo_labels:
                f.write(f"{label}\n")

if __name__ == '__main__':
    main()
```

### 3. 创建`data.yaml`文件

创建一个`data.yaml`文件来定义数据集路径和类别信息：

```yaml
train: ./path/to/train/images
val: ./path/to/val/images

nc: 3  # 类别数量
names: ['locust', 'caterpillar', 'beetle']
```

请根据实际情况调整路径。

### 4. 划分数据集

如果尚未划分训练集和验证集，可以使用以下脚本进行划分：

```python
import os
import random
import shutil

dataset_dir = './path/to/dataset'
train_dir = './path/to/train'
val_dir = './path/to/val'
split_ratio = 0.8

os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)

image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
random.shuffle(image_files)

split_idx = int(len(image_files) * split_ratio)
for i, image_file in enumerate(image_files):
    src_image = os.path.join(dataset_dir, image_file)
    src_label = os.path.join(dataset_dir, image_file.replace('.jpg', '.txt'))
    
    if i < split_idx:
        dst_dir = train_dir
    else:
        dst_dir = val_dir
    
    shutil.copy(src_image, os.path.join(dst_dir, 'images', image_file))
    shutil.copy(src_label, os.path.join(dst_dir, 'labels', image_file.replace('.jpg', '.txt')))
```

### 5. 模型训练

使用YOLOv8进行训练：

```bash
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.yaml')  # 或者使用 'yolov8s.pt', 'yolov8m.pt' 等

# 开始训练
results = model.train(data='path/to/data.yaml', epochs=100, imgsz=640)
```

### 6. 评估

训练完成后，可以使用以下命令对验证集进行评估并生成预测结果的可视化：

```bash
# 评估模型性能
results = model.val()
```

### 7. 推理

对于新的图片或视频流，你可以使用类似的命令进行推理：

```python
# 对新图片进行推理
results = model.predict(source='path/to/new/image.jpg', save_txt=True, save_conf=True)
```

这步骤，可以有效地使用YOLOv8处理农业害虫检测任务。数据处理和模型训练，如何评估模型性能以及对新数据进行推理的应用实例。
