 # GWD-Net：砂轮片缺陷检测网络

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0) 

## 项目简介

GWD-Net (Grinding Wheel Net) 是一个专门为检测砂轮片表面缺陷而设计的深度学习模型。该项目基于 **YOLOv8** (由 Ultralytics 提供) 构建，并融合了多项关键改进，以提升在这一具有挑战性的工业应用场景中的性能：

*   **跨尺度特征融合模块 (CCFM - Cross-Scale Feature Fusion Module):** 有效地融合不同尺度的特征，改善对尺寸差异巨大的缺陷（尤其是小尺寸缺陷）的检测能力。
*   **动态检测头 (DyHead - Dynamic Head):** 使用基于注意力的动态检测头替代标准检测头，能够更好地适应缺陷和背景中常见的不规则形状与复杂纹理。
*   **知识蒸馏 (CWDLoss - Channel-wise Distillation Loss):** 采用通道级知识蒸馏策略，从一个性能更强的教师模型迁移知识，使得相对轻量化的 GWD-Net (学生模型) 能够在不显著增加计算开销的情况下获得更高的准确率（相比仅添加 CCFM/DyHead）。

本项目旨在为砂轮片制造过程中的自动化质量控制提供一套准确、实用的解决方案。

## 主要特性

*   **更高精度:** 在目标砂轮片数据集上显著优于基线 YOLOv8n 模型 (例如：mAP@0.5-0.95 提升 8.1%，mAP@0.5 提升 6.6%)。详见“实验结果”部分。
*   **增强小缺陷检测:** CCFM 模块专门解决了标准架构难以检测微小缺陷的问题。
*   **对不规则形状的鲁棒性:** DyHead 提高了对不规则形状、低对比度缺陷的定位精度并减少了误报。
*   **性能优化:** 知识蒸馏有助于在给定模型大小下最大化检测精度。
*   **基于 YOLOv8 构建:** 充分利用了 Ultralytics 高效且维护良好的框架。

## 快速开始

### 环境要求

*   Python 3.8 或更高版本
*   Pip (Python 包管理器)
*   Git (用于克隆代码仓库 - 项目上传后需要)
*   (推荐) Conda 用于环境管理
*   (GPU 加速需要) NVIDIA GPU + CUDA + cuDNN (请确保 CUDA 版本与 PyTorch 2.2.x 兼容)

### 安装步骤

1.  **获取代码:**
    *   **方式 A (项目上传至 GitHub 后):**
        ```bash
        # git clone https://github.com/Souny-hub/GWD-Net
        # cd GWD-Net
        ```
    *   **方式 B (本地副本 / ZIP 压缩包):**
        将项目文件解压到您选择的目录，然后使用命令行进入项目根目录：
        ```bash
        cd /path/to/your/GWD-Net-Project-Folder
        ```
        (请将 `/path/to/your/GWD-Net-Project-Folder` 替换为实际路径)

2.  **创建环境 (推荐使用 Conda):**
    ```bash
    conda create --name gwdnet_env python=3.9 # 建议使用您开发时的 Python 版本
    conda activate gwdnet_env
    ```
    *(或者，使用 `python -m venv venv` 并激活虚拟环境)*

3.  **安装 PyTorch:**
    访问 [PyTorch 官方网站](https://pytorch.org/get-started/locally/)，根据您的操作系统、包管理器 (Conda) 和计算平台 (CPU 或特定 CUDA 版本) 选择合适的安装命令。您的环境使用了 `torch==2.2.2`。请查找支持此版本的安装命令，例如：
    ```bash
    # !!! 仅为示例 - 请务必从 PyTorch 官网获取适合您系统的命令 !!!
    # 例如 CUDA 11.8:
    # conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
    # 例如 CUDA 12.1:
    # conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
    # 例如 仅 CPU:
    # conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 cpuonly -c pytorch
    ```

4.  **安装依赖:**
    建议在项目根目录提供一个 `requirements.txt` 文件 (可从您的 `pip freeze` 输出生成，**建议移除** 通过 conda 单独安装的 torch, torchvision, torchaudio 行)。
    ```bash
    pip install -r requirements.txt
    ```
    *   **关于 `mmcv`:** 如果 `pip install mmcv==2.2.0` 失败，可能需要根据您的 PyTorch/CUDA 版本参考 [MMCV 安装指南](https://mmcv.readthedocs.io/zh_CN/latest/get_started/installation.html) 使用特定命令或源码编译进行安装。

## 使用方法

### 数据准备

1.  按照 YOLOv8 兼容的格式组织您的数据集 (例如 COCO 或 YOLO 格式)。
    *   **图像:** 存放于目录 (如 `dataset/images/train`, `dataset/images/val`)。
    *   **标签:** 对应的标注文件 (如 YOLO 格式的 `.txt` 文件) 存放于平行目录 (如 `dataset/labels/train`, `dataset/labels/val`)。
2.  创建一个数据集配置文件 (例如 `grinding_wheel.yaml`)，告知 YOLOv8 数据位置和类别信息。示例结构：
    ```yaml
    path: ../dataset # 数据集根目录 (相对路径或绝对路径)
    train: images/train # 训练集图片路径 (相对于 path)
    val: images/val   # 验证集图片路径 (相对于 path)
    # test: images/test # 可选：测试集图片路径
    
    # Classes
    names:
      0: 商标缺失 # 请替换为您数据集中实际的缺陷类别名称
      1: 商标偏离
      2: 边缘缺失
      3: 透砂
      4: 铁环瑕疵 # 论文中没有明确列全9类，这里根据论文提及和常见缺陷补充，请核对
      5: 重网
      6: 铁环不齐
      7: 孔洞
      8: 铁环缺失
      # ... 添加所有类别，确保索引与标签文件对应
    ```

### 模型评估

在验证集上评估您训练好的模型的性能。

```
      # 示例评估命令
yolo val \
  model=[path/to/your/trained_gwdnet_weights.pt] # 训练好的 GWD-Net 权重路径
  data=[path/to/your/grinding_wheel.yaml]     # 数据集配置文件
  imgsz=640                                     # 训练时使用的图像尺寸
  split=val                                     # 在验证集上评估
  # ... 其他 YOLOv8 验证参数 (device, batch, conf, iou 等)
    
```

IGNORE_WHEN_COPYING_START

 content_copy  download 

 Use code [with caution](https://support.google.com/legal/answer/13505487). Bash

IGNORE_WHEN_COPYING_END

### 推理/检测

使用训练好的 GWD-Net 模型对新图片或视频进行缺陷检测。

```
      # 示例推理命令
yolo predict \
  model=[path/to/your/trained_gwdnet_weights.pt] # 训练好的 GWD-Net 权重路径
  source=[path/to/image.jpg]                     # 单张图片路径
  # 或 source=[path/to/video.mp4]                 # 视频文件路径
  # 或 source=[path/to/image_directory/]          # 图片目录路径
  # 或 source=0                                   # 使用 0 号摄像头
  imgsz=640                                     # 图像尺寸
  conf=0.25                                     # 置信度阈值
  iou=0.45                                      # NMS IoU 阈值
  save=True                                     # 保存带有预测结果的图片/视频
  # ... 其他 YOLOv8 推理参数 (device, save_txt, save_conf 等)
    
```

### 训练命令

```
import warnings
import ultralytics

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"\\\\\")
    #model.load('yolov8n.pt')
    model.train(data=r'\\\\\\\\',
                # resume=True,
                split='test',
                #cache=False,
                #imgsz=640,
                epochs=100,
                batch=16,
                close_mosaic=0,
                workers=0,
                device=0,
                optimizer='SGD',
                amp=False,
    )
# import torch
# print(torch.version.cuda)  # 检查 PyTorch 使用的 CUDA 版本
# print(torch.cuda.is_available())  # 检查 CUDA 是否可用
#
# print(torch.__version__)
# print(torch.cuda.is_available())
```

IGNORE_WHEN_COPYING_START

 content_copy  download 

 Use code [with caution](https://support.google.com/legal/answer/13505487). Bash

IGNORE_WHEN_COPYING_END

## 实验结果

我们的实验表明，GWD-Net 相较于基线 YOLOv8n 模型，在砂轮片缺陷检测任务上取得了显著提升：

- 
- **mAP@0.5-0.95:** 提升了 **8.1%**
- **mAP@0.5:** 提升了 **6.6%**
- 召回率 (Recall): 提升了 4.4%
- 精确率 (Precision): 提升了 4.3%
- F1 分数: 提升了 5.0%

## 许可证 (License)

**重要提示:** GWD-Net 基于 Ultralytics YOLOv8 开发，该框架的许可证为 。GWD-Net 对原始 YOLOv8 的架构进行了修改 (引入了 CCFM 和 DyHead)。**GNU Affero General Public License v3.0 (AGPL-3.0)**

这意味着：

- 
- **如果您分发 GWD-Net，或将其用于提供网络服务（例如 SaaS），您必须将您修改后的版本（即 GWD-Net）的完整、对应的源代码，在同样的 AGPL-3.0 许可下公开提供。**
- 即使您仅在此类服务中使用了本项目提供的预训练模型，该条款依然适用。
- **对于无法满足 AGPL-3.0 合规要求的商业应用场景，** 您可能需要向 Ultralytics 购买。详情请参阅 **商业许可证**[Ultralytics 许可页面](https://www.google.com/url?sa=E&q=https%3A%2F%2Fultralytics.com%2Flicense)。

本项目亦使用了其他第三方库。关于这些组件及其许可证的详细列表，请参阅  文件。使用本软件即表示您同意遵守 AGPL-3.0 许可证以及所有包含组件的许可证条款。[链接至您的开源合规说明文件 OSS_Compliance.md 或在此列出关键项]

## 联系方式

- 孙正豪

如有疑问或遇到问题，请通过以下方式联系：[zhenghaosun5@gmail.com]

## 致谢

- 
- 本工作大量借鉴了[Ultralytics YOLOv8](https://www.google.com/url?sa=E&q=https%3A%2F%2Fgithub.com%2Fultralytics%2Fultralytics) 框架。
- 感谢 PyTorch、MMCV 以及本项目中使用的其他开源库的开发者。

```
    
```

IGNORE_WHEN_COPYING_START

 content_copy  download 

 Use code [with caution](https://support.google.com/legal/answer/13505487). 

IGNORE_WHEN_COPYING_END
