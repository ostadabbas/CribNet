# CribNet: Enhancing Infant Safety in Cribs through Vision-based Hazard Detection

Codes and experiments for the following paper: 

Shaotong Zhu, Amal Mathew,Elaheh Hatamimajoumerd , Sarah Ostadabbas, “CribNet: Enhancing Infant Safety in Cribs through Vision-based Hazard Detection,” in IEEE International Conference on Automatic Face and Gesture Recognition (FG), Jan, 2024.


Contact: 

[Amal Mathew](mathew.ama@northeastern.edu)

[Shaotong Zhu](shawnzhu@ece.neu.edu)]

[Elaheh Hatamimajoumerd](e.hatamimajoumerd@neu.edu)]

[Sarah Ostadabbas](ostadabbas@ece.neu.edu)

## Table of Contents
  * [Introduction](#introduction)
  * [Main Results](#main-results)
  * [Environment](#environment)
  * [Data preparation](#data-preparation)
  * [Testing Blanket Occlusion Pipeline](#testing-blanket-occlusion-pipeline)
  * [Training Detectron2 with CribHD Dataset](#training-detectron2-with-cribhd-dataset)
  * [Toy Filter Pipeline with CribHD Dataset](#toy-filter-pipeline-with-cribhd-dataset)
  * [Citation](#citation)
  * [Acknowledgments](#acknowledgments)

## Introduction
This is the official implementation & Dataset Introduction of our study on enhancing infant sleep safety through computer vision, presented in the context of the concerning uptick in U.S. infant mortality rates linked to unsafe sleep environments. Our research introduces CribNet, a novel framework leveraging state-of-the-art computer vision algorithms for the detection and segmentation of toys and blankets in crib settings—a critical step forward in mitigating risks associated with infant sleep. The cornerstone of CribNet is the Crib Hazard Detection (CribHD) dataset, which is meticulously curated to include diverse subsets: CribHD-T for toys, CribHD-B for blankets, and CribHD-C for simulated hazard scenes within crib environments. This dataset, designed to fill the gap in data availability for crib safety research, will be made publicly accessible. By rigorously evaluating various advanced detection methods using CribNet, we validate the robustness of the CribHD dataset and underscore its potential for broad application.    </br>

## Main Results
### Object Detection Results on CribHD Dataset
#### Performance comparison of object detection models trained on the CribHD dataset in mean average precision (mAP) under IoU thresholds of 50%, 70%, and 90%.
| Model       | mAP_50 (Toy Detection) | mAP_70 (Toy Detection) | mAP_90 (Toy Detection) | mAP_50 (Blanket Detection) | mAP_70 (Blanket Detection) | mAP_90 (Blanket Detection) |
|-------------|------------------------|------------------------|------------------------|----------------------------|----------------------------|----------------------------|
| YOLOV8      | 86.1                   | 85.5                   | 83.4                   | 88.3                       | 84.7                       | 78.6                       |
| Detectron2  | 78.2                   | 77.7                   | 75.0                   | 65.4                       | 58.3                       | 56.5                       |
| YOLACT      | 81.9                   | 78.7                   | 68.5                   | 31.5                       | 27.6                       | 26.8                       |
| **Average** | 82.1                   | 80.6                   | 75.6                   | 61.7                       | 56.9                       | 54.0                       |

### Segmentation Results on CribHD Dataset
#### Performance comparison of segmentation models trained on the CribHD dataset in mean average precision (mAP) under IoU thresholds of 50%, 70%, and 90%.
| Model       | mAP_50 (Toy Segmentation) | mAP_70 (Toy Segmentation) | mAP_90 (Toy Segmentation) | mAP_50 (Blanket Segmentation) | mAP_70 (Blanket Segmentation) | mAP_90 (Blanket Segmentation) |
|-------------|---------------------------|---------------------------|---------------------------|-------------------------------|-------------------------------|-------------------------------|
| YOLOV8      | 88.3                      | 64.7                      | 68.6                      | 68.4                          | 64.3                          | 61.2                          |
| Detectron2  | 80.4                      | 77.5                      | 74.7                      | 70.4                          | 66.2                          | 63.1                          |
| YOLACT      | 77.3                      | 71.4                      | 62.5                      | 30.7                          | 29.8                          | 25.7                          |
| **Average** | 82.0                      | 71.2                      | 68.6                      | 56.5                          | 53.4                          | 50.0                          |

 
## Environment
The code is developed using python 3.6 on NVIDIA GPUs are needed. The code is developed and tested using one NVIDIA Tesla K80 card. 

## Data preparation
For CribHD data, please download from [CribHD dataset]( https://coe.northeastern.edu/Research/AClab/CribHD/). Download and extract them under {CribHD}/ Folder, and make them look like this:
```
CribHD/
├── CRIBHD-B/ (Repeated structure for CRIBHD-T)
│   ├── COCO/ - COCO format annotations
|   |   |
│   │   ├── train/ - Training images and annotations
│   │   │   ├── images/ - Contains all training images (e.g., train00001.jpg)
│   │   │   └── _annotations.coco.json - Annotation file for training images
│   │   ├── test/ - Testing images and annotations
│   │   │   ├── images/ - Contains all testing images (e.g., test00001.jpg)
│   │   │   └── _annotations.coco.json - Annotation file for testing images
│   │   └── valid/ - Validation images and annotations
│   │       ├── images/ - Contains all validation images (e.g., valid00001.jpg)
│   │       └── _annotations.coco.json - Annotation file for validation images
│   │
│   ├── PascalVOC/ - PascalVOC format annotations
│   │   ├── train/, test/, valid/ - Same structure as COCO, with XML annotations
│   │   │   ├── images/ - Contains images for the respective dataset split
│   │   │   └── annotations/ - Contains XML annotation files for images
│   │
│   └── YOLOv8/ - YOLOv8 format annotations
│       ├── train/, test/, valid/ - Directories for dataset splits
│       │   ├── images/ - Contains images (e.g., train00001.jpg)
│       │   └── labels/ - Contains corresponding TXT label files (e.g., train00001.txt)
│
├── CRIBHD-T/
│   ├── (Similar structure as CRIBHD-B)
└── CRIBHD-C/
    ├── images/
        ├── cribc00001.jpg
        ├── cribc00002.jpg
        ├── ...
```

### Example of an Annotation Entry

- **id**: Unique identifier for the annotation.
- **image_id**: Identifier for the image this annotation is associated with.
- **category_id**: Identifier for the category of the object being annotated.
- **bbox**: Bounding box for the object in the format `[x, y, width, height]`, where `(x, y)` represents the top-left corner of the box.
- **area**: Area of the bounding box.
- **segmentation**: A set of points defining the precise outline of the object, useful for segmentation tasks. For example, `[[x1, y1, x2, y2, ..., xn, yn]]` represents a polygon around the object.
- **iscrowd**: Indicates if the annotation represents a single object (`0`) or a group of objects (`1`).


## Testing Blanket Occlusion Pipeline 

CribNet utilizes state-of-the-art computer vision techniques to identify and assess potential dangers presented by objects such as toys and blankets in crib environments. This pipeline aims to mitigate risks associated with infant sleep by providing a system capable of detecting occlusions and enhancing infant safety monitoring.

### Prerequisites

Before you can run the CribNet script, you need to ensure your environment is set up correctly. The following are required:

- Python 3.9 or later
- OpenCV
- PyTorch and TorchVision (appropriate versions for your CUDA version if using GPU acceleration)
- Detectron2

You can install Detectron2 by running the following command:

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

```

```bash
python code/occlusion-detection.py <image_path> <json_data_path> <output_image_path> <output_txt_path>
```
When executing the script, you will need to provide paths for the following:

- `image_path`: Path to the input image file.
  - Example: `/path/to/your/input_image.jpg`
- `json_data_path`: Path to the JSON file containing keypoints data.
  - Example: `/path/to/your/keypoints.json`
- `output_image_path`: Path where the output image will be saved.
  - Example: `/path/to/your/output_image.jpg`
- `output_txt_path`: Path where the output text file will be saved, indicating which keypoints are occluded.
  - Example: `/path/to/your/occlusions.txt`

Ensure to replace the example paths with the actual paths to your files.

#### Generating Keypoints for the Infant

To generate keypoints for the infant, you can utilize the Infant Pose Estimation repository available at [FIDIP](https://github.com/ostadabbas/Infant-Pose-Estimation). Follow the steps provided in the repository to set up the environment and generate keypoints data for your images.

#### Steps:

1. Visit the [Infant Pose Estimation GitHub repository](https://github.com/ostadabbas/Infant-Pose-Estimation).
2. Follow the installation instructions detailed in the repository to set up the necessary environment.
3. Utilize the provided scripts and models to generate keypoints data for your infant images.
4. Save the keypoints data in a JSON format compatible with our `occlusion-detection.py` script.

This process will provide you with the necessary keypoints data for your images, allowing you to visualize keypoints and detect occlusions using our CribNet script.

## Training Detectron2 with CribHD Dataset

Train a Detectron2 model on the CribHD dataset using the provided IPython Notebook in the `Code/Detectron2_Segmentation.ipynb` directory:

1. **Navigate to the Notebook**: Open the `Code/` folder and access the training notebook.

2. **Run the Notebook**: Follow the instructions within the notebook to setup the environment, prepare the CribHD dataset, and configure training parameters as specified in the YAML file.

3. **Training Process**: Execute the notebook cells to start training, monitoring the progress through the output logs.

4. **Model Evaluation**: After training, evaluate the model's performance as guided in the notebook.

This process will result in a Detectron2 model trained on the CribHD dataset.

## Training YOLACT with CribHD Dataset

For training a model using YOLACT with the CribHD dataset, follow the steps outlined in the [YOLACT GitHub repository](https://github.com/dbolya/yolact):

1. **Prepare the Dataset**: Use the COCO data format from the CRIBHD dataset for training the YOLACT
2. **Configuration**: Adjust the training configuration files to align with the CribHD dataset specifics.
3. **Run Training**: Execute the training commands as specified in the YOLACT documentation, adjusting paths and parameters as necessary for the CribHD dataset.
4. **Evaluate the Model**: Follow the YOLACT evaluation instructions to measure your model's performance on the test split of the CribHD dataset.

## Training YOLOv8 with CribHD Dataset

For training YOLOv8 on the CribHD dataset, use the guidance provided in the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics):

1. **Dataset Setup**:Use the YOLOV8 data format from the CRIBHD dataset for training the YOLACT
2. **Training Configuration**: Modify the YOLOv8 training configurations to suit the CribHD dataset, including paths, model architecture, and hyperparameters.
3. **Initiate Training**: Start the training process by running the appropriate YOLOv8 training script, pointing it to your dataset and configuration files.
4. **Model Evaluation**: Utilize the provided tools and instructions for evaluating your trained YOLOv8 model on the CribHD dataset to assess its performance.

## Toy Filter Pipeline with CribHD Dataset

Implement a Toy Filter Pipeline on the CribHD dataset using the provided IPython Notebook located at `Code/TOY-FILTER-Pipeline.ipynb`:

1. **Navigate to the Notebook**: 
    Access the `Code/` directory and open the `TOY-FILTER-Pipeline.ipynb` notebook.

2. **Setup Environment**: 
    The notebook contains detailed instructions for setting up the required environment, including installing necessary packages. Follow these steps to prepare your environment.

3. **Dataset Preparation**: 
    Instructions within the notebook guide you through preparing the CribHD dataset for the pipeline. This involves organizing the dataset and possibly adjusting paths to match your local setup.

4. **Toy Detection and Keypoint Extraction**: 
    Execute the code blocks that implement toy detection and extract keypoints from the faces of infants in the dataset. This step utilizes pre-trained models and custom logic as detailed in the notebook.

5. **Filtering Toys Based on Criteria**: 
    Follow the notebook's guidance to filter the detected toys based on their distance from the infant's mouth and the size of the toys. This step is crucial for assessing potential risks posed by the toys.

6. **Model Evaluation and Filtering Results**: 
    After filtering, evaluate the effectiveness of the pipeline. The notebook provides instructions for assessing the results, including visualizations and metrics.

By completing this process, you will have implemented a Toy Filter Pipeline that detects toys, extracts relevant keypoints, and filters toys based on predefined criteria to ensure infant safety.


## Citation

If you use our code or models in your research, please cite with:

```
Zhu S, Mathew A, Hatamimajoumerd E, Wan M, Ostadabbas S. CribNet: Enhancing Infant Safety in Cribs through Vision-based Hazard Detection. IEEE International Conference on Automatic Face and Gesture Recognition (FG), 2024
```

## Acknowledgements

This project makes use of several open-source software and frameworks. We extend our gratitude to the developers and contributors of these projects:

- **FIDIP**: For the "Invariant Representation Learning for Infant Pose Estimation with Small Data". This work has been instrumental in advancing infant pose estimation research. [GitHub Repository](https://github.com/ostadabbas/Infant-Pose-Estimation)

- **YOLACT**: A real-time instance segmentation tool that has significantly contributed to the efficiency and accuracy of our object detection tasks. [GitHub Repository](https://github.com/dbolya/yolact)

- **YOLOv8**: For providing a state-of-the-art approach to object detection that is both fast and accurate, facilitating rapid development and testing. [Ultralytics GitHub Repository](https://github.com/ultralytics/ultralytics)

- **Detectron2**: Developed by Facebook AI Research, this framework has been crucial in our segmentation and object detection efforts, thanks to its flexibility and powerful modeling capabilities. [GitHub Repository](https://github.com/facebookresearch/detectron2)

- **Roboflow**: Dwyer, B., Nelson, J., Hansen, T., et. al. (2024). For Roboflow (Version 1.0) [Software], which has significantly streamlined our computer vision workflows. Available from [Roboflow](https://roboflow.com).

Each of these tools has played a vital role in the success of our project, and we are immensely thankful for the community's efforts in developing and maintaining them.


## License 
* This code is for non-commertial purpose only. 
* For further inquiry please contact: Augmented Cognition Lab at Northeastern University: http://www.northeastern.edu/ostadabbas/ 




