# MRL Infrared Eye Images Dataset for Drowsiness Detection (Forked Version)

This dataset is a **forked version** of the original MRL Eye Dataset, containing infrared eye images categorized into **Awake** and **Sleepy** states. It is split into training, validation, and test sets, comprising over 85,000 images captured under various lighting conditions using multiple sensors. This dataset is tailored for tasks such as eye detection, gaze estimation, blink detection, and drowsiness analysis in computer vision.

## Dataset Structure:
- **Train**: Awake (25,770), Sleepy (25,167)
- **Validation**: Awake (8,591), Sleepy (8,389)
- **Test**: Awake (8,591), Sleepy (8,390)

## Directory Tree:
```data/ train/ awake/ sleepy/ val/ awake/ sleepy/ test/ awake/ sleepy/ ```


## Metadata:
- **subject_id**: Unique identifier for each subject (37 subjects)
- **Attributes**: Eye state, gender, glasses, reflections, lighting, and sensor ID

## Original Dataset Overview

The **MRL Eye Dataset** is a large-scale dataset of human eye images designed for computer vision tasks such as eye detection and blink detection. The original dataset includes 84,898 images captured under various conditions.

## Downloads:
- **Forked Dataset**: [Download Forked Dataset](http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip)
- **Original Dataset**: [Download Original Dataset](http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip)
- **Pupil Annotations**: [Download Pupil Annotations](http://mrl.cs.vsb.cz/data/eyedataset/pupil.txt)

## Contact:
For any questions about the dataset, please contact [Radovan Fusek](http://mrl.cs.vsb.cz//people/fusek/).

## Example Images

The dataset includes both open and closed eye images. Below are examples:

![Eye Image Example](http://mrl.cs.vsb.cz/images/eyedataset/eyedataset01.png)

## References

- [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset)
- [Forked Dataset on Kaggle](https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset)

