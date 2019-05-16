# Overview
```
.
├── OPwrapper.py
├── README.md
├── SQL_DB
│   ├── ClassDeclarations.py
│   ├── DBWrapper.py
│   └── __init__.py
├── faster_rcnn_inception_v2_coco_2018_01_28
│   ├── checkpoint
│   ├── coco_classes.txt
│   ├── config.pbtxt
│   ├── frozen_inference_graph.pb
│   ├── model.ckpt.data-00000-of-00001
│   ├── model.ckpt.index
│   ├── model.ckpt.meta
│   ├── pipeline.config
│   └── saved_model
│       ├── saved_model.pb
│       └── variables
├── mask_rcnn
│   ├── frozen_inference_graph.pb
│   └── mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
├── parse_clips.py
├── pull.py
├── temp.jpg
├── tensorflow_human_detection.py
```

## Data Pulling
This section documents how/where restaurant footage is pulled.

### Relevant Files
```
├── pull.py
```

### Further Description
Pulling process is performed (and commented) in `pull.py` above - pulled from riptydz stream and stitched together every two hours using ffmpeg. Placed into `/mnt/harpdata/gastronomy_clips/` (our mounted theorem network attached storage).



## Clip Extraction
This section documents how dining scenes are extracted from restaurant footage.

### Relevant Files
```
├── parse_clips.py
├── OPwrapper.py
```

### Further Description
The `parse_dirs` function iterates through the stitched together clips, checking (withing hardcoded regions of interest) every two minutes for the presence of human poses over a certain confidence threshold (using openpose). It extracts and saves interesting clips that exceed a cutoff length (to decrease the number of false positives), also stored within our theorem NAS, and saves the metadata into the sqlite db, for feature extraction next.


## Clip Analysis
This section documents what features are extracted from a given dining scene and where they're stored.

### Relevant Files
```
├── mask_rcnn
│   ├── frozen_inference_graph.pb
│   └── mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
├── parse_clips.py
├── tensorflow_human_detection.py
├── OPwrapper.py
```

### Further Description
Here, for every extracted dining scene focused on a region-of-interest (one table within the larger captured scene), we extract features with corresponding metadata for each frame, all saved into our sqlite db. Models we're using to extract features for a given frame of a dining scene are (1) OpenPose and (2) Mask RCNN. OpenPose gives the pose information and mask rcnn is trained on the Common Objects in Context, or COCO dataset - which can identify cutlery, bowls, plates, cups, and various types of furniture. 