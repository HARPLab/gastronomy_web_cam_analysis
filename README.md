## Overview
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

### Data Pulling
This section documents how/where restaurant footage is pulled.

#### Relevant Files
```
├── pull.py
```



### Clip Extraction
This section documents how dining scenes are extracted from restaurant footage.

#### Relevant Files
```
├── parse_clips.py
├── mask_rcnn
│   ├── frozen_inference_graph.pb
│   └── mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
├── tensorflow_human_detection.py
├── OPwrapper.py
```


### Clip Analysis
This section documents what features are extracted from a given dining scene and where they're stored.

#### Relevant Files
```
├── mask_rcnn
│   ├── frozen_inference_graph.pb
│   └── mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
├── parse_clips.py
├── mask_rcnn
│   ├── frozen_inference_graph.pb
│   └── mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
├── tensorflow_human_detection.py
├── OPwrapper.py
```