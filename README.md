## Introduction

ETNet is a bounding box detection model based on a CNN feature extractor, a Transformer Encoder, and a prediction head. Given an image, the attention layers built in Transformer can capture long-range spatial relationships between extreme keypoints generated from COCO dataset. The keypoints then grouped using a center grouping method to generate bounding boxes for objects. 
