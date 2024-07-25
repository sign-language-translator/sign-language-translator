"""
Video Embedding Models
----------------------

This module provides a collection of deep learning models pretrained on video based tasks.
These models are designed to capture essential features and characteristics from videos,
which can be used for various applications such as gesture recognition, action analysis,
and sign language translation.

Available Models:
-----------------

- **`VideoEmbeddingModel`**: An abstract base class representing a video embedding model.
    This class defines common attributes and methods (such as embed()) for video embedding models.

- **`MediaPipeLandmarksModel`**: A model that utilizes MediaPipe's pose & hand solution to generate video embeddings.
    It detects keypoints representing body joints and estimates their position in 3D world coordinates and in the frame pixels.

Usage:
------

.. code-block:: python

    from sign_language_translator.models import MediaPipeLandmarksModel

    model = MediaPipeLandmarksModel()

    # Define 'frames' as a list of numpy arrays (Width, Height, Channels)
    frames = [...]  # Replace with actual video frames

    # Generate video embeddings using the MediaPipeLandmarksModel
    embeddings = model.embed(frames, landmark_type = "world")
    print(embeddings.shape) # (n_frames, n_landmarks * 5)
"""

from sign_language_translator.models.video_embedding.mediapipe_landmarks_model import (
    MediaPipeLandmarksModel,
)
from sign_language_translator.models.video_embedding.video_embedding_model import (
    VideoEmbeddingModel,
)

__all__ = [
    "VideoEmbeddingModel",
    "MediaPipeLandmarksModel",
]

# TODO: more models

# OpenPose:
#   OpenPose is a popular model for human pose estimation.
#   It detects keypoints representing body joints and estimates their spatial relationships in images or videos.
#   The OpenPose model has been pretrained on a large dataset and can be fine-tuned for specific pose-related tasks,
#   which could be useful for capturing sign language gestures.

# I3D (Inflated 3D ConvNet):
#   I3D is a deep learning model designed for action recognition in videos.
#   It can capture both spatial and temporal information in video frames.
#   Pretrained I3D models are available and can be fine-tuned for recognizing specific sign language gestures.

# 3D Resnet:

# EfficientNet:
#   EfficientNet is an architecture that provides a good trade-off between model size and performance.
#   It has been used for various computer vision tasks, including object detection and classification.
#   You can fine-tune an EfficientNet model on images or video frames containing sign language gestures.

# Motion Vector Generator
#   A motion vector generator estimates the motion between consecutive frames in a video sequence.
#   It works by comparing corresponding blocks or pixels in two frames and calculating the displacement (motion vector) needed to align them.
#   Motion vectors represent the direction and magnitude of motion for different regions of the video frames.
#   Dynamic Gesture Analysis, Feature Tracking, Video Compression, Enhanced Visualization, Gesture Recognition, Quality Assessment.
#   Remember that motion vector analysis can be computationally intensive, especially for large video datasets. Consider performance optimizations and trade-offs to ensure efficient processing
