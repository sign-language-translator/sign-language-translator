import os

import numpy as np

from sign_language_translator import Settings
from sign_language_translator.models.utils import VideoEmbeddingPipeline
from sign_language_translator.models.video_embedding import VideoEmbeddingModel
from sign_language_translator.utils import download_resource


class DummyVideoEmbeddingModel(VideoEmbeddingModel):
    def embed(self, frame_sequence, **kwargs):
        # Dummy embedding function for testing
        return np.random.rand(len(list(frame_sequence)), 100)


def get_dummy_model():
    return DummyVideoEmbeddingModel()


def test_process_videos_parallel():
    download_resource("videos/wordless_wordless.mp4")
    source_video_path = os.path.join(
        Settings.RESOURCES_ROOT_DIRECTORY, "videos", "wordless_wordless.mp4"
    )
    video_paths = [source_video_path] * 10
    temp_dir = "temp"

    pipeline = VideoEmbeddingPipeline(get_dummy_model())
    pipeline.process_videos_parallel(
        video_paths, n_processes=2, output_dir=temp_dir, overwrite=True
    )

    embedding_paths = [
        os.path.join(temp_dir, f"{os.path.basename(video_path)}.csv")
        for video_path in video_paths
    ]
    for embedding_path in embedding_paths:
        assert os.path.exists(embedding_path)
