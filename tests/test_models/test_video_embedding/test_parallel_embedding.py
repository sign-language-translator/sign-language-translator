import os

import numpy as np

from sign_language_translator import Settings
from sign_language_translator.models.utils import VideoEmbeddingPipeline
from sign_language_translator.models.video_embedding import VideoEmbeddingModel
from sign_language_translator.utils import download_resource


class DummyVideoEmbeddingModel(VideoEmbeddingModel):
    def embed(self, frame_sequence, **kwargs):
        # Dummy embedding function for testing
        return np.random.rand(len(list(frame_sequence)), 10)


def get_dummy_model():
    return DummyVideoEmbeddingModel()


def test_process_videos_parallel():
    video_paths = []
    for filename in ["wordless_wordless.mp4", "pk-hfad-2_hour.mp4"]:
        download_resource(f"videos/{filename}")
        video_paths.append(os.path.join(
            Settings.RESOURCES_ROOT_DIRECTORY, "videos", filename
        ))

    video_paths = video_paths * 10
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
