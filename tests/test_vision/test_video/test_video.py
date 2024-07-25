import numpy as np
import pytest

from sign_language_translator.vision.video.video import Video
from sign_language_translator.vision.video.video_iterators import (
    SequenceFrames,
    VideoSource,
)


def test_video_initialization():
    video_from_dataset = Video.load_asset("videos/wordless_wordless.mp4")
    assert len(video_from_dataset) > 10

    frames = np.random.randint(0, 255, size=(10, 100, 100, 3), dtype=np.uint8)
    video_from_frames = Video(frames, fps=30.0)
    assert video_from_frames is not None
    assert video_from_frames.shape == (10, 100, 100, 3)
    assert video_from_frames.duration == 10 / 30.0

    # save
    video_from_frames.save("temp/noise.mp4", codec="mp4v", overwrite=True)
    video_from_dataset.save_frames_grid(
        "temp/grid.jpg", rows=2, columns=3, overwrite=True
    )
    video_from_frames.save_frame("temp/noise.jpg", index=5, overwrite=True)


def test_video_trim_concat():
    base = Video.load_asset("wordless_wordless")

    joined_video = base + base.trim(start_index=0, end_index=int(base.n_frames / 2))

    joined_video.append(base)

    v3 = Video.concatenate(
        [
            base,
            joined_video.trim(
                start_index=int(len(joined_video) / 2), end_index=len(joined_video) - 1
            ),
        ]
    )

    assert v3 is not None
    assert len(v3) == int((int(len(base) * 1.5) + 1 + len(base)) / 2) + 1 + len(base)


def test_video_iteration():
    base = Video.load_asset("videos/xx-wordless-1_wordless.mp4")
    joined_video = base + base.trim(start_index=0, end_index=int(len(base) / 2))
    joined_video.append(base)

    for frame in joined_video:
        assert frame is not None

    for frame in base.iter_frames(3, len(base) - 3, 2):
        assert frame is not None

    frames = np.arange(10).astype(np.uint8).reshape(-1, 1, 1, 1)
    base = Video(frames, fps=10)
    joined_video = base[:3:1] + base[5:7] + base[9:]
    joined_video.append(base)  # type: ignore

    new_frames = list(joined_video.iter_frames(1, 15, 2))  # type: ignore
    new_frames = np.array(new_frames).squeeze().tolist()
    assert new_frames == [1, 5, 9, 1, 3, 5, 7]

    assert (np.array(base) == frames).all()
    assert (base.torch().numpy() == frames).all()

    iterable_source_video = Video(
        (base.get_frame(index=i) for i in range(1, 7, 2)), total_frames=3, fps=1
    )
    assert len(iterable_source_video) == 3
    assert iterable_source_video.duration == 3
    assert iterable_source_video.height == 1
    assert iterable_source_video.width == 1
    assert iterable_source_video.n_channels == 1
    assert [f.item() for f in iterable_source_video] == [1, 3, 5]


# def test_video_display():
#     pass


def test_video_transformations():
    frames = np.random.randint(0, 255, size=(10, 100, 100, 3), dtype=np.uint8)
    video = Video(frames, fps=30.0)
    assert video[:5, :50, 50:].shape == (5, 50, 50, 3)

    joined_video = video + video.trim(start_index=0, end_index=int(len(video) / 2))
    joined_video.append(video)  # type: ignore
    assert len(joined_video) == 26

    new_video = Video.concatenate([video, video[:5]])  # type: ignore
    new_video.transform(lambda x: (x + 1).astype(np.uint8))
    assert np.all(new_video[-1] == video[4] + 1)  # type: ignore
    assert len(video.transformations) == 0

    # TODO: fix this
    # assert joined_video[:, :50, 50:].shape == (26, 50, 50, 3)


def test_video_frames_grid():
    frames = np.arange(10).astype(np.uint8).reshape(-1, 1, 1, 1).repeat(2, axis=-1)
    base = Video(frames, fps=10)

    assert base.frames_grid(2, 1).tolist() == [[[0, 0]], [[9, 9]]]
    assert base[1:-2].frames_grid(2, 2).tolist() == [[[1, 1], [3, 3]], [[5, 5], [7, 7]]]  # type: ignore

    assert base.frames_grid(2, 3, width=9).shape[:2] == (6, 9)
    assert base.frames_grid(2, 3, height=10).shape[:2] == (10, 15)


def test_video_stacking():
    videos = [
        Video(np.arange(start, start + 3).astype(np.uint8).reshape(-1, 1, 1, 1), fps=3)
        for start in [0, 10, 20, 30, 40]
    ]

    dim_0_stack = Video.stack([videos[0], videos[1]], dim=0).numpy().squeeze().tolist()
    assert dim_0_stack == [0, 1, 2, 10, 11, 12]

    dim_1_stack = Video.stack([videos[1], videos[2]], dim=1).numpy().tolist()
    assert dim_1_stack == [[[[10]], [[20]]], [[[11]], [[21]]], [[[12]], [[22]]]]

    dim_2_stack = Video.stack([videos[2], videos[3]], dim=2).numpy().tolist()
    assert dim_2_stack == [[[[20], [30]]], [[[21], [31]]], [[[22], [32]]]]

    dim_3 = Video.stack([videos[3], videos[4].trim(0, 0.1)], dim=3).numpy().tolist()
    assert dim_3 == [[[[30, 40]]], [[[31, 0]]], [[[32, 0]]]]


def test_video_source():
    frames = np.arange(10).astype(np.uint8).reshape(-1, 1, 1, 1)
    sequence = SequenceFrames(frames, fps=1)  # type: ignore
    source = VideoSource(sequence)
    assert len(source) == 10
    assert source.height == 1
    assert source.width == 1
    assert source.n_channels == 1
    assert list(source) == list(range(10))

    source.start_index = 3
    source.end_index = 7
    source.step_size = 2
    assert len(source) == 3
    assert [f.item() for f in source] == [3, 5, 7]

    source.step_size = -1
    assert len(source) == 5
    assert [f.item() for f in source] == [7, 6, 5, 4, 3]

    source.close()


def test_video_validation():
    frames = np.random.randint(0, 255, size=(10, 100, 100, 3), dtype=np.uint8)
    video = Video(frames, fps=1)

    # --- INVALID INDEX --- #

    assert video[-10].shape == (100, 100, 3)
    assert video.get_frame(timestamp=-9.5).shape == (100, 100, 3)

    with pytest.raises(ValueError) as exc_info:
        _ = video.get_frame(timestamp=-4.5, index=5)
    assert "either" in str(exc_info.value).lower()

    with pytest.raises(ValueError):
        _ = video[10]

    with pytest.raises(ValueError):
        _ = video[10, 10:12, 20:-24, :, 4]

    assert video[-10].shape == (100, 100, 3)
    with pytest.raises(ValueError):
        _ = video[-11]

    with pytest.raises(TypeError):
        _ = video[4:, "[12,34]"]  # type: ignore

    assert video[4:, ..., 1].shape == (6, 100, 100, 1)  # type: ignore

    with pytest.raises(ValueError):
        _ = video.get_frame(timestamp=100)

    with pytest.raises(ValueError):
        _ = video.trim(start_index=-100)

    with pytest.raises(ValueError):
        _ = video.trim(start_time=-3, end_index=-7)

    # --- TRANSFORM --- #

    with pytest.raises(ValueError) as exc_info:
        Video.concatenate([])
    assert "empty" in str(exc_info.value).lower()

    with pytest.raises(ValueError):
        video.transform(5)  # type: ignore

    # --- FILE ERRORS --- #

    with pytest.raises(FileNotFoundError):
        _ = Video.load_asset("videos/xx-.^-1_.^.mp4")

    video.save("temp/noise.mp4", codec="mp4v", overwrite=True)
    with pytest.raises(FileExistsError):
        video.save("temp/noise.mp4", codec="mp4v", overwrite=False)

    video.save_frames_grid("temp/grid.jpg", rows=2, columns=3, overwrite=True)
    with pytest.raises(FileExistsError):
        video.save_frames_grid("temp/grid.jpg", rows=2, columns=3, overwrite=False)

    video.save_frame("temp/noise.jpg", index=5, overwrite=True)
    with pytest.raises(FileExistsError):
        video.save_frame("temp/noise.jpg", index=5, overwrite=False)

    # --- INITIALIZATION --- #
