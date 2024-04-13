import numpy as np

from sign_language_translator.vision import Video


def test_video_initialization():
    video_from_dataset = Video("videos/wordless_wordless.mp4")
    assert video_from_dataset is not None

    frames = np.random.randint(0, 255, size=(10, 100, 100, 3), dtype=np.uint8)
    video_from_frames = Video(frames, fps=30.0)
    assert video_from_frames is not None
    assert video_from_frames.shape == (10, 100, 100, 3)
    assert video_from_frames.duration == 10 / 30.0


def test_video_trim_concat():
    base = Video("videos/wordless_wordless.mp4")

    joined_video = base + base.trim(start_index=0, end_index=int(len(base) / 2))

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
    base = Video("videos/wordless_wordless.mp4")
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

    assert (base.numpy() == frames).all()
    assert base.torch() is not None


# def test_video_display():
#     pass


def test_video_transformations():
    frames = np.random.randint(0, 255, size=(10, 100, 100, 3), dtype=np.uint8)
    video = Video(frames, fps=30.0)
    assert video[:5, :50, 50:].shape == (5, 50, 50, 3)

    joined_video = video + video.trim(start_index=0, end_index=int(len(video) / 2))
    joined_video.append(video)  # type: ignore
    assert len(joined_video) == 26
    # TODO: fix this
    # assert joined_video[:, :50, 50:].shape == (26, 50, 50, 3)
