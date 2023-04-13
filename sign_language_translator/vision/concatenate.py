"""concatenate multiple clips (video & pose) with transitions

Raises:
    NotImplementedError: numpy.ndarray or torch.Tensor
    ValueError: various
"""

from typing import List, Tuple, Union

import moviepy.editor as mpy
import numpy as np
import torch
from scipy.signal import find_peaks

from ..utils.landmarks_info import LandmarksInfo
from . import transforms as T


def _find_edge_peaks_heights(
    values: Union[List[float], np.ndarray, torch.Tensor],
    tolarance_val: float = 0.02,
    height_frac: float = 0.7,
    max_n_steps: int = 25,
) -> Tuple[float, float]:
    """finds the heights of the peaks near the edges of the data

    Args:
        values (Union[List[float], np.ndarray, torch.Tensor]): the timeseries data or y_values of a function e.g. vertical position of a wrist over time
        tolarance_val (float, optional): how much rise from initial height should be ignored. this plus the minimum value in data defines a threshold. Peaks below the threshold are discarded. Defaults to 0.02.
        height_frac (float, optional): the fraction of the edge peak height relative to the lowest level to return. Defaults to 0.7.
        max_n_steps (int, optional): maximum number of steps from each side of the data to consider for a peak. Defaults to 25.

    Raises:
        NotImplementedError: provide numpy.ndarray or torch.Tensor values

    Returns:
        Tuple[int,int]: pair of heights of edge peaks
    """

    values = np.array(values)
    num_values = len(values)

    left_lowest = min(values[:max_n_steps])
    right_lowest = min(values[-max_n_steps:])

    peaks_indexes = find_peaks(values)[0]
    peaks_indexes = np.sort(peaks_indexes)

    left_peaks_indexes = peaks_indexes[peaks_indexes < max_n_steps]
    right_peaks_indexes = peaks_indexes[peaks_indexes >= num_values - max_n_steps]

    valid_left_peaks_indexes = left_peaks_indexes[
        values[left_peaks_indexes] > (left_lowest + tolarance_val)
    ]
    valid_right_peaks_indexes = right_peaks_indexes[
        values[right_peaks_indexes] > (right_lowest + tolarance_val)
    ]

    left_height = (
        values[valid_left_peaks_indexes[0]]
        if len(valid_left_peaks_indexes) > 0
        else max(values[:max_n_steps])
    )
    right_height = (
        values[valid_right_peaks_indexes[-1]]
        if len(valid_left_peaks_indexes) > 0
        else max(values[-max_n_steps:])
    )

    return (
        left_height * height_frac + (1 - height_frac) * left_lowest,
        right_height * height_frac + (1 - height_frac) * right_lowest,
    )


def find_cut_indexes(
    landmarks_list: List[Union[np.ndarray, torch.Tensor]],
    coordinate_type: str = "2D",
    landmark_number_to_track: Union[int, List[int]] = 16,
    landmarks_to_use_for_rectification: List[int] = [11, 12, 23, 24],
    max_cut_steps: int = 25,
    tolarance_val: float = 0.02,
    height_frac: float = 0.7,
    rectify: bool = True,
) -> List[Tuple[int, int]]:
    """find the index numbers near to edges where a landmark clip should be cut. these are the points where a tracked landmark is at a prticular height.

    Args:
        landmarks_list (List[Union[np.ndarray, torch.Tensor]]): list of multi_frame_landmarks which are 2D or 3D sequences of body landmarks.
        coordinate_type (str, optional): the type of coodinates in the landmarks. If the are image coordinates, use "2D". if they are world coordinates, use "3D". Defaults to "2D".
        landmark_number_to_track (Union[int, List[int]], optional): movement of which landmark should be considered to find cut indexes. Defaults to 16 (right wrist).
        landmarks_to_use_for_rectification (List[int], optional): the index of landmarks that should be used stabilization. Defaults to (torso) [11, 12, 23, 24].
        max_cut_steps (int, optional): how many indexes to cut at most from one side. Defaults to 25.
        tolarance_val (float, optional): how much rise from initial height should be ignored. Defaults to 0.02.
        height_frac (float, optional): what fraction of landmark's max or peak height should be used to cut at. Defaults to 0.7.
        rectify (bool, optional): stabilize landmark_list into upright position before tracking the landmark. Defaults to True.

    Raises:
        ValueError: use "2d" or "3d" as coordinate_type
        ValueError: use int or list of int as landmark_number_to_track

    Returns:
        List[Tuple[int, int]]: list of pairs of indexes where multi_frame_landmarks should be cut.
    """

    reshapers = [LandmarksInfo.Reshaper() for _ in range(len(landmarks_list))]
    landmarks_list = [
        reshaper.fold(landmarks)
        for landmarks, reshaper in zip(landmarks_list, reshapers)
    ]

    if str(coordinate_type).lower() == "3d":
        if rectify:
            landmarks_list = T.stabilize_landmarks(
                landmarks_list,
                landmarks_to_use=landmarks_to_use_for_rectification,
                coordinates_to_use=[0, 1, 2],
                target=[
                    [0.16, -0.5, 0],
                    [-0.16, -0.5, 0],
                    [0.10, 0, 0],
                    [-0.10, 0, 0],
                ],
                infer_target=False,
            )

    elif str(coordinate_type).lower() == "2d":
        if rectify:
            landmarks_list = T.stabilize_landmarks(
                landmarks_list,
                landmarks_to_use=landmarks_to_use_for_rectification,
                coordinates_to_use=[0, 1],
                infer_target=True,
            )

    else:
        raise ValueError('use "2d" or "3d" as coordinate_type')

    if isinstance(landmark_number_to_track, int):
        landmark_number_to_track = [landmark_number_to_track] * len(landmarks_list)

    elif isinstance(landmark_number_to_track, list):
        assert len(landmark_number_to_track) == len(
            landmarks_list
        ), "please specify as many landmarks to track as items in landmark_list"
        assert all(
            [isinstance(num, int) for num in landmark_number_to_track]
        ), "please use int landmark_number_to_track"

    else:
        raise ValueError("use int or list of int as landmark_number_to_track")

    peaks_heights = [
        _find_edge_peaks_heights(
            -landmarks[:, landmark_num_to_track, 1],
            tolarance_val=tolarance_val,
            height_frac=height_frac,
            max_n_steps=max_cut_steps,
        )
        for landmarks, landmark_num_to_track in zip(
            landmarks_list, landmark_number_to_track
        )
    ]

    cut_indexes = [[0, len(landmarks) - 1] for landmarks in landmarks_list]
    for i in range(len(landmarks_list) - 1):

        cut_height = min(peaks_heights[i][1], peaks_heights[i + 1][0])

        mask = -landmarks_list[i][:, landmark_number_to_track[i], 1] >= cut_height
        cut_indexes[i][1] = len(landmarks_list[i]) - 1 - np.array(mask)[::-1].argmax()

        mask = (
            -landmarks_list[i + 1][:, landmark_number_to_track[i + 1], 1] >= cut_height
        )
        cut_indexes[i + 1][0] = np.array(mask).argmax()

    return cut_indexes


def insert_transitions(
    landmarks_list: List[Union[np.ndarray, torch.Tensor]], n_transition_frames: int
) -> List[Union[np.ndarray, torch.Tensor]]:
    """calculate transitional frames from edge frames of two landmark_clips using linear interpolation and insert it between them.

    Args:
        landmarks_list (List[np.ndarray | torch.Tensor]): list of multi_frame_landmarks which are 2D or 3D sequences of body landmarks.
        n_transition_frames (int): number of transtional frames to calculate and insert between two landmark_clips

    Raises:
        NotImplementedError: use numpy.ndarray or torch.Tensor landmarks

    Returns:
        List[np.ndarray | torch.Tensor]: same as input list but with transition clips inserted.
    """

    if len(landmarks_list) <= 1 or n_transition_frames == 0:
        return landmarks_list

    if isinstance(landmarks_list[0], np.ndarray):
        concatenate = np.concatenate
    elif isinstance(landmarks_list[0], torch.Tensor):
        concatenate = torch.concatenate
    else:
        raise NotImplementedError("use numpy.ndarray or torch.Tensor")

    # make transitions
    edge_frames = concatenate(
        [
            frame
            for (landmarks_1, landmarks_2) in zip(
                landmarks_list[:-1], landmarks_list[1:]
            )
            for frame in (landmarks_1[-1:], landmarks_2[:1])
        ]
    )
    n_frames = len(edge_frames)
    old_timesteps = np.arange(n_frames)

    # correct timesteps to create transitions only:
    # n_f=4, n_t=1 --> [0, 0.5, 1, 1.5, 2, 2.5, 3] --> [0.5, 2.5]
    new_timesteps = (
        np.linspace(0, n_frames - 1, n_transition_frames * (n_frames - 1) + n_frames)[
            :-1
        ]
        .reshape((-1, n_transition_frames + 1))[0::2, 1:]
        .reshape(-1)
    )

    transition_frames = T._interpolate_multi_frame_landmarks(
        edge_frames,
        old_timesteps,
        new_timesteps,
        time_dimention=0,
    ).reshape((-1, n_transition_frames) + edge_frames[0].shape)

    # insert transitions
    landmarks_list = [
        item
        for landmarks, transition in zip(landmarks_list[:-1], transition_frames)
        for item in (landmarks, transition)
    ] + landmarks_list[-1:]

    return landmarks_list


def concatenate_landmarks(
    landmarks_list: List[Union[np.ndarray, torch.Tensor]],
    n_transition_frames: int = 0,
    trim: bool = False,
    cut_indexes: List[Tuple[int, int]] = None,
    infer_cut_indexes: bool = True,
    landmark_number_to_track: Union[int, List[int]] = 16,
    coordinate_type: str = "3D",
    max_cut_steps: int = 25,
    tolarance_val: float = 0.02,
    height_frac: float = 0.7,
    rectify_for_infering_cut_indexes: bool = True,
    landmarks_to_use_for_rectification: List[int] = [11, 12, 23, 24],
) -> Union[np.ndarray, torch.Tensor]:
    """Concatenate a list of multi_frame_landmarks into one. optionally trim the clips to form continuous sentences without having the hand go back to resting position after each word. Also optionally insert transition frames between clips to avoid "choppy" cuts.

    Args:
        landmarks_list (List[np.ndarray | torch.Tensor]): list of multi_frame_landmarks which are 2D or 3D sequences of body landmarks.
        n_transition_frames (int, optional): number of transtional frames to calculate and insert between two landmark_clips. Defaults to 0.
        trim (bool, optional): whether to cut out parts between the landmark_clips where the hand is just rising from or going down to the resting position. Defaults to False.
        cut_indexes (List[Tuple[int, int]], optional): list pairs of indexes where multi_frame_landmarks should be cut. Defaults to None.
        infer_cut_indexes (bool, optional): whether to use provided cut_index_list (False) or figure them out (True). Defaults to True.
        landmark_number_to_track (Union[int, List[int]], optional): movement of which landmark should be considered to find cut indexes. Defaults to 16 (right wrist).
        coordinate_type (str, optional): the type of coodinates in the landmarks. If the are image coordinates, use "2D". if they are world coordinates, use "3D". Defaults to "3D".
        max_cut_steps (int, optional): how many indexes to cut at most from one side. Defaults to 25.
        tolarance_val (float, optional): how much rise from initial height should be ignored (for triming). Defaults to 0.02.
        height_frac (float, optional): what fraction of landmark's max or peak height should be used to cut at. Defaults to 0.7.
        rectify_for_infering_cut_indexes (bool, optional): stabilize landmark_list into upright position before tracking the landmark. Defaults to True.
        landmarks_to_use_for_rectification (List[int], optional): the index of landmarks that should be used stabilization. Defaults to (torso) [11, 12, 23, 24].

    Raises:
        ValueError: Empty landmark_list. Nothing to concatenate.
        ValueError: must provide cut_indexes or set infer_cut_indexes True
        NotImplementedError: use numpy.ndarray or torch.Tensor

    Returns:
        Union[np.ndarray, torch.Tensor]: a single sequence of landmarks
    """

    if len(landmarks_list) == 0:
        raise ValueError("Empty landmark_list. Nothing to concatenate.")

    if trim:
        if cut_indexes is None:
            if infer_cut_indexes:
                cut_indexes = find_cut_indexes(
                    landmarks_list,
                    coordinate_type=coordinate_type,
                    landmark_number_to_track=landmark_number_to_track,
                    rectify=rectify_for_infering_cut_indexes,
                    landmarks_to_use_for_rectification=landmarks_to_use_for_rectification,
                    max_cut_steps=max_cut_steps,
                    tolarance_val=tolarance_val,
                    height_frac=height_frac,
                )
            else:
                raise ValueError(
                    "must provide cut_indexes or set infer_cut_indexes True"
                )

        landmarks_list = [
            landmarks[start : end + 1]
            for landmarks, (start, end) in zip(landmarks_list, cut_indexes)
        ]

    if n_transition_frames > 0:
        # round(np.random.normal(4,1,(1,)).clip(0))  # or .clip(2,6) or .clip(2)
        landmarks_list = insert_transitions(landmarks_list, n_transition_frames)

    if isinstance(landmarks_list[0], np.ndarray):
        concatenate = np.concatenate
    elif isinstance(landmarks_list[0], torch.Tensor):
        concatenate = torch.concatenate
    else:
        raise NotImplementedError("use numpy.ndarray or torch.Tensor")

    return concatenate(landmarks_list)


def concatenate_clips(
    clips: List[mpy.VideoFileClip],
    n_transition_frames: int = 0,
    trim: bool = False,
    return_trimmed_only:bool=False,
    triming_reference_landmarks: List[Union[np.ndarray, torch.Tensor]] = None,
    cut_indexes: List[Tuple[int, int]] = None,
    coordinate_type: str = "2D",
    landmark_number_to_track: Union[int, List[int]] = 16,
    landmarks_to_use_for_rectification: List[int] = [11, 12, 23, 24],
    max_cut_steps: int = 25,
    tolarance_val: float = 0.02,
    height_frac: float = 0.7,
    rectify_landmarks: bool = False,
) -> mpy.VideoFileClip:
    """Concatenate a list of video clips into one. optionally trim the clips to form continuous sentences without having the hand go back to resting position after each word. optionally also insert transition frames between clips to avoid "choppy" cuts.

    Args:
        clips (List[mpy.VideoFileClip]): list of video clips
        n_transition_frames (int, optional): number of transtional frames to calculate and insert between two landmark_clips. Defaults to 0.
        trim (bool, optional): whether to cut out parts between the clips where the hand is just rising from or going down to the resting position. Defaults to False.
        triming_reference_landmarks (List[np.ndarray | torch.Tensor], optional): the landmarks extracted from the clips. Defaults to None.
        cut_indexes (List[Tuple[int, int]], optional): list pairs of indexes where the provided multi_frame_landmarks should be cut. Defaults to None.
        coordinate_type (str, optional): the type of coodinates in the landmarks. If they are image coordinates, use "2D". if they are world coordinates, use "3D". Defaults to "2D".
        landmark_number_to_track (Union[int, List[int]], optional): movement of which landmark should be considered to find cut indexes. Defaults to 16 (right wrist).
        landmarks_to_use_for_rectification (List[int], optional): the index of landmarks that should be used stabilization. Defaults to (torso) [11, 12, 23, 24].
        max_cut_steps (int, optional): how many frames to cut at most from one side. Defaults to 25.
        tolarance_val (float, optional): how much rise from initial height should be ignored (for triming). Defaults to 0.02.
        height_frac (float, optional): what fraction of landmark's max or peak height should be used to cut at. Defaults to 0.7.
        rectify_landmarks (bool, optional): stabilize landmark_list into upright position before tracking the landmark. Defaults to False.

    Raises:
        ValueError: provide triming_reference_landmarks or cut_indexes

    Returns:
        mpy.VideoFileClip: a single video clip
    """

    if trim:
        if cut_indexes is None:
            if triming_reference_landmarks is None:
                raise ValueError("provide triming_reference_landmarks or cut_indexes")

            cut_indexes = find_cut_indexes(
                triming_reference_landmarks,
                coordinate_type=coordinate_type,
                landmark_number_to_track=landmark_number_to_track,
                landmarks_to_use_for_rectification=landmarks_to_use_for_rectification,
                max_cut_steps=max_cut_steps,
                tolarance_val=tolarance_val,
                height_frac=height_frac,
                rectify=rectify_landmarks,
            )

        cut_durations = [
            [
                clip.duration * cut_index[0] / (len(landmarks) - 1),
                clip.duration * cut_index[1] / (len(landmarks) - 1),
            ]
            for clip, cut_index, landmarks in zip(
                clips, cut_indexes, triming_reference_landmarks
            )
        ]

        clips = [
            clip.subclip(cut_duration[0], cut_duration[1])
            for clip, cut_duration in zip(clips, cut_durations)
        ]

    if return_trimmed_only:
        return clips

    if n_transition_frames > 0:
        final = mpy.concatenate_videoclips(
            clips[:1]
            + [clip.crossfadein(n_transition_frames / clip.fps) for clip in clips[1:]],
            padding=-n_transition_frames / clips[0].fps,
            method="compose",
        )
    else:
        final = mpy.concatenate_videoclips(clips, method="compose")

    return final
