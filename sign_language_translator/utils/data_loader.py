"""download from S3 or read from disk: video, landmarks [and text] data
"""

import json

import moviepy.editor as mpy
import numpy as np
import pandas as pd

from .sign_data_attributes import SignDataAttributes


class DataLoader:
    # re init
    def __init__(
        self,
        words,
        person_nos=["*"],
        camera_angles=["*"],
        signs_collection="*",
        landmarks_type="all_world_landmarks",
        dataset_directory=None,
    ) -> None:

        self.attributes = SignDataAttributes(
            words,
            person_numbers=person_nos,
            camera_angles=camera_angles,
            signs_collection=signs_collection,
            landmarks_type=landmarks_type,
            dataset_directory=dataset_directory,
        )

    def load_videos(self):
        return [mpy.VideoFileClip(pth) for pth in self.attributes.video_file_paths]

    def load_landmarks(self, reshape=False, drop_visibility=False, concat_cols=None):

        # load data by chuncks
        chunksize = 300

        chunks = pd.read_csv(
            self.attributes.landmarks_file_path,
            usecols=["Word", "Person_Number", "Camera_Angle"]
            + self.attributes.landmarks_type,
            iterator=True,
            chunksize=chunksize,
        )

        df = pd.concat(
            [
                chunk[
                    (
                        chunk["Word"].isin(self.attributes.words)
                        if "*" not in self.attributes.words
                        else np.repeat(True, len(chunk))
                    )
                    & (
                        chunk["Person_Number"].isin(self.attributes.person_nos)
                        if "*" not in self.attributes.person_nos
                        else np.repeat(True, len(chunk))
                    )
                    & (
                        chunk["Camera_Angle"].isin(self.attributes.camera_angles)
                        if "*" not in self.attributes.camera_angles
                        else np.repeat(True, len(chunk))
                    )
                ]
                for chunk in chunks
            ]
        )

        if drop_visibility and not reshape:
            reshape = True
            undo_reshape = True
        else:
            undo_reshape = False

        for land_type in self.attributes.landmarks_type:
            landmark_converter_pipeline = [lambda x: np.array(json.loads(x))]

            if reshape:
                shape = (-1, 33, 4) if land_type.startswith("Pose_") else (-1, 21, 3)
                landmark_converter_pipeline += [lambda x: x.reshape(shape)]

            if drop_visibility:
                landmark_converter_pipeline += [lambda x: x[:, :, :3]]

            if undo_reshape:
                shape = (-1, 33 * 3) if land_type.startswith("Pose_") else (-1, 21 * 3)
                landmark_converter_pipeline += [lambda x: x.reshape(shape)]

            def landmark_converter(x):
                for lc in landmark_converter_pipeline:
                    x = lc(x)
                return x

            df[land_type] = df[land_type].apply(landmark_converter)

        if concat_cols:
            new_cols = []
            removed_cols = []

            for dst_col, src_cols in concat_cols.items():
                df[dst_col] = df.apply(
                    lambda row: np.concatenate([row[col] for col in src_cols], axis=1),
                    axis=1,
                )

                new_cols += [dst_col]
                removed_cols += src_cols

            df = df[
                [col for col in list(df.columns) + new_cols if col not in removed_cols]
            ]

        return df
