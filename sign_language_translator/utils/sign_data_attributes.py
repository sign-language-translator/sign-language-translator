"""file validates and stores properties of data in a single object

Raises:
    ValueError: unhandled type of words.
    ValueError: unhandled type of person_numbers.
    ValueError: unhandled type of camera_angle.
    ValueError: unknown type of signs_collection.
    ValueError: Invalid type of landmarks_type.
    ValueError: directory to Signs dataset does not exit.
"""

from glob import glob
from os.path import abspath, dirname, exists, join
from typing import List, Union


class SignDataAttributes:
    """validates and stores properties of data in a single object"""

    def __init__(
        self,
        words: Union[str, List[str]],
        person_numbers: Union[str, int, List[int]] = "*",
        camera_angles: Union[str, List[str]] = "*",
        signs_collections: Union[str, List[str]] = "*",
        landmarks_type: Union[str, List[str]] = "all_world_landmarks",
        dataset_directory: str = None,
    ):
        """__init__ function. initializes an object containing all attributes of data. '*' means load all available.

        Args:
            words (Union[str, List[str]]): The sign labels to be loaded
            person_numbers (Union[str, int, List[int]], optional): The IDs of people who performed the sign. Defaults to "*".
            camera_angles (Union[str, List[str]], optional): The camera angle with which the sign was recorded. Defaults to "*".
            signs_collections (Union[str, List[str]], optional): The provider of reference clip of the sign. Defaults to "*".
            landmarks_type (Union[str, List[str]], optional): The type of pose vector (body, hand, 2D, 3D). provide from ["Pose_World_Landmarks", "Left_Hand_World_Landmarks", "Right_Hand_World_Landmarks", "Pose_Image_Landmarks", "Left_Hand_Image_Landmarks", "Right_Hand_Image_Landmarks"]. Defaults to "all_world_landmarks".
            dataset_directory (str, optional): where data is contained. Defaults to None.
        """

        self.words = SignDataAttributes.validate_words(words)
        self.person_nos = SignDataAttributes.validate_person_numbers(person_numbers)
        self.camera_angles = SignDataAttributes.validate_camera_angles(camera_angles)
        self.signs_collections = SignDataAttributes.validate_signs_collections(
            signs_collections
        )
        self.landmarks_type = SignDataAttributes.validate_landmarks_type(landmarks_type)
        self.dataset_directory = SignDataAttributes.validate_dataset_directory(
            dataset_directory
        )

        self.landmarks_file_path = join(
            self.dataset_directory, "Landmarks", "mini_mediapipe_landmarks_dataset.zip"
        )

        self.video_file_paths = [
            p
            for word in self.words
            for person_no in self.person_nos
            for camera_angle in self.camera_angles
            for signs_collection in self.signs_collections
            for p in glob(
                join(
                    self.dataset_directory,
                    "Videos",
                    signs_collection,
                    f"person{person_no}",
                    f"{word}_person{person_no}_{camera_angle}.mp4",
                )
            )
        ]

    LANDMARKS_TYPES = [
        # DON'T SHUFFLE
        "Pose_World_Landmarks",
        "Left_Hand_World_Landmarks",
        "Right_Hand_World_Landmarks",
        "Pose_Image_Landmarks",
        "Left_Hand_Image_Landmarks",
        "Right_Hand_Image_Landmarks",
    ]

    WORLD_LANDMARKS_TYPES = LANDMARKS_TYPES[:3]
    IMAGE_LANDMARKS_TYPES = LANDMARKS_TYPES[3:]

    def validate_words(words: Union[str, List[str]]) -> List:
        """validate the provided words and convert to a handled format

        Args:
            words (Union[str, List[str]]): The sign labels to be loaded

        Raises:
            ValueError: unhandled type of words argument

        Returns:
            List: validated sign labels to be loaded
        """

        # warning word not in [...], "unknown words, use from [...]"
        if isinstance(words, str):
            return [words]
        elif isinstance(words, list):
            if "*" in words:
                return ["*"]
            assert all([isinstance(w, str) for w in words]), "use str words"
            return words
        else:
            raise ValueError("unhandled type of words")

    def validate_person_numbers(person_numbers: Union[str, int, List[int]]) -> List:
        """validate the provided person_numbers and convert to a handled format

        Args:
            person_numbers (Union[str, int, List[int]]): The IDs of people who performed the sign.

        Raises:
            ValueError: unhandled type of person_numbers argument

        Returns:
            List: Validated IDs of people who performed the sign.
        """

        # warning person_no not in [...], "unknown person_no, use from [...]"

        if isinstance(person_numbers, str):
            person_numbers = person_numbers.lstrip("person")
            person_numbers = (
                int(person_numbers) if person_numbers != "*" else [person_numbers]
            )

        if isinstance(person_numbers, int):
            return [person_numbers]

        elif isinstance(person_numbers, list):
            if "*" in person_numbers:
                return ["*"]

            assert all(
                [isinstance(n, int) for n in person_numbers]
            ), "use int person_numbers or '*'"

            return person_numbers

        else:
            raise ValueError("unhandled type of person_numbers")

    def validate_camera_angles(camera_angles: Union[str, List[str]]) -> List:
        """validate the provided camera_angles and convert to a handled format

        Args:
            camera_angles (Union[str, List[str]]): The camera angles with which the sign was recorded

        Raises:
            ValueError: unhandled type of camera_angles

        Returns:
            List: Validated camera_angles with which the sign was recorded
        """

        # warning camera_angle not in ["front","left","right","below"], "unknown camera angle, use from ['front', 'left', 'right', 'below']"

        if isinstance(camera_angles, str):
            return [camera_angles]

        elif isinstance(camera_angles, list):
            if "*" in camera_angles:
                return ["*"]

            assert all(
                [isinstance(ca, str) for ca in camera_angles]
            ), "use str camera_angles"

            return camera_angles
        else:
            raise ValueError("unhandled type of camera_angles")

    def validate_signs_collections(signs_collections: Union[str, List[str]]) -> List:
        """validate the provided signs_collections and convert to a handled format

        Args:
            signs_collections (Union[str, List[str]]): The provider of reference clip of the sign

        Raises:
            ValueError: unknown type of signs_collection

        Returns:
            List: Validated signs_collections
        """

        if isinstance(signs_collections, str):
            return [signs_collections]

        if isinstance(signs_collections, list):
            if "*" in signs_collections:
                return ["*"]

            assert set(signs_collections) <= {
                "HFAD_Book1",
                "None",
            }, "signs_collection not right, use from ['*', 'HFAD_Book1', 'None']"

            return signs_collections
        else:
            raise ValueError("unknown type of signs_collection")

    def validate_landmarks_type(landmarks_type: Union[str, List[str]]) -> List:
        """validate the provided landmarks_types and convert to a handled format

        Args:
            landmarks_type (Union[str, List[str]]): The type of pose vector (body, hand, 2D, 3D). provide from ["Pose_World_Landmarks", "Left_Hand_World_Landmarks", "Right_Hand_World_Landmarks", "Pose_Image_Landmarks", "Left_Hand_Image_Landmarks", "Right_Hand_Image_Landmarks"].

        Raises:
            ValueError: invalid landmarks_type, use from ["Pose_World_Landmarks", "Left_Hand_World_Landmarks", "Right_Hand_World_Landmarks", "Pose_Image_Landmarks", "Left_Hand_Image_Landmarks", "Right_Hand_Image_Landmarks"]
            ValueError: Invalid type of landmarks_type. Use list or str

        Returns:
            List: Validated landmarks_types
        """

        if isinstance(landmarks_type, str):
            landmarks_type = [landmarks_type]

        if isinstance(landmarks_type, list):
            landmarks_type_set = set()
            for land_type in landmarks_type:

                if land_type in SignDataAttributes.LANDMARKS_TYPES:
                    landmarks_type_set.add(land_type)

                elif land_type.lower() in ["all_landmarks", "*"]:
                    landmarks_type_set = SignDataAttributes.LANDMARKS_TYPES
                    break

                elif land_type.lower() == "all_world_landmarks":
                    landmarks_type_set |= set(SignDataAttributes.WORLD_LANDMARKS_TYPES)

                elif land_type.lower() == "all_image_landmarks":
                    landmarks_type_set |= set(SignDataAttributes.IMAGE_LANDMARKS_TYPES)

                else:
                    raise ValueError(
                        f"Invalid landmark type. Use from {SignDataAttributes.LANDMARKS_TYPES}"
                    )

            return list(landmarks_type_set)
        else:
            raise ValueError(f"Invalid type of landmarks_type. Use list or str")

    def validate_dataset_directory(dataset_directory: str) -> str:
        """validate the provided dataset_directory and convert to a handled format

        Args:
            dataset_directory (str): directory to Signs dataset

        Raises:
            ValueError: directory to Signs dataset does not exit.

        Returns:
            str: validated dataset_directory
        """

        if dataset_directory is None:
            dataset_directory = abspath(
                join(
                    dirname(dirname(__file__)),
                    "datasets",
                    "Signs_recordings",
                )
            )

        if not exists(dataset_directory):
            raise ValueError(
                f"directory:'{dataset_directory}' to Signs dataset does not exit."
            )

        return dataset_directory
