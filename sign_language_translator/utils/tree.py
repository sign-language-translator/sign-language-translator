"""utility to print out directory hierarchy

Bugs:
    multiple extra_line printed when parameter is True in nested calls
"""

import os
import re
from glob import glob


def tree(
    cur_path: str = ".",
    directory_only=True,
    extra_line=True,
    ignore=["__pycache__", "temp", "__init__.py"],
    regex=True,
) -> None:
    """prints out directory hierarchy

    Args:
        cur_path (str, optional): the root node of tree or the starting parent directory. Defaults to ".".
        directory_only (bool, optional): True means files will not be listed, only folders. Defaults to True.
    """

    # TODO: ANSI colors: branches, directories, files

    def make_tree(
        cur_path: str,
        prev_indent: str,
        directory_only: bool,
        extra_line=True,
        ignore=None,
        regex=True,
    ) -> None:
        """actually makes the directory hierarchy

        Args:
            cur_path (str): the root node of tree or the starting parent directory.
            prev_indent (str): Whatever string was printed behind the provided directory's name
            directory_only (bool): True means files will not be listed, only folders.
        """
        # TODO: glob patterns

        if ignore is None:
            ignore = ["__pycache__", "temp"]

        # list and sort subdirectories & contained files
        children_paths = glob(os.path.join(cur_path, "*"))
        # TODO: use os.listdir to show hidden files
        child_files = []
        child_dirs = []
        for child in children_paths:
            filename = os.path.split(child)[-1]
            if (regex and any(re.match(regex, filename) for regex in ignore)) or (
                (not regex) and (filename in ignore)
            ):
                continue

            if os.path.isdir(child):
                child_dirs.append(filename)
            else:
                if not directory_only:
                    child_files.append(filename)

        total_children = (
            len(child_dirs) if directory_only else len(child_dirs) + len(child_files)
        )

        for i, child in enumerate(sorted(child_files) + sorted(child_dirs)):
            if i < total_children - 1:
                child_indent, grandchild_indent = "├── ", "│   "
            else:
                child_indent, grandchild_indent = "└── ", "    "

            print(prev_indent + child_indent + child)
            make_tree(
                os.path.join(cur_path, child),
                prev_indent + grandchild_indent,
                directory_only,
                extra_line=extra_line,
                ignore=ignore,
            )

        if extra_line and (total_children > 0):
            line = prev_indent.rstrip()
            if line != "":
                print(line)

    print(os.path.split(os.path.abspath(cur_path))[-1])
    make_tree(
        os.path.abspath(cur_path),
        "",
        directory_only,
        extra_line=extra_line,
        ignore=ignore,
        regex=regex,
    )
