import os

from sign_language_translator.utils.archive import Archive


def test_archive_utility():
    # make demo files
    os.makedirs(os.path.join("temp", "extracted"), exist_ok=True)
    for i in range(10):
        with open(os.path.join("temp", f"temp_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(f"hello world#{i}")

    # create archive
    Archive.create("temp/temp_?.txt", "temp/temp_archive.zip", overwrite=True)
    assert os.path.exists("temp/temp_archive.zip")

    # list archive contents
    assert len(Archive.list("temp/temp_archive.zip")) == 10
    assert len(Archive.list("temp/temp_archive.zip", pattern="temp_*.txt")) == 10
    assert len(Archive.list("temp/temp_archive.zip", regex=r"temp_[0-4].txt")) == 5

    # extract
    Archive.extract(
        "temp/temp_archive.zip",
        output_dir=os.path.join("temp", "extracted"),
        regex=r"temp_[3-6].txt",
        overwrite=True,
    )
    for i in range(3, 7):
        assert os.path.exists(os.path.join("temp", "extracted", f"temp_{i}.txt"))
