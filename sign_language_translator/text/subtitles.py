"""generate subtitles in WebVTT format
"""

from typing import List, Tuple


def _make_timestamp(total_seconds) -> str:
    hours, remaining_seconds = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remaining_seconds, 60)

    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"


def make_captions(
    texts: List[str],
    durations: List[float],
    highlighted_ranges: List[Tuple[int, int]],
    vertical_position_frac: float = 0.6,
    align: str = "left",
    emphasis_tag="b",
) -> str:

    captions = ["WEBVTT"]

    elapsed_time = 0.0
    for text, duration, (h_start, h_end) in zip(texts, durations, highlighted_ranges):
        captions.append("")
        captions.append(
            f"{_make_timestamp(elapsed_time)} --> {_make_timestamp(elapsed_time + duration)}"
            + f" line:{vertical_position_frac:.0%}"
            + f" align:{align}"
        )
        captions.append(
            text[:h_start]
            + f"<{emphasis_tag}>"
            + text[h_start:h_end]
            + f"</{emphasis_tag}>"
            + text[h_end:]
        )
        elapsed_time += duration

    return "\n".join(captions)


def make_chapters(
    texts: List[str], durations: List[float], highlighted_ranges: List[Tuple[int, int]]
) -> str:
    return make_captions(texts, durations, highlighted_ranges)
