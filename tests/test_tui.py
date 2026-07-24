from yakiflow.tui import format_start_time, subtitle_text_widths


def test_format_start_time_uses_minutes_and_centiseconds() -> None:
    assert format_start_time(0) == "00:00.00"
    assert format_start_time(65.678) == "01:05.68"
    assert format_start_time(3_665) == "61:05.00"


def test_subtitle_columns_fit_the_table_viewport_with_padding() -> None:
    source, translation = subtitle_text_widths(76, start_width=8, cell_padding=1)

    rendered_width = (5 + 2) + (8 + 2) + (source + 2) + (translation + 2)
    assert rendered_width == 76
    assert (source, translation) == (27, 28)
