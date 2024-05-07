from typing import Dict, NamedTuple


class Color(NamedTuple):
    r: int
    g: int
    b: int


class ColorSet(NamedTuple):
    background: Color
    foreground: Color


light_foreground: Color = Color(248, 246, 242)
dark_foreground: Color = Color(117, 110, 102)

default_colorset = ColorSet(Color(128, 128, 128), light_foreground)
win_background_color: Color = Color(185, 173, 161)

color_palette: Dict[int, ColorSet] = {
    0: ColorSet(Color(202, 193, 181), dark_foreground),
    2: ColorSet(Color(236, 228, 219), dark_foreground),
    4: ColorSet(Color(236, 225, 204), dark_foreground),
    8: ColorSet(Color(233, 181, 130), light_foreground),
    16: ColorSet(Color(233, 154, 109), light_foreground),
    32: ColorSet(Color(231, 131, 103), light_foreground),
    64: ColorSet(Color(229, 105, 72), light_foreground),
    128: ColorSet(Color(232, 209, 128), light_foreground),
    256: ColorSet(Color(232, 205, 114), light_foreground),
    512: ColorSet(Color(231, 202, 101), light_foreground),
}
