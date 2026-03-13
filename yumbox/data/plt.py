import arabic_reshaper
from bidi import algorithm as bidialg

# https://stackoverflow.com/questions/65493638/glyph-23130-missing-from-current-font
# sudo apt-get install fonts-noto-cjk
# fc-list :family
# have to install fonts + rebuild matplotlib cache for it to work?
# plt.rcParams["font.family"] = ["Noto Sans CJK"]  # Show Chinese labels
# plt.rcParams["font.sans-serif"] = ["Noto Sans CJK"]  # Show Chinese labels
# plt.rcParams["axes.unicode_minus"] = False


def fix_farsi(text):
    text = arabic_reshaper.reshape(text)
    return bidialg.get_display(text)


def calc_subplot_height(width, count_of_items):
    height = count_of_items
    if height % width == 0:
        return int(height / width)
    else:
        return int(height / width) + 1


def print_available_fonts():
    import matplotlib.font_manager as fm

    for font in fm.findSystemFonts(fontpaths=None, fontext="ttf"):
        print(fm.FontProperties(fname=font).get_name())
