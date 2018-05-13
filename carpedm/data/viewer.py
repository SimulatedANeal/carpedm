#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

"""This module provides methods and classes for viewing images.

Attributes:
    FONT_FILE (str): Filename of ``.ttc`` file for displaying Japanese
        character fonts.

References:
    Annotation code based on Stack Overflow answer `here`_.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, TextBox, CheckButtons
from mpl_toolkits.axes_grid1 import Divider, LocatableAxes, Size
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

from carpedm.data.ops import in_region
from carpedm.data.lang import code2char


FONT_FILE = "HiraginoMaruGothic.ttc"
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), FONT_FILE)


def font(size):
    """Fonts helper for matplotlib."""
    from matplotlib.font_manager import FontProperties

    return FontProperties(fname=FONT_PATH, size=size)


class Viewer(object):

    def __init__(self, images, shape):
        fig = plt.figure(figsize=(10, 6))

        # subplot positions
        h_nav = [Size.Fixed(0.5), Size.Fixed(2.5)]
        v_nav = [Size.Fixed(0.5), Size.Scaled(1.0), Size.Fixed(0.5)]
        h_im = [Size.Fixed(0.5), Size.Scaled(1.0), Size.Fixed(0.5)]
        v_im = [Size.Fixed(0.5), Size.Scaled(1.0), Size.Fixed(0.5)]
        nav = Divider(fig, (0.0, 0.0, 0.2, 1.), h_nav, v_nav, aspect=False)
        image = Divider(fig, (0.2, 0.0, 0.8, 1.), h_im, v_im, aspect=True)
        image.set_anchor('C')

        # Toolbar menu box
        ax1 = LocatableAxes(fig, nav.get_position())
        ax1.set_axes_locator(nav.new_locator(nx=1, ny=1))
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        fig.add_axes(ax1, label='toolbar')

        ax1.text(0.05, 0.45, "Filter", weight='heavy',
                 transform=ax1.transAxes)

        # Image space
        ax2 = LocatableAxes(fig, image.get_position())
        ax2.set_axes_locator(image.new_locator(nx=1, ny=1))
        fig.add_axes(ax2, label='image_space')

        self.callback = ImageIndex(images, shape, fig)

        # Navigation
        ## Go to
        ax_text_index = plt.axes([0.59, 0.05, 0.1, 0.075])
        ip = InsetPosition(ax1, [0.2, 0.84, 0.3, 0.05])
        ax_text_index.set_axes_locator(ip)
        entry_index = TextBox(ax_text_index, 'Go to', initial="0")
        entry_index.on_submit(self.callback.submit_index)
        ## Previous
        ax_prev = plt.axes([0.7, 0.05, 0.075, 0.075])
        ip = InsetPosition(ax1, [0.55, 0.84, 0.15, 0.05])
        ax_prev.set_axes_locator(ip)
        bprev = Button(ax_prev, '<<')
        bprev.on_clicked(self.callback.prev)
        ## Next
        ax_next = plt.axes([0.81, 0.05, 0.075, 0.075])
        ip = InsetPosition(ax1, [0.75, 0.84, 0.15, 0.05])
        ax_next.set_axes_locator(ip)
        bnext = Button(ax_next, '>>')
        bnext.on_clicked(self.callback.next)

        # Bounding Boxes
        ax_chec = plt.axes([0.1, 0.05, 0.35, 0.075])
        ip = InsetPosition(ax1, [0.05, 0.5, 0.9, 0.3])
        ax_chec.set_axes_locator(ip)
        ax_chec.text(0.05, 0.85, "Bounding Boxes", transform=ax_chec.transAxes)
        check = CheckButtons(ax_chec,
                             ('characters', 'lines'),
                             (False, False))
        check.on_clicked(self.callback.update_bboxes)

        # Filtering
        ## Image
        ax_text_image = plt.axes([0.1, 0.1, 0.1, 0.075])
        ip = InsetPosition(ax1, [0.26, 0.38, 0.64, 0.05])
        ax_text_image.set_axes_locator(ip)
        entry_image = TextBox(ax_text_image, 'images',
                              initial="image_id,image_id")
        entry_image.on_submit(self.callback.submit_images)
        ## Characters
        ax_text_char = plt.axes([0.1, 0.2, 0.1, 0.075])
        ip = InsetPosition(ax1, [0.21, 0.3, 0.69, 0.05])
        ax_text_char.set_axes_locator(ip)
        entry_char = TextBox(ax_text_char, 'chars',
                             initial="U+3055,U+3056")
        entry_char.on_submit(self.callback.submit_chars)
        ## Reset
        ax_reset = plt.axes([0., 0., 1., 1.])
        ip = InsetPosition(ax1, [0.05, 0.2, 0.2, 0.05])
        ax_reset.set_axes_locator(ip)
        breset = Button(ax_reset, 'Reset')
        breset.on_clicked(self.callback.reset)

        plt.show()


class ImageIndex(object):
    ind = 0

    def __init__(self, images, shape, fig):
        self._all = images
        self._filtered = self._all
        self._filter_im = []
        self._filter_char = []
        self._shape = shape
        self._fig = fig
        self._nav, self._ax = fig.get_axes()
        self.show_char_bbox = False
        self.show_line_bbox = False

        self._fig.canvas.mpl_connect("motion_notify_event", self.hover)
        self.show_image()

    def next(self, event):
        self.ind += 1
        self.show_image()

    def prev(self, event):
        self.ind -= 1
        self.show_image()

    def submit_index(self, text):
        try:
            i = int(text)
        except ValueError:
            self._nav.text(0.075, 0.01, "Please enter a valid integer.",
                           color='red',
                           transform=self._nav.transAxes)
        else:
            i = i % len(self._filtered)
            self.ind = i
            self.show_image()

    def submit_images(self, text):
        self._filter_im += text.split(',')
        self.show_image()

    def submit_chars(self, text):
        self._filter_char += text.split(',')
        self.show_image()

    def reset(self, event):
        self._filtered = self._all
        self._filter_im = []
        self._filter_char = []
        self.ind = 0
        self.show_image()

    def show_image(self):
        self._ax.clear()
        # Filter by image_id
        if len(self._filter_im) > 0:
            self._filtered = [i for i in self._all
                              if any([f in _image_id(i)
                                      for f in self._filter_im])]
        else:
            self._filtered = self._all

        # Filter by character
        if len(self._filter_char) > 0:
            self._filtered = [i for i in self._filtered
                              if any([c in i.char_labels
                                      for c in self._filter_char])]
        num = len(self._filtered)
        index = self.ind % num
        meta = self._filtered[index]
        image = meta.load_image(self._shape)
        if len(image.shape) == 2:
            self._ax.imshow(image, cmap='gray')
        else:
            self._ax.imshow(image)

        def make_box(bbox, color):
            rect = patches.Rectangle((bbox.xmin, bbox.ymin),
                                     bbox.xmax - bbox.xmin,
                                     bbox.ymax - bbox.ymin,
                                     linewidth=2, edgecolor=color,
                                     facecolor='none')
            self._ax.add_patch(rect)

        if self.show_char_bbox:
            for b, c in zip(*(meta.char_bboxes, meta.char_labels)):
                if len(self._filter_char) > 0:
                    if c in self._filter_char:
                        make_box(b, 'g')
                else:
                    make_box(b, 'g')

        if self.show_line_bbox:
            for b in meta.line_bboxes:
                make_box(b, 'b')

        self._nav.texts = [self._nav.texts[0]]
        self._nav.text(0.05, 0.95,
                       "{}/{} ({})".format(index, num - 1, _image_id(meta)),
                       weight='bold',
                       transform=self._nav.transAxes)

        plt.draw()

    def hover(self, event):
        self._ax.texts = []
        ind = self.ind % len(self._filtered)
        if event.inaxes == self._ax:
            bbs = self._filtered[ind].char_bboxes
            if self.show_line_bbox:
                lines = self._filtered[ind].line_bboxes
                chars_show = []
                for bb in lines:
                    if in_region((event.xdata, event.ydata), bb):
                        chars_show.insert(
                            0, [i for i in range(len(bbs))
                                if in_region(bbs[i], bb)]
                        )
            else:
                chars_show = [[
                    i for i in range(len(bbs))
                    if in_region((event.xdata, event.ydata), bbs[i])
                ]]
            # Update annotation
            if len(chars_show) > 0:
                all_chars = self._filtered[ind].char_labels
                for i in range(len(chars_show)):
                    for j in chars_show[i]:
                        x = self._ax.get_xlim()[1] + 20 + i * 25
                        y = ((bbs[j].ymax - bbs[j].ymin) / 2.
                             + bbs[j].ymin)
                        self._ax.text(x, y,
                                     code2char(all_chars[j]),
                                     color='black',
                                     fontproperties=font(12))
                self._ax.figure.canvas.draw_idle()
        else:
            self._ax.figure.canvas.draw_idle()

    def update_bboxes(self, label):
        if label == 'characters':
            self.show_char_bbox = not self.show_char_bbox
        elif label == 'lines':
            self.show_line_bbox = not self.show_line_bbox
        self.show_image()


def _image_id(meta):
    """Pull image_id from filepath."""
    return meta.filepath.split('/')[-1].strip('.jpg')
