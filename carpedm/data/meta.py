#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

"""Image metadata management.

This module loads and manages metadata stored as CSV files in the
raw data directory.

Attributes:
    FONT_FILE (str): Filename of ``.ttc`` file for displaying Japanese
        character fonts.
    DEFAULT_SEED (int): The default random seed.

Examples:

    .. literalinclude:: ../../source/examples/data_exploration.py
        :language: python
        :lines: 17

    Load, view, and generate a dataset of single kana characters.

    .. literalinclude:: ../../source/examples/data_exploration.py
        :language: python
        :lines: 21, 26, 31

    Load and view a dataset of sequences of 3 kanji.

    .. literalinclude:: ../../source/examples/data_exploration.py
        :language: python
        :lines: 22, 27

    Load and view a dataset of full pages.

    .. literalinclude:: ../../source/examples/data_exploration.py
        :language: python
        :lines: 23, 28

.. note::
    :name: note on image shape

    Unless stated otherwise, image shape arguments in this module should
    be a tuple (height, width). Tuple values may be one of the following:

    1. :obj:`int`
        specifies the absolute size (in pixels) for that axis
    2. :obj:`float`
        specifies a rescale factor relative to the original image size
    3. :obj:`None`
        the corresponding axis size will be computed such that the
        aspect ratio is maintained. If both height and width are `None`,
        no resize is performed.

    .. caution::

        If the new shape is smaller than the original, information will be
        lost due to interpolation.

Todo:
    * Tests
        * generate_dataset
    * Sort characters by reading order, i.e. character ID.
    * Rewrite data as CSV following original format
    * Data generator option instead of writing data.
    * Old file cleanup if files overwritten in ``generate_dataset``
    * Output formats and/or generator return types for ``generate_dataset``
        * numpy
        * hdf5
        * pandas DataFrame
    * Chunked ``generate_dataset`` option to include partial characters.
    * Low-priority:
        * Fix bounding box display error in ``view_images``
        * specify number of character type in sequence
            * e.g. 2 Kanji, 1 kana
        * Filter examples by image in which they appear
        * Instead of padding, fill specified shape with surrounding
"""
import itertools
import glob
import json
import os
import random
import shutil
import sys
import warnings
from collections import Counter

import numpy as np

from carpedm.data.download import get_books_list
from carpedm.data.providers import TFDataSet
from carpedm.data.io import CSVParser, DataWriter
from carpedm.data.lang import JapaneseUnicodes, Vocabulary, code2char
from carpedm.data.ops import in_region

FONT_FILE = "HiraginoMaruGothic.ttc"
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), FONT_FILE)
DEFAULT_SEED = 123456


def _get_split(split):
    """Returns (ratio, heldout)."""
    valid_books = get_books_list('pmjtc')
    if isinstance(split, float):
        ratio = split
        assert 0 <= ratio <= 1, "Invalid split {}.".format(split)
        heldout = None
    elif isinstance(split, str):
        if split in valid_books:
            ratio = None
            heldout = [split]
        else:
            try:
                ratio = float(split)
                heldout = None
            except ValueError:
                ratio = None
                heldout = split.split(',')
                for bib in heldout:
                    assert bib in valid_books, "Invalid ID %s" % bib
            else:
                assert 0 <= ratio <= 1, "Invalid split {}.".format(split)
    else:
        raise ValueError(
            "Invalid split {}. Must be float or string.".format(split)
        )

    return ratio, heldout


def num_examples_per_epoch(data_dir, subset):
    """Retrieve number of examples per epoch.

    Args:
        data_dir (str): Directory where processed dataset is stored.
        subset (str): Data subset.

    Returns:
        int: Number of examples.

    """
    filepath = os.path.join(data_dir, 'num_examples.json')
    try:
        with open(filepath, 'r') as fp:
            return json.load(fp)[subset]
    except FileNotFoundError:
        print("Expected %s in %s. Returning None." %
              ('num_examples.json', data_dir))
        return


def _num_examples_per_epoch(data_dir):
    """Retrieve number of examples.

    Args:
        data_dir: Directory where processed dataset is stored.

    Returns:
        dict: Number examples per subset

    """
    filepath = os.path.join(data_dir, 'num_examples.json')
    if os.path.exists(filepath):
        with open(filepath, 'r') as fp:
            return json.load(fp)
    else:
        return {'train': None, 'dev': None, 'test': None}


def font(size):
    """Fonts helper for matplotlib."""
    from matplotlib.font_manager import FontProperties

    return FontProperties(fname=FONT_PATH, size=size)


class MetaLoader(object):
    """Class for loading image metadata."""

    def __init__(self,
                 data_dir,
                 test_split='hnsd00000',
                 dev_split=0.1,
                 dev_factor=1,
                 vocab_size=None,
                 min_freq=0,
                 reserved=('<PAD>', '<GO>', '<END>', '<UNK>'),
                 charset='all',
                 image_scope='char',
                 seq_len=None,
                 seq_maxlen=None,
                 verbose=False,
                 seed=None):
        """Initializer.

        Args:
            data_dir (str): Top level directory containing directories
                for each bibliography (e.g. 200003076, hnsd00000).
            test_split (float or str): Either the ratio of all data
                to use for testing or specific bibliography ID(s). Use
                comma-separated IDs for multiple books.
            dev_split (float or str): Either the ratio of all data
                to use for dev/val or specific bibliography ID(s). Use
                comma-separated IDs for multiple books.
            dev_factor (int): Size of development set should be
                divisible by this value. Useful for training on
                multiple GPUs.
            vocab_size (None or int): Maximum size of the vocabulary.
                If None, include all possible characters, minus those
                filtered out after applying min_freq.
            min_freq (int): Minimum frequency of tokens in vocab.
            reserved (tuple): Strings for reserved tokens,
                e.g. ("<PAD>", "<S>", "<UNK>", "</S>").
                Note: Indices of token in given tuple will be used
                for its corresponding integer ID.
            charset (str): ID for the character set to use.
            image_scope (str): Image problem scope.
            seq_len (None or int): (Minimum) number of characters to
                include in image if image_scope is 'seq'. If seq_maxlen
                is None, specifies the deterministic sequence length.
            seq_maxlen (None or int): Maximum sequence length.
            verbose (bool): Display verbose output.
            seed (int): Number for seeding random number generator.
                If None, the DEFAULT_SEED is used.

        """
        # for bib in get_books_list('pmjtc'):
        #     assert bib in os.listdir(data_dir), (
        #         "Expected directory {} in data_dir {}. Check path or "
        #         "download data with 'download-pmjtc'".format(bib, data_dir),
        #     )
        if image_scope == 'seq':
            assert seq_len and seq_len > 0, "seq_len must be positive."
            if seq_maxlen:
                assert seq_maxlen >= seq_len, (
                    "{} < {}".format(seq_maxlen, seq_len)
                )
        assert image_scope in ['char', 'seq', 'page', 'line']
        self.data_dir = data_dir
        self._test_ratio, self._test_heldout = _get_split(test_split)
        self._dev_ratio, self._dev_heldout = _get_split(dev_split)
        self.reserved_tokens = reserved
        self.image_scope = image_scope
        self._dev_factor = dev_factor
        self._charset = charset
        self._C = JapaneseUnicodes(self._charset)
        self._V = vocab_size
        self._min_freq = min_freq
        self._seq_len = seq_len
        self._seq_maxlen = seq_maxlen if seq_maxlen else seq_len
        if seed is None:
            random.seed(DEFAULT_SEED)
        else:
            random.seed(seed)
        self._verbose = verbose
        self._image_meta = {}
        self._build_image_lists()
        self._build_vocabulary()

    def generate_dataset(self,
                         out_dir,
                         subset,
                         format_store='tfrecords',
                         shape_store=None,
                         shape_in=None,
                         num_shards=8,
                         num_threads=4,
                         target_id='image/seq/char/id',
                         sparse_labels=False,
                         chunk=False,
                         character=True,
                         line=False,
                         label=True,
                         bbox=False,
                         overwrite=False):
        """Generate data usable by machine learning algorithm.

        Args:
            out_dir (str): Directory to write the data to if 'generator'
                not in ``format_store``.
            subset (str): The subset of data to generate.
            format_store (str): Format to save the data as.
            shape_store (tuple or None): Size to which images are
                resized for storage (on disk). The default is to not
                perform any resize. Please see this
                `note on image shape`_ for more information.
            shape_in (tuple or None): Size to which images are resized
                by interpolation or padding before being input to a
                model. Please see this `note on image shape`_ for
                more information.
            num_shards (int): Number of sharded output files.
            num_threads (int): Number of threads to run in parallel.
            target_id (str): Determines the target feature (one of keys
                in dict returned by ImageMeta.generate_features).
            sparse_labels (bool): Provide sparse_labels, only used for
                TFRecords.
            chunk (bool): Instead of using the original image,
                extract non-overlapping chunks and corresponding
                features from the original image on a regular grid. Pad
                the original image to divide by ``shape`` evenly.

                .. note::

                    Currently only characters that fit entirely in
                    the block will be propagated to appropriate features.

            character (bool): Include character  info, e.g. label, bbox.
            line (bool): Include line info (bbox) in features.
            label (bool): Include label IDs in features.
            bbox (str or None): If not None, include bbox in features
                as unit (e.g. 'pixel', 'ratio' [of image]))
            overwrite (bool): Overwrite any existing data.

        Returns:
            :obj:`carpedm.data.providers.DataProvider`:
                Object for accessing batches of data.

        """
        assert num_shards < num_threads or not num_shards % num_threads, (
            "num_shards %d and num_threads %d not compatible" %
            (num_shards, num_threads)
        )
        assert format_store in DataWriter.available_formats + ['generator'], (
            "Unsupported data format %s" % format_store
        )
        self._check_subset(subset)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        if 'generator' in format_store:
            raise NotImplementedError
        else:
            examples_per_epoch = _num_examples_per_epoch(out_dir)
            if not examples_per_epoch[subset] or overwrite:
                writer = DataWriter(format_out=format_store,
                                    images=self._image_meta[subset],
                                    image_shape=shape_store,
                                    vocab=self.vocab, chunk=chunk,
                                    character=character, line=line,
                                    label=label, bbox=bbox,
                                    subdirs=self.image_scope == 'char')
                fname_prefix = os.path.join(out_dir, subset)
                if overwrite:
                    def remove_files(f, p, i):
                        for f in glob.glob(p + '*'):
                            os.remove(f)
                    shutil.rmtree(fname_prefix, onerror=remove_files)
                    print("Deleted old files...") if self._verbose else None
                examples_per_epoch[subset] = writer.write(
                    fname_prefix, num_threads, num_shards)
                numex_file = 'num_examples.json'
                with open(os.path.join(out_dir, numex_file), 'w') as fp:
                    json.dump(examples_per_epoch, fp)

            if format_store == 'tfrecords':
                return TFDataSet(
                    data_dir=out_dir, subset=subset, target_id=target_id,
                    num_examples=examples_per_epoch[subset],
                    pad_shape=self.max_image_size(None, shape_in),
                    sparse_labels=sparse_labels)
            elif format_store == 'jpeg' or format_store == 'jpg':
                # TODO: return an object for Task accessing raw images
                return None

    def max_image_size(self, subset, static_shape=(None, None)):
        """Retrieve the maximum image size (in pixels).

        Args:
            subset (str or None): Data subset from which to get image
                sizes. If None, return max sizes of all images.
            static_shape (:obj:`tuple` of :obj:`int`): Define static
                dimensions. Axes that are None will be of variable size.

        Returns:
            tuple: Maximum size (height, width)

        """
        if subset:
            self._check_subset(subset)
            subset = [subset]
        else:
            subset = self._image_meta.keys()
        heights, widths = zip(*[img.new_shape(static_shape)
                                for s in subset
                                for img in self._image_meta[s]])
        return max(heights), max(widths)

    def view_images(self, subset, shape=None):
        """View and explore images in a data subset.

        References:
            Annotation code based on Stack Overflow answer `here`_.

        Args:
            subset (str): The subset to iterate through.
                One of {'train', 'dev', 'test'}.
            shape (tuple or None): Shape to which images are
                resized. Please see this `note on image shape`_ for
                more information.

        .. _here: https://stackoverflow.com/a/47166787

        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.widgets import Button, TextBox, CheckButtons
        except ImportError:
            warnings.warn("The view_images method is not available."
                          "Please install matplotlib if you wish to use it.")
            return

        self._check_subset(subset)
        images = self._image_meta[subset]
        n = len(images)

        class Index(object):
            ind = 0
            fig, ax = plt.subplots()

            def __init__(self):
                self.show_char_bbox = False
                self.show_line_bbox = False

                plt.subplots_adjust(bottom=0.2)
                self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
                self.show_image()

            def next(self, event):
                self.ind += 1
                self.show_image()

            def prev(self, event):
                self.ind -= 1
                self.show_image()

            def submit(self, text):
                try:
                    i = int(text)
                except ValueError:
                    self.ax.text(0.075, 0.1, "Please enter a valid integer.",
                                 transform=plt.gcf().transFigure)
                else:
                    i = i % n
                    self.ind = i
                    self.show_image()

            def show_image(self):
                self.ax.clear()
                index = self.ind % n
                meta = images[index]
                image = meta.load_image(shape)
                if len(image.shape) == 2:
                    self.ax.imshow(image, cmap='gray')
                else:
                    self.ax.imshow(image)

                if self.show_char_bbox:
                    for b in meta.char_bboxes:
                        rect = patches.Rectangle((b.xmin, b.ymin),
                                                 b.xmax - b.xmin,
                                                 b.ymax - b.ymin,
                                                 linewidth=2, edgecolor='g',
                                                 facecolor='none')
                        self.ax.add_patch(rect)

                if self.show_line_bbox:
                    for b in meta.line_bboxes:
                        rect = patches.Rectangle((b.xmin, b.ymin),
                                                 b.xmax - b.xmin,
                                                 b.ymax - b.ymin,
                                                 linewidth=2, edgecolor='b',
                                                 facecolor='none')
                        self.ax.add_patch(rect)

                self.ax.text(0.3, 0.9,
                             "{}/{} ({})".format(
                                 index, n - 1, meta.filepath.split('/')[-1]),
                             transform=plt.gcf().transFigure)

                plt.draw()

            def hover(self, event):
                self.ax.texts = [self.ax.texts[0]]  # only keep index
                ind = self.ind % n
                if event.inaxes == self.ax:
                    bbs = images[ind].char_bboxes
                    if self.show_line_bbox:
                        lines = images[ind].line_bboxes
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
                        all_chars = images[ind].char_labels
                        for i in range(len(chars_show)):
                            for j in chars_show[i]:
                                x = self.ax.get_xlim()[1] + 20 + i * 25
                                y = ((bbs[j].ymax - bbs[j].ymin) / 2.
                                     + bbs[j].ymin)
                                self.ax.text(x, y,
                                             code2char(all_chars[j]),
                                             color='black',
                                             fontproperties=font(12))
                        self.ax.figure.canvas.draw_idle()
                else:
                    self.ax.figure.canvas.draw_idle()

            def update_bboxes(self, label):
                if label == 'show character bboxes':
                    self.show_char_bbox = not self.show_char_bbox
                elif label == 'show line bboxes':
                    self.show_line_bbox = not self.show_line_bbox
                self.show_image()

        callback = Index()
        axtext = plt.axes([0.59, 0.05, 0.1, 0.075])
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        axchec = plt.axes([0.1, 0.05, 0.35, 0.075])
        entry = TextBox(axtext, 'Go to', initial="0")
        entry.on_submit(callback.submit)
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(callback.next)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(callback.prev)
        check = CheckButtons(axchec,
                             ('show character bboxes', 'show line bboxes'),
                             (False, False))
        check.on_clicked(callback.update_bboxes)
        plt.show()

    def data_stats(self,
                   which_sets=('train', 'dev', 'test'),
                   which_stats=('majority', 'frequency', 'unknowns'),
                   save_figures=False,
                   id_start=None,
                   id_stop=None):
        """Print or show data statistics.

        Args:
            which_sets (tuple): Data subsets to see statistics for.
            which_stats (tuple): Statistics to view. Default gives all
                options.
            save_figures (bool): Save figures when possible.
            id_start (int): lowest character ID to include in
                visualizations.
            id_stop (int): highest character ID to include in
                visualizations.

        """

        def all_labels_flat(sub):
            meta = self._image_meta[sub]
            return [u for img in meta for u in img.char_labels]

        if 'frequency' in which_stats:

            import matplotlib.pyplot as plt

            rr = len(self.reserved_tokens)
            start = rr if id_start is None else id_start
            stop = self.vocab.get_num_classes() if id_stop is None else id_stop
            nn = stop - start + 1
            colors = ['green', 'blue', 'red']
            fig, ax = plt.subplots()
            rects = []
            bar_width = 1.0 / (len(which_sets) + 1)
            # center groups of bars
            ind = np.arange(nn)
            centers = ind + bar_width * len(which_sets) / 2.
            max_count = 0
            for i in range(len(which_sets)):
                chars = [self.vocab.char_to_id(u)
                         for u in all_labels_flat(which_sets[i])]
                chars = [cid for cid in chars if start <= cid <= stop]
                counts = Counter(chars)
                x = ind + i * bar_width
                y = [0] * nn
                for cid, count in counts.items():
                    if count > max_count:
                        max_count = count
                    y[cid - start] = count
                rects.append(ax.bar(x, y, bar_width, color=colors[i]))
            char_list = [self.vocab.id_to_char(i)
                         for i in range(start, stop + 1)]
            if start < rr:
                char_list = char_list[:rr - start] + list(
                    map(code2char, char_list[rr - start:])
                )
            else:
                char_list = map(code2char, char_list)

            ax.set_ylabel("Counts")
            ax.set_ylim([0, max_count])
            ax.set_title("Characters {0}-{1} Relative Frequency".format(
                start, stop))
            ax.set_xlabel("Characters")
            ax.set_xticks(centers)
            ax.set_xticklabels(char_list, fontproperties=font(10))
            ax.legend([r[0] for r in rects], which_sets)
            ax.autoscale()
            if save_figures:
                plt.savefig("{0}-{1}_frequency.svg".format(start, stop))
            plt.show()

        if 'majority' in which_stats:
            for primary in which_sets:
                chars = all_labels_flat(primary)
                total = len(chars)
                counts = Counter(chars)
                majority, count = counts.most_common(1)[0]
                print("Majority class from {0}: {1} ({2}), {3:.2f}%".format(
                    primary,
                    self.vocab.char_to_id(majority),
                    code2char(majority),
                    float(count) / total * 100.))
                secondary = list(which_sets)
                secondary.remove(primary)
                for subset in secondary:
                    chars = all_labels_flat(subset)
                    total = len(chars)
                    count = chars.count(majority)
                    print("\t% of {0}: {1:.2f}".format(
                        subset, count / float(total) * 100.))

        if 'unknowns' in which_stats:
            for subset in which_sets:
                chars = all_labels_flat(subset)
                chars = set(chars)
                unknowns = [c for c in chars if
                            self.vocab.id_to_char(self.vocab.char_to_id(c))
                            == "<UNK>"]
                print("Unknowns in {}: {}".format(subset, len(unknowns)))

    def _build_image_lists(self):
        bib_meta = {}
        for bib in [d for d in os.listdir(self.data_dir)
                    if '.' not in d and '_' not in d]:
            meta_file = os.path.join(self.data_dir, bib,
                                     bib + '_coordinate.csv')
            bib_meta[bib] = self._parse_coordinate_csv(meta_file, bib)
        self._split_dataset(bib_meta)

    def _parse_coordinate_csv(self, csvfn, bib_id):
        """Parse a CSV file containing character coordinate info.

        Args:
            csvfn (str): Path to csvfile to parse.
            bib_id (str): Bibliography id.

        """
        encodings_check = ['shift_jis', 'shift_jisx0213', 'cp932', 'utf8']

        if self._verbose:
            print("Reading CSV for %s..." % bib_id)

        # check multiple encodings
        encoding = 'utf8'
        for code in encodings_check:
            try:
                temp = open(csvfn, encoding=code)
                temp.read()
            except UnicodeDecodeError:
                temp.close()
                if self._verbose:
                    print("File %s not encoded as %s" % (csvfn, code))
            else:
                temp.close()
                encoding = code
                if self._verbose:
                    print("File %s is valid %s encoding" % (csvfn, code))

        with open(csvfn, 'r', encoding=encoding) as csvfile:
            parser = CSVParser(csvfile, self.data_dir, bib_id)
            if self.image_scope == 'page':
                result = parser.parse_pages()
            elif self.image_scope == 'line':
                result = parser.parse_lines()
            elif self.image_scope == 'seq':
                result = parser.parse_sequences(self._C, self._seq_len,
                                                self._seq_maxlen)
            else:
                result = parser.parse_characters(self._C)
        if self._verbose:
            print("Finished reading %s" % bib_id)
        return result

    def _split_dataset(self, bib_meta):
        """Splits data into train, development, and test subsets.

        Args:
            bib_meta (dict): Dictionary of lists of Image namedtuples,
                one for each bibliography entry.

        """
        total_examples = sum(map(lambda x: len(x), bib_meta.values()))

        if self._test_heldout:
            self._image_meta['test'] = []
            for test_bib in self._test_heldout:
                if self._dev_heldout and test_bib in self._dev_heldout:
                    warnings.warn("%s is in both dev and test" % test_bib)
                    self._image_meta['test'] += bib_meta[test_bib]
                else:
                    self._image_meta['test'] += bib_meta.pop(test_bib)
        if self._dev_heldout:
            self._image_meta['dev'] = []
            for dev_bib in self._dev_heldout:
                self._image_meta['dev'] += bib_meta.pop(dev_bib)

        remaining_meta = list(itertools.chain(*bib_meta.values()))
        random.shuffle(remaining_meta)
        if isinstance(self._test_ratio, float):
            test_ix = int(total_examples * self._test_ratio)
            assert test_ix < len(remaining_meta)
            self._image_meta['test'] = remaining_meta[:test_ix]
            remaining_meta = remaining_meta[test_ix:]
        if isinstance(self._dev_ratio, float):
            dev_ix = int(total_examples * self._dev_ratio)
            assert dev_ix < len(remaining_meta)
            self._image_meta['dev'] = remaining_meta[:dev_ix]
            remaining_meta = remaining_meta[dev_ix:]
        self._image_meta['train'] = remaining_meta
        while len(self._image_meta['dev']) % self._dev_factor != 0:
            # take examples from training set
            self._image_meta['dev'].append(
                self._image_meta['train'].pop())

    def _build_vocabulary(self):
        counts = Counter()
        for x in self._image_meta['train']:
            for unicode in x.char_labels:
                if self._C.in_charset(unicode):
                    counts[unicode] = counts[unicode] + 1
        vocab = [t[0] for t in counts.most_common(self._V)
                 if t[1] >= self._min_freq]
        self.vocab = Vocabulary(reserved=self.reserved_tokens, vocab=vocab)

    def _check_subset(self, subset):
        assert subset in self._image_meta.keys(), (
            "Invalid data subset %s" % subset
        )
