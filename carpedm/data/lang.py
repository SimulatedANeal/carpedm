#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

"""Language-specific and unicode utilities.

Todo:
    * Variable UNK token in Vocabulary
"""
import abc
import os


def code2hex(unicode):
    """Returns hex integer for a unicode string."""
    if 'U+' in unicode:
        unicode = unicode.lstrip('U+')
    return int(unicode, 16)


def code2char(unicode):
    """Returns the unicode string for the character."""
    return chr(code2hex(unicode))


class CharacterSet(object):
    """Character set abstract class."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, charset):
        """Initializer

        Args:
            charset (str): ID for types of characters to include.
        """
        self._ranges = self._unicode_ranges(charset)

    @abc.abstractmethod
    def _unicode_ranges(self, charset):
        """Returns appropriate unicode ranges for specified ``charset``.

        Args:
            charset (str): ID of character set to use.

        Returns:
            :obj:`list` of :obj:`tuple`: Unicode ranges [(low, high)]
        """

    def in_charset(self, unicode):
        """Check if a character is in the defined character set.

        Args:
            unicode (str): String representation of unicode value.

        """
        hexcode = code2hex(unicode)
        if any([r[0] <= hexcode <= r[1] for r in self._ranges]):
            return True
        else:
            return False


class JapaneseUnicodes(CharacterSet):
    """Utility for accessing and manipulating Japanese character
    unicodes.

    Inherits from :obj:`CharacterSet`.

    Unicode ranges taken from [1] with edits for exceptions.

    References:
        [1] http://www.unicode.org/charts/

    """

    PUNCTUATION = [
        (int('25a0', 16), int('25ff', 16)),  # square
        (int('25b2', 16), int('25b3', 16)),  # triangle
        (int('25cb', 16), int('25cf', 16)),  # circle
        (int('25ef', 16), int('25ef', 16)),  # big circle
        (int('3200', 16), int('32ff', 16)),  # filled big circles
        (int('3000', 16), int('303f', 16)),  # CJK symbols, punctuation
        (int('3099', 16), int('309e', 16)),  # voicing, iteration marks
        (int('30a0', 16), int('30a0', 16)),  # double hyphen
        (int('30fb', 16), int('30fe', 16)),  # dot, prolonged, iteration
        (int('ff5b', 16), int('ff64', 16)),  # brackets, halfwidth punctuation
        (int('ffed', 16), int('ffee', 16))  # halfwidth square, circle
    ]

    HIRAGANA = [
        (int('3040', 16), int('3096', 16)),
        (int('309f', 16), int('309f', 16))  # より
    ]

    KATAKANA = [
        (int('30a1', 16), int('30fa', 16)),
        (int('30ff', 16), int('30ff', 16)),  # コト
        (int('ff65', 16), int('ff9d', 16))  # halfwidth
    ]

    KANA = HIRAGANA + KATAKANA

    MISC = [
        (int('0030', 16), int('0039', 16)),  # digits
        (int('ff00', 16), int('ff5a', 16)),  # roman characters
        (int('ffa0', 16), int('ffdc', 16)),  # hangul characters
        (int('ffe0', 16), int('ffec', 16)),  # symbols
        # (int('003f', 16), int('003f', 16)),  # question mark
    ]

    # Kanji covers full CJK set and extensions
    KANJI = [
        (int('3400', 16), int('4db5', 16)),
        (int('4e00', 16), int('9fea', 16)),
        (int('f900', 16), int('fad9', 16)),
        (int('20000', 16), int('2ebe0', 16)),
    ]

    ALL = HIRAGANA + KATAKANA + KANJI + PUNCTUATION + MISC

    def __init__(self, charset):
        super(JapaneseUnicodes, self).__init__(charset)

    def _unicode_ranges(self, charset):
        if charset == 'all':
            ranges = JapaneseUnicodes.ALL
        else:
            ranges = []
            if 'hiragana' in charset:
                ranges += JapaneseUnicodes.HIRAGANA
            elif 'katakana' in charset:
                ranges += JapaneseUnicodes.KATAKANA
            elif 'kana' in charset:
                ranges += JapaneseUnicodes.KANA
            if 'kanji' in charset:
                ranges += JapaneseUnicodes.KANJI
            if 'punct' in charset:
                ranges += JapaneseUnicodes.PUNCTUATION
            if 'misc' in charset:
                ranges += JapaneseUnicodes.MISC
        assert len(ranges) > 0, "Invalid character set."
        return ranges


class Vocabulary(object):
    """Simple vocabulary wrapper.

    References:
        Lightly modified TensorFlow "im2txt" `Vocabulary`_.

    ..  _Vocabulary: https://github.com/tensorflow/models/blob/master/
        research/im2txt/im2txt/data/build_mscoco_data.py

    """

    UNK = "<UNK>"

    def __init__(self, reserved, vocab):
        """Initializes the vocabulary.

        Args:
            reserved (tuple): Tuple of reserved tokens.
            vocab: (list): List of vocabulary entries, ideally (for
                visualization) in descending order by frequency.

        """
        self._vocab = {}

        for ix, char in enumerate(vocab):
            self._vocab[char] = ix

        add2id = 0
        for i in range(len(reserved)):
            if i in self._vocab.values():
                add2id += 1
        self._vocab = {key: idx + add2id for key, idx in self._vocab.items()}
        for i, char in enumerate(reserved):
            self._vocab[char] = i

        try:
            self._unk_id = reserved.index(self.UNK)
        except ValueError:
            print("'{}' token not provided. Setting to highest ID.".format(
                self.UNK
            ))
            self._vocab[self.UNK] = len(self._vocab)
        self._rev_vocab = {idx: key for key, idx in self._vocab.items()}

    def save(self, out_dir, as_unicode=False):
        types = self.types()
        with open(os.path.join(out_dir, 'vocab.txt'), 'w') as f:
            for token in types:
                if as_unicode:
                    try:
                        token = code2char(token)
                    except ValueError:
                        token = token
                f.write(token + '\n')

    def types(self):
        return [self._rev_vocab[idx] for idx in sorted(self._rev_vocab.keys())]

    def char_to_id(self, char):
        """Returns the integer id of a character string."""
        if char in self._vocab:
            return self._vocab[char]
        else:
            return self._unk_id

    def id_to_char(self, char_id):
        """Returns the character string of a integer id."""
        if char_id in self._rev_vocab:
            return self._rev_vocab[char_id]
        else:
            return self.UNK

    def get_num_classes(self):
        """Returns number of classes, includes <UNK>."""
        return len(self._vocab)
