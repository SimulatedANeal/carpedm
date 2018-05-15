#
# Copyright (C) 2018 Neal Digre.
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

import unittest

from carpedm.data.lang import Vocabulary, JapaneseUnicodes
from carpedm.data.lang import code2hex, code2char, char2code


class VocabularyTestCase(unittest.TestCase):

    types = ['U+4E4B', 'U+6CD5', 'U+5247', 'U+53EF']
    reserved = ('<PAD>', '<UNK>')

    def test_get_num_classes(self):
        # Unknown token provided in reserved
        vocab = Vocabulary(reserved=self.reserved, vocab=self.types)
        self.assertEqual(vocab.get_num_classes(),
                         len(self.types) + len(self.reserved))
        # Adds class for unknown type
        vocab = Vocabulary(reserved=(), vocab=self.types)
        self.assertEqual(vocab.get_num_classes(), len(self.types) + 1)

    def test_build_default_reserved(self):
        vocab = Vocabulary(reserved=self.reserved, vocab=self.types)
        chars = list(self.reserved) + self.types
        char_ids = list(range(len(chars)))
        for c, cid in zip(chars, char_ids):
            self.assertEqual(vocab.char_to_id(c), cid,
                             msg='{} != {}'.format(c, cid))
            self.assertEqual(vocab.id_to_char(cid), c,
                             msg='{} != {}'.format(cid, c))
        # Unknown
        self.assertEqual(vocab.id_to_char(vocab.char_to_id('U+5555')),
                         '<UNK>')

    def test_build_empty_reserved(self):
        vocab = Vocabulary(reserved=(), vocab=self.types)
        self.assertEqual(vocab.char_to_id('<UNK>'), len(self.types))


class JapaneseUnicodeTestCase(unittest.TestCase):

    def test_invalid_charset_id(self):
        self.assertRaises(AssertionError, JapaneseUnicodes, 'blah')

    def test_kana_charset(self):
        kana = JapaneseUnicodes('kana')
        self.assertTrue(kana.in_charset('U+3074'))
        self.assertTrue(kana.in_charset('30cf'))
        self.assertFalse(kana.in_charset('U+80FD'))

    def test_kanji_charset(self):
        kanji = JapaneseUnicodes('kanji')
        self.assertTrue(kanji.in_charset('U+80FD'))
        self.assertFalse(kanji.in_charset('U+3074'))

    def test_kana_and_kanji_charset(self):
        both = JapaneseUnicodes('kana+kanji')
        self.assertTrue(both.in_charset('U+30CF'))
        self.assertTrue(both.in_charset('80FD'))
        self.assertFalse(both.in_charset('U+3002'))

    def test_punctuation_charset(self):
        punctuation = JapaneseUnicodes('punct')
        self.assertTrue(punctuation.in_charset('U+303f'))
        self.assertFalse(punctuation.in_charset('U+3040'))

    def test_misc_charset(self):
        misc = JapaneseUnicodes('misc')
        self.assertTrue(misc.in_charset('ff00'))
        self.assertFalse(misc.in_charset('U+30ff'))

    def test_all_charset(self):
        all_set = JapaneseUnicodes('all')
        self.assertTrue(all_set.in_charset('fF00'))
        self.assertTrue(all_set.in_charset('U+303f'))
        self.assertTrue(all_set.in_charset('80FD'))
        self.assertTrue(all_set.in_charset('U+3074'))
        self.assertTrue(all_set.in_charset('30cf'))
        self.assertFalse(all_set.in_charset('U+1708'))

    def test_code_to_char(self):
        self.assertEqual(code2char('U+3400'), u'\u3400')

    def test_char_to_code(self):
        self.assertEqual(char2code(u'\u3400'), 'U+3400')
        self.assertEqual(char2code('さ'), 'U+3055')
        self.assertEqual(char2code('c'), 'U+0063')


if __name__ == '__main__':
    unittest.main()
