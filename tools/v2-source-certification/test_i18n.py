"""Prevent the recovered POT's missing plural-message regression."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inputs import ROOT

spec = importlib.util.spec_from_file_location(
    "pot", ROOT / "wordpress-theme/skyyrose-flagship-2/scripts/build-pot.py"
)
pot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pot)


class TranslationTests(unittest.TestCase):
    def test_runtime_plural_is_retained(self):
        plural, references = pot.collect_plurals()["%d published view"]
        self.assertEqual(plural, "%d published views")
        self.assertEqual(len(references), 1)
        source, line_number = references[0].rsplit(":", 1)
        self.assertEqual(source, "template-parts/commerce/product-hero.php")
        self.assertGreater(int(line_number), 0)
        source_lines = (pot.THEME / source).read_text().splitlines()
        self.assertIn(
            "_n( '%d published view', '%d published views',",
            source_lines[int(line_number) - 1],
        )
        rendered = pot.render(pot.collect())
        self.assertIn('msgid_plural "%d published views"\nmsgstr[0] ""\nmsgstr[1] ""', rendered)

    def test_excludes_fixtures_and_parses_double_quotes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scripts").mkdir()
            (root / "scripts/test.php").write_text(
                "<?php _n('fake', 'fakes', $n, 'skyyrose-flagship-2');"
            )
            (root / "page.php").write_text('<?php _n("one", "many", $n, "skyyrose-flagship-2");')
            with patch.object(pot, "THEME", root):
                self.assertEqual(pot.collect_plurals(), {"one": ("many", ["page.php:1"])})
