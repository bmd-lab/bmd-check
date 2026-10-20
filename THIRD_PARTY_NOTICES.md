# Third-party notices

bmd-check's own source code and documentation are released under the MIT
License in `LICENSE`. That license does not cover the third-party material
listed here, which keeps its own license and attribution.

## VASP Wiki reference corpus

Location: `src/bmd_agent/reference_corpus/vasp_wiki/`

This directory holds verbatim copies of VASP Wiki pages (MediaWiki wikitext at
pinned upstream revisions, under `pages/`) and the verbatim text of their
license (`COPYING.GFDL-1.2.txt`). Those page and license files are third-party
material. They are distributed unmodified under the license recorded in the
directory's `manifest.json` and `NOTICE`, not under the MIT License.

Corpus release 1.0.0 contains 1,047 pages of the VASP Wiki
(<https://vasp.at/wiki/>), acquired on 2026-10-20 from the official MediaWiki
API. The site states that its content is available under the GNU Free
Documentation License 1.2 unless otherwise noted; the maintainer review
recorded the license as GFDL-1.2-only with no Invariant Sections and no
Front-Cover or Back-Cover Texts, and found no page-specific exception. The
authors of each page are the contributors listed in its upstream history.
The directory's `NOTICE` records the site, the site rights captured from the
MediaWiki API, the reviewed license fields, and every page's revision
identity, canonical URL, permalink to the preserved revision and history
(authors) URL. Images and other media referenced by the pages are not
included. The `README.md`, `manifest.json`, `NOTICE`, `RELEASES.json` and
`SHA256SUMS` files in that directory are BMD-authored metadata.

VASP itself, POTCAR/PAW datasets and other third-party dependencies remain
governed by their own licenses and access terms.

## Recorded VASP test fixtures (pymatgen)

Location: `tests/fixtures/electronic_character/`

The `*.vasprun.xml.gz` files are unmodified copies of recorded VASP outputs
from the pymatgen test suite (commit `52e69c26e45b0205e4f5206e30c94a0619c0d64a`).
They are distributed under pymatgen's MIT License, reproduced in
`tests/fixtures/electronic_character/LICENSE.pymatgen`; the directory's
`README.md` lists each file's upstream path and SHA-256. They are test data
only and are not part of the installed package.
