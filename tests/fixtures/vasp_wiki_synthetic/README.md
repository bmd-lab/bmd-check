# SYNTHETIC test fixture corpus

This directory is a synthetic test fixture for BMD Agent's frozen VASP Wiki
corpus loader. It contains no VASP Wiki content. Titles, page text, site
rights and the license text are invented placeholders served from the
reserved host synthetic.invalid. BMD Check's default loader refuses this
corpus kind.

Regenerate with:
PYTHONPATH=src python -B tests/vasp_wiki_support.py tests/fixtures/vasp_wiki_synthetic
