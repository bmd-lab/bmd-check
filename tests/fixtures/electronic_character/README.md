# Recorded VASP fixtures for the C2 electronic-character assessment

These gzip-compressed `vasprun.xml` files are unmodified copies of recorded
VASP outputs from the pymatgen test suite
(<https://github.com/materialsproject/pymatgen>, commit
`52e69c26e45b0205e4f5206e30c94a0619c0d64a`). They are third-party material
distributed under pymatgen's MIT License (`LICENSE.pymatgen`), not under this
repository's license.

They are used only as recorded electronic evidence for tests. They were not
produced by BMD Compute and they do not validate any material's experimental
electronic character.

| Fixture | Upstream path | Evidence exercised |
|---|---|---|
| `Al_mp_smearing.vasprun.xml.gz` | `test-files/io/vasp/outputs/vasprun.Al.xml.gz` | Al, ISPIN=2, Methfessel-Paxton smearing, 13x13x13 mesh |
| `LiF_tetrahedron.vasprun.xml.gz` | `test-files/io/vasp/outputs/vasprun.LiF.xml.gz` | LiF, tetrahedron method, Fermi level at the valence-band edge |
| `Si_static.vasprun.xml.gz` | `test-files/io/vasp/fixtures/static_silicon/vasprun.xml.gz` | Si, ISPIN=2, 4x4x4 Monkhorst-Pack mesh |
| `Cu_gamma_only.vasprun.xml.gz` | `test-files/io/vasp/outputs/vasprun.etest1.xml.gz` | Cu, single (Gamma) k-point |
| `Si_extra_electron.vasprun.xml.gz` | `test-files/io/vasp/outputs/vasprun.charged.xml.gz` | Si with an added electron (partially filled conduction band) |
| `Li_sparse_tetrahedron.vasprun.xml.gz` | `test-files/io/vasp/fixtures/relaxation/vasprun.xml.gz` | Li metal, Bloechl tetrahedron weights, 6x6x6 mesh |
| `Xe_automatic_mesh.vasprun.xml.gz` | `test-files/io/vasp/outputs/vasprun.xe.xml.gz` | Xe, fully automatic mesh (divisions inferred from k-points) |
| `Si_unconverged.vasprun.xml.gz` | `test-files/io/vasp/outputs/vasprun.dfpt.unconverged.xml.gz` | Si, final electronic step not converged |

SHA-256:

```
85d10f9c6589b13632e225a55b280387e1dcf4250e1df236baf7588bdc967741  Al_mp_smearing.vasprun.xml.gz
5ae99b32b06eb053344752cee9ea7aafd2b3a9c97f4d609eb8ff4e4b382e69ff  Cu_gamma_only.vasprun.xml.gz
de84b8b85ea770ffd71ad2939f4b6e436d00b34f743897e0d37da8ebbd2a9cb6  LiF_tetrahedron.vasprun.xml.gz
b67bf93325050140f4ee1f49ab89aef7955abc4d6ad5d91550326c7d6526fc49  Li_sparse_tetrahedron.vasprun.xml.gz
c7c8b50af08769a45e46fd7510161aba888bf85633c504a9366291aee093974c  Si_extra_electron.vasprun.xml.gz
a9d74c3f928cd32880d77b81a1968a6abf2ca8b88a41516ecb8bdfe210e333fc  Si_static.vasprun.xml.gz
aaf12f732bf6b260ec1e9d517b5ab4ad29bde598a0be50c8bbb4bca27d347a7b  Si_unconverged.vasprun.xml.gz
ae956ffa8b97c7def497b189554f967e880bba115c205cc6dc2a95ef9cc30319  Xe_automatic_mesh.vasprun.xml.gz
```
