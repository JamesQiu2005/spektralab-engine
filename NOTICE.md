# Notice

Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.
Licensed under GPL-3.0-or-later; see [LICENSE](LICENSE).

## Mathematics reproduced from spektrafilm

The film simulation model in this engine is reproduced from
[spektrafilm](https://github.com/andreavolpato/spektrafilm) by Andrea Volpato and optimized for mobile. That applies to these parts:

| Area | Files |
|---|---|
| Spectral upsampling, film exposure, density curves, DIR couplers | `src/film/setup.cpp`, `src/film/develop.cpp`, `kernels/film.metal` |
| Auto-exposure meter | `src/film/meter.cpp` |
| Enlarger, filtration, printing, paper response, scan | `src/print/build.cpp`, `src/print/setup.cpp`, `src/print/run.cpp`, `kernels/print.metal` |
| CAM16-UCS gamut compression | `src/print/cam16.cpp`, `kernels/print.metal` |
| Gaussian blurs and lens blur | `src/fx/gaussian.cpp`, `kernels/fx_blur.metal` |
| DIR-coupler diffusion | `src/fx/dir_blur.cpp` |
| Halation | `src/fx/halation.cpp`, `kernels/fx_halation.metal` |
| Film grain and its random draws | `src/fx/grain.cpp`, `src/fx/rng.cpp`, `kernels/fx_grain.metal`, `kernels/fx_random.h` |
| Print glare | `src/fx/glare.cpp`, `kernels/fx_glare.metal` |
| Scanner blur and unsharp mask | `src/fx/scanner.cpp` |

The parameter names and the profile file format in `docs/` also follow spektrafilm's.
