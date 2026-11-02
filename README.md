# spektralab-engine

A spectral film simulation behind a C ABI, on Apple Metal. It takes a linear RGB frame, exposes it onto a negative or slide film, prints it through an enlarger onto paper, scans the result and returns 16-bit RGBA.

**This repository ships no film or paper data, so it builds but cannot render a picture until you supply a resources folder in the documented format.** See [docs/RESOURCE_FORMAT.md](docs/RESOURCE_FORMAT.md).

The API contract is in [docs/API_SPEC.md](docs/API_SPEC.md) and in the comments of [include/spektralab.h](include/spektralab.h).

## Build and test

Needs Xcode command line tools; plain `clang++` and `metal` through `xcrun`.

```sh
make                 # build/libspektralab.dylib, build/spektralab.metallib (macOS 15, arm64)
make ios             # build/ios and build/ios-sim: libspektralab.a, spektralab.metallib (iOS 17, arm64)
make xcframework     # build/xcframework/spektralab.xcframework
make test            # the tests that need no resource data
make test RES=<dir>  # those, plus the tests that read <dir>, a resources folder (these need numpy)
make clean
```

`BUILD=<dir>` puts every product in another directory.

## Usage

Copy the matching `spektralab.metallib` into your resources folder first.

```c
#include "spektralab.h"

spk_engine* engine = spk_engine_create("/path/to/resources", NULL);
if (!engine) { puts(spk_last_error(NULL)); return 1; }

/* width x height x 3 float32, row 0 at the top, linear ProPhoto RGB by default */
spk_image input = { pixels, width, height, 3 };
spk_session* session = spk_open(engine, &input, "{\"film_stock\":\"kodak_portra_400\"}", NULL);
if (!session) { puts(spk_last_error(NULL)); return 1; }

spk_result out;
if (spk_render(session, "preview", &out) == SPK_OK) {
    /* pixel (x, y), channel c */
    uint16_t v = out.rgba16[4 * (y * out.row_stride_px + x) + c];
    spk_result_free(&out);
}

/* a print-side change reuses the developed negative */
spk_set_params(session, "{\"print_exposure\":1.2}", NULL);
if (spk_reprint(session, "preview", &out) == SPK_OK) spk_result_free(&out);

spk_session_release(session);
spk_engine_destroy(engine);
```

## Platforms

macOS 15 and iOS 17, device and simulator, on arm64. Metal is required.

## Licence

The code is licensed under GPL-3.0-or-later; see [LICENSE](LICENSE). Copyright and acknowledgements are in [NOTICE.md](NOTICE.md).
