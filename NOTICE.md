# Third-party notices

This project builds on open-source software and research data. Their licenses
apply to their respective components; the MIT License in [`LICENSE`](LICENSE)
does not replace them.

## Data

- **MAESTRO v3.0.0** — piano audio and aligned MIDI, used for training and
  evaluation. Distributed by the Magenta team under
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  The dataset is not included in this repository or in the released checkpoint.
  See [`MODEL_LICENSE.md`](MODEL_LICENSE.md) for the resulting checkpoint terms.

## Runtime dependencies

The project depends on, among others:

- PyTorch (BSD-style license)
- NumPy, SciPy, pandas (BSD-style licenses)
- librosa, soundfile, pretty_midi, music21, mir_eval (BSD/MIT-style licenses)
- verovio (LGPL-3.0-or-later, with additional terms for the Verovio toolkit)
- PyMuPDF (AGPL-3.0 or commercial license; review before commercial deployment)
- FastAPI, Uvicorn, python-multipart (MIT/BSD-style licenses)
- FluidSynth and FluidR3_GM SoundFont when installed in the Docker image;
  review their own licenses before redistribution or commercial use.

Dependency versions and licenses can change. Before a commercial deployment,
audit the exact versions in `requirements.txt` and the Docker images.

## Fonts and notation

- Music notation is rendered with verovio and the Bravura music font. Their
  license terms apply to the rendered output and embedded font data.
- The application UI uses system/Noto fonts when available.

## References

The project cites external piano-transcription work for comparison only. No
external model weights or source code are included unless explicitly stated.
