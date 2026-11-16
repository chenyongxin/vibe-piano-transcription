# Model checkpoint terms

The source code in this repository is released under the MIT License in
[`LICENSE`](LICENSE).

The trained checkpoint `best.pt` (distributed as a GitHub Release asset) is
**not** covered by the MIT License. It is made available for non-commercial
research, education, and personal evaluation, with the following conditions:

1. **No commercial use.** You may not use the checkpoint or a model derived
   from it for commercial products or paid services without separate written
   permission from the author.
2. **Attribution.** If you publish results obtained with the checkpoint,
   please cite this repository and acknowledge the MAESTRO dataset.
3. **Research purpose.** The checkpoint is a research artifact. It has known
   failure modes on fast passages, ornaments, triplets, arpeggios, pedaling,
   and audio-score alignment; see
   [`docs/10_limitations.md`](docs/10_limitations.md).
4. **No warranty.** The checkpoint is provided "as is" without warranty of any
   kind. The author is not responsible for any loss arising from its use.
5. **No dataset redistribution.** The checkpoint does not include the MAESTRO
   audio or MIDI dataset. Obtain MAESTRO separately under its own terms.

## Training data attribution

The checkpoint was trained and evaluated on the MAESTRO v3.0.0 dataset:

- Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna
  Huang, Sander Dieleman, Erich Elsen, Jesse Engel, and Douglas Eck.
  *Enabling Factorized Piano Music Modeling and Generation with the MAESTRO
  Dataset.* ICLR 2019.
- MAESTRO is distributed under the
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
  license. Its terms apply to the dataset; they are also the reason the
  distributed checkpoint is limited to non-commercial research use.

## Code and model licensing are separate

- Code: MIT License.
- Checkpoint: the non-commercial research terms above.
- Third-party libraries, fonts, and soundfonts keep their own licenses; see
  [`NOTICE.md`](NOTICE.md).
