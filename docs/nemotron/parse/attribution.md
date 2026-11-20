# Third-Party Attribution

The Parse recipe includes fonts for the dental renderer and adapted document
scoring implementations. The original notices and licenses remain distributed
with the recipe; copies of the license texts are available here.

## Dental rendering fonts

The renderer uses DejaVu Sans and DejaVu Serif from the DejaVu font project.
The font notice retains Bitstream's copyright and trademark notice and states
that the DejaVu changes are in the public domain.

{download}`DejaVu font license and notices <licenses/DejaVu-LICENSE.txt>`

## GriTS table scoring

The GriTS scorer adapts the reference implementation from Microsoft Table
Transformer and the ParseBench vendored implementation. Copyright (C) 2021
Microsoft Corporation. Licensed under MIT.

Reference: Smock, Pesala and Abraham, *GriTS: Grid Table Similarity Metric for
Table Structure Recognition*, ICDAR 2023.

{download}`GriTS MIT license <licenses/GriTS-MIT.txt>`

## TEDS table scoring

The TEDS scorer contains the single-table TEDS implementation from PubTabNet.
Copyright 2020 IBM. Licensed under Apache-2.0. The unused batch-evaluation method
was removed; the source retains its original notice.

{download}`TEDS Apache-2.0 license <licenses/TEDS-Apache-2.0.txt>`

## Chart and text scoring

Chart parsing, associated fact extraction, Unicode normalization and numeric
assignment provide component scores for document evaluation and task-specific
reward functions. See the [scoring guide](rewards.md) for their input contracts
and limitations.
