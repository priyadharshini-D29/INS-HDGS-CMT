# `checkpoints/` — Trained model weights

Per-fold ensemble checkpoints produced by training, e.g.
`ins_hdgs_cmt_headline/fold_XX/member_Y.pt`.

These files are large and tracked with **Git LFS** (`*.pt` in `.gitattributes`).
If they exceed hosting limits they are distributed via the GitHub Release or a
Zenodo archive instead — regenerate locally with `reproducibility/train.sh`.
