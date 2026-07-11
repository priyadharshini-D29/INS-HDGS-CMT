# Release Checklist — v1.0

Steps to cut the `v1.0.0` GitHub release for this repository.

## Before tagging
- [ ] `pytest -q` passes locally and in CI
- [ ] `ruff check .` and `ruff format --check .` clean
- [ ] `README.md` badges resolve; architecture image renders
- [ ] `requirements.txt` / `environment.yml` install cleanly in a fresh env
- [ ] `CITATION.cff` validates (e.g. `cffconvert --validate`)
- [ ] Git LFS objects pushed (`git lfs push --all origin`)
- [ ] No raw dataset / secrets tracked (`git ls-files | grep -Ei '\.xdf$'` is empty)
- [ ] `docs/REPRODUCIBILITY_CHECKLIST.md` has the reported-run commit hash

## Tag & release
```bash
git tag -a v1.0.0 -m "INS-HDGS-CMT v1.0.0 — initial public release"
git push origin v1.0.0
```
- [ ] Create the GitHub Release from tag `v1.0.0`
- [ ] Paste `docs/RELEASE_NOTES_v1.0.md` into the release body
- [ ] Attach the compiled manuscript PDF (optional)

## After release
- [ ] Enable Zenodo–GitHub integration and publish → obtain DOI
- [ ] Add the DOI badge to `README.md` and `CITATION.cff`
- [ ] Enable GitHub Discussions with the suggested categories (see `docs/DISCUSSION_CATEGORIES.md`)
- [ ] Announce / link from the paper's data-availability statement
