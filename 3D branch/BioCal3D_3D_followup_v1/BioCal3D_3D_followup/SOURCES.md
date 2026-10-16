# Source and provenance

The unchanged original runner and official backend adapter are included to preserve compatibility with the uploaded 23-run results. Their SHA-256 fingerprints are tested by `test_followup.py`:

- `biocal3d_train_3d_investigations.py`: `9fde487d059ea6083fdbc29bc4ecd3bd51de02ef3fd775e1ac09f1270b1c9b98`
- `official_backends.py`: `25ffb1743c4b3ee95ac659b6085cae5ba79d253d80d8ce06dbc1fb4a16e8d270`

These match the provenance recorded in the user's completed 3d_prepared_20261013 runs. They load the user's existing official PointNeXt, Sonata and Concerto backends, rather than supplying substitute networks. Existing repositories and pretrained weights retain their upstream licenses; no checkpoints or third_party repositories are redistributed here.

New pathway/experiment/report source fingerprints are included in follow-up surface and run identities. The separate follow-up output is checked against the original split and geometry manifest. Uploaded results motivating the design: 3d_results_review_20261015_132940.zip, reviewed 2026-10-15.
