# Source lineage

Built from the current retained copies of the user's earlier packages:

- `BioCal3D_3D_investigations_v1.zip` — SHA-256 `1cbb7346f06c375e0e67fd43f6ca0a1eb1f1f8bd3988a2cfed2b06fdd99a0764`
- `BioCal3D_3D_followup_v1.zip` — SHA-256 `74f0e191a6bc882c452ecd31fe6c581807fe3850ddc3f4e37c3c400708db75e8`
- `BioCal3D_ROI_review_v3_9.zip` — SHA-256 `6ff5298a1abd107ad47ac81d773650f5f84a81218393694615c32ad146ec9c48`

Merged and extended the v1/follow-up training, sampling, controls and reporting implementation. Official model adapters remain from v1; third-party repositories and pretrained weights are not bundled. The latest ROI rebuild code was inspected to ground schema-v3 approval inventory, relative sample paths, geometry hashes, embedded vertex RGB and extraction settings.

Added or changed: portable approved-snapshot validation and split policy; unified staged runner/launchers and outputs; source-preserving interior sampling; input perturbations on new checkpoints; paired stage-order diagnostics; RGB focus including confident errors; output locking; new regression/integration checks. Old-checkpoint reproduction against old ROI geometry was omitted.

No old datasets, cached features, pretrained weights, training results or copied third-party model repositories are included.
