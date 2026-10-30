# Data provenance and redistribution

The implementation uses the **GiantSteps Key Dataset** annotations and audio
previews described by Peter Knees et al., *Two Data Sets for Tempo Estimation
and Key Detection in Electronic Dance Music Annotated from User Corrections*
(ISMIR 2015).

- Dataset repository: https://github.com/GiantSteps/giantsteps-key-dataset
- Dataset page and requested citation: https://www.cp.jku.at/datasets/giantsteps/
- Primary preview mirror used by the upstream downloader:
  `https://www.cp.jku.at/datasets/giantsteps/backup/`
- Fallback preview host: `https://geo-samples.beatport.com/lofi/`

The project downloads one preview into a temporary file, verifies its upstream
MD5 where available, extracts a 36-number chroma feature vector, and deletes
the preview immediately. Raw audio is never committed or retained.

The MIT license in this repository covers only original project source code.
It does not purport to relicense GiantSteps annotations, audio, or derived
feature caches. No explicit redistribution license was found in the upstream
repository during implementation. Consequently, `data/annotations/`,
`data/features/`, `data/work/`, and trained models remain excluded from the
public repository until a license or permission basis is recorded here.

Until then, GitHub runs the harness self-tests on every push but does not claim
the real-data acceptance job passed. That job becomes active when an authorized
`KEYCLF_FEATURES_URL` is configured or `KEYCLF_FEATURES_COMMITTED=1` explicitly
records that the committed cache has been cleared for publication.
