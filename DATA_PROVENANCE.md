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

## Additional research metadata (2026-10-30)

The research audit also inspected the upstream
[GiantSteps-MTG key annotations](https://github.com/GiantSteps/giantsteps-mtg-key-dataset).
The current snapshot has 1,486 annotation files. The audit identified 1,349
single-key candidates and recorded 137 ambiguous/unknown annotations separately.
It identified 429 major-key candidates after checking exact track IDs and MD5
records against the original 604 tracks. These checks do not establish the
absence of remixes, near duplicates, or artist overlap.

This audit collected metadata only. No additional recordings have entered the
model, and this project does not grant redistribution rights over these
annotations or any corresponding audio. Acquisition manifests, derived
features, point coordinates, trained artifacts, and the technical presentation
remain local under ignored data/report directories until their publication
basis is documented.
