# ONNX release management

ISSUE: How do we manage IR and operator version #'s for a given release/pre-release both pre-declaration and post-declaration?

ISSUE: How do we use experimental on operators (or potentially IR features) for a given release/pre-release?

ISSUE: How do we communicate schedules and exit criteria for a given pre-release/release?

ISSUE: How do we stablize one pre-release/release while we are working on subsequent ones?

### Fodder from versioning.md that belongs here somewhere.
Per the rules of SemVer 2.0, during the initial development of ONNX:
* We use a MAJOR version of 0 for both the IR version and operator version.
* We will only increment the MINOR version in the face of either a breaking change as defined in this specification or the need to stabilize a version for specific engineering purposes.

Once we declare a stable/released version of ONNX (e.g., we hit 1.0.0), we will adhere to the standard SemVer rules for versioning.

