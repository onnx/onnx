<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->


## Overview

ONNX uses sigstore to provide assurances regarding the authenticity of
its release artifacts, with the goal of mitigating its exposure to downstream software
supply chain issues.

For the purposes of all security checks described here GitHub effectively acts as the trust root.  

### Sigstore signatures ###

#### Scope ####

onnx's release pipeline in GitHub Actions uses the Github Actions OIDC token
to certify release artifacts as originating from a specific repository or ref.
It does so using the public Sigstore <https://sigstore.dev> instance.
The following are important to keep in mind:

 * The machine identity used to obtain the Sigstore signature is also the one
   used to authenticate to PyPI.
 * While Sigstore's OIDC-based keyless signing procedure does not rely on any
   maintainer-controlled secrets, deploying cannot be done without manual
   maintainer review, and only repository admins can push ``v*`` tags.

Therefore as long as you trust GitHub's security controls, these checks
are appropriate.

#### Verifying Sigstore signatures issued through GitHub Actions OIDC ####

  * Install ``sigstore``
  * Download the ``.sigstore.json`` bundles from the GitHub release page
  * Download the release artifacts you are interested in through whichever channel you prefer
    (e.g. using ``pip wheel``, or manual download from GitHub/PyPI)

```bash
    #!/bin/bash

    export EXPECTED_VERSION=<version number goes here>
    export REPO=onnx/onnx
    sigstore verify github \
        --cert-identity "https://github.com/$REPO/.github/workflows/create_release.yml@refs/tags/v$EXPECTED_VERSION" \
        --ref "refs/tags/v$EXPECTED_VERSION" \
        --repo "$REPO" \
        onnx-$EXPECTED_VERSION-*.whl onnx-$EXPECTED_VERSION.tar.gz


    sigstore verify github  --bundle onnx-$EXPECTED_VERSION-py3-none-any.whl.sigstore.json --repo "$REPO" onnx-$EXPECTED_VERSION-*.whl
```

