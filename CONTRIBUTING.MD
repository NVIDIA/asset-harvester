# Asset Harvester Contribution Rules

## Coding Guidelines

- Please follow the existing conventions in the relevant file, submodule, module, and project when you add new code or when you extend/fix existing functionality.

- Make sure that you can contribute your work to open source (no license and/or patent conflict is introduced by your code). You will need to [`sign`](#signing-your-work) your commit.

## Linting & Formatting

We use [Ruff](https://docs.astral.sh/ruff/) for both linting and code formatting.

### Setup
 If not installed already
```bash
pip install ruff
```

### Usage

Check for lint issues:
```bash
ruff check asset_harvester run_inference.py scripts
```

Auto-fix lint issues where possible:
```bash
ruff check --fix asset_harvester run_inference.py scripts
```

Check formatting (no changes):
```bash
ruff format --check asset_harvester run_inference.py scripts docs
```

Apply formatting:
```bash
ruff format asset_harvester run_inference.py scripts docs
```

### License Headers

Every `.py` file must start with the SPDX / Apache-2.0 license header (see any
existing source file for the exact text).  A checker script enforces this:

```bash
python scripts/check_license_headers.py          # check all files
python scripts/check_license_headers.py --fix     # auto-prepend missing headers
```

### Before Submitting a PR

Please ensure your changes pass all three checks:
```bash
ruff check asset_harvester run_inference.py scripts
ruff format --check asset_harvester run_inference.py scripts docs
python scripts/check_license_headers.py
```

Configuration lives in [`pyproject.toml`](pyproject.toml) at the repo root.

- Thanks in advance for your patience as we review your contributions; we do appreciate them!

## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1
    
    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129
    
    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1
    
    By making a contribution to this project, I certify that:
    
    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
    
    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
    
    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
    
    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```
