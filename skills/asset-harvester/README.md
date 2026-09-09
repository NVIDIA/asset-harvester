# asset-harvester agent skill

An [agent skill](https://code.claude.com/docs/en/skills) that installs and
drives NVIDIA Asset Harvester. `SKILL.md` is the entry point; the agent
runtime loads it and the files it references.

Installed skills are resolved by the `name:` field in `SKILL.md`
(`asset-harvester`), typically into `.claude/skills/<name>/` or
`.agents/skills/<name>/`.

## License

This directory is released under dual **CC-BY-4.0 AND Apache-2.0** terms,
which is what the `SKILL.md` frontmatter records as
`license: CC-BY-4.0 AND Apache-2.0`. Every file here is covered:

| Files | License |
|---|---|
| `scripts/` (`validate_setup.py`) | **Apache-2.0**, same as the rest of this repository. Source files carry `SPDX-License-Identifier: Apache-2.0` headers. Full text travels with this directory in [`LICENSE-Apache-2.0.txt`](./LICENSE-Apache-2.0.txt). |
| `SKILL.md`, `skill-card.md`, `references/`, `evals/`, this `README.md` | dual **CC-BY-4.0 AND Apache-2.0**. Full Creative Commons Attribution 4.0 International text: [`LICENSE-CC-BY-4.0`](./LICENSE-CC-BY-4.0). |

This directory is designed to be copied out standalone into an agent skills
directory, so both licence texts and the [`NOTICE`](./NOTICE) attribution
travel with it rather than being reachable only from the repository root.

Note that the repository root `README.md` describes this project as
Apache-2.0. That remains true of the project's own source; the CC-BY-4.0
terms above apply **only to the documentation inside this skill
directory**, and the license text is distributed alongside it here rather
than at the repository root.

## Provenance

This skill was authored in NVIDIA's NuRec skills repository
(<https://github.com/NVIDIA/nurec-skills>, `main` at `5b9d287`,
2026-09-02), where its
documentation was already licensed under the same dual CC-BY-4.0 AND
Apache-2.0 terms; that licensing is carried over unchanged rather than
re-granted.

It was modified when it moved here: references to sibling skills that do
not ship in this repository were reduced to plain names, and documented
behaviour was corrected against this repository's actual code (CLI flag
semantics, camera sensor IDs, output layout, and the prerequisites the
validator really checks).

The skill only *drives* upstream artifacts — this repository's code and
Hugging Face models and datasets. Those retain their own
licenses; see the `metadata:` block in `SKILL.md` for the upstream
pointers.
