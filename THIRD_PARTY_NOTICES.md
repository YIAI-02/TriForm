# Third-party notices

The DOPS source tree contains partial materialized snapshots of optional
third-party projects. Their licenses remain in their respective directories
and continue to apply to those files. The required analytical Functional-badge
workflow does not import or execute either project.

| Component | Location | Upstream | License | Revision |
|---|---|---|---|---|
| CENT | `submodules/CENT/` | https://github.com/Yufeng98/CENT | MIT | derived from `3b0f874aa2d0501b85e69164c5112106a40a941c` |
| LLMCompass | `submodules/LLMCompass/` | https://github.com/PrincetonUniversity/LLMCompass (`ISCA_AE`) | BSD-3-Clause | derived from `2e015fd2ee750e6cad8d7152df52551e5b41ef20` |

The retained CENT file contents match the listed upstream commit, but the
snapshot omits upstream Git metadata, the nested `aim_simulator` gitlink, and
generated Python bytecode; several script mode bits were normalized. The
omitted `aim_simulator` gitlink points to
`2eb1ee04f4e8f3c255c9926e7721081726aa4fe4`. The LLMCompass snapshot also
omits its nested `cost_model/supply_chain` gitlink at
`d98ac0586faaf63f8f4eac94aa7cd50f37421c1f`; all 136 retained file blobs and
mode bits match the listed upstream commit. Do not remove or replace the
license files shipped with either component.

The prebuilt `src/ramulator2` executable is intentionally excluded from the
Zenodo archive because the current repository snapshot does not record its
build provenance. The required Functional workflow uses the analytical PIM
backend and does not depend on that executable.

Optional CENT/Ramulator2 and LLMCompass workflows require obtaining the missing
nested dependencies from their upstream projects at documented revisions; the
included partial snapshots must not be described as standalone installations.
