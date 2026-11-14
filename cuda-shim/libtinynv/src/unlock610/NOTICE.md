# Vendored gpu-unlock-ada portable parsers

Source: https://github.com/JustAResearcher/gpu-unlock-ada
Commit: `8855e7bca8ff142204e02301331fd56b26699819` (v1.1.1) — the same pinned,
independently exercised revision selected during input research
(`tools/rtx4090-unlock/manifest.json`,
`docs/driver/rtx4090-tensor-unlock/assets.md`).

These files are copied VERBATIM from `kernel/gpu_unlock/` at that commit. They
are the pure byte parsers / layout arithmetic of the upstream transaction
(booter geometry, ELF sections, radix plan, WPR meta packing, FWSEC-FRTS ROM
extraction, GFW evaluation, reservation layout). Each already carries a
userspace path (`#ifdef __KERNEL__`); no line was changed. The kernel-only
transaction driver (`unlock_610.c`) is NOT copied; it is adapted to the
TinyGPU transport in `../tensor_unlock.c`, which records this provenance too.

License: GPL-2.0-only (upstream `LICENSE`, preserved in
`tools/rtx4090-unlock/upstream/UPSTREAM-LICENSE` and in every prepared bundle).
Personal supervised use on the owner's Mac only, per the current scope
revision in `docs/driver/rtx4090-tensor-unlock/spec.md`. No redistribution.

sha256 at the pin (verified against the research checkout on 2026-11-12):

```
0083606528ad11277b52f2ae7ad2e3b9f501fcf17e2120e6f205804eab0fba5a  arch_610.h
fce58df506620383e26f2523eeee354f5b73a8b48e385829b4900a924a77f521  diagnostic.h
5a36666290df66edc703611086a36adb148cdb953babf4529d114c9ef098e1b6  gfw_eval.c
530ef704290d566bd4300a2036c07808d56214a814925349e0a1db926ede35ad  gfw_eval.h
bde52cb4c22a16bb345fef9caaa1c6f15c19c674d6985fde98e9d4c518aca375  layout_610.c
afe45505f455f7af90a9d1920b416301bfb7e4ff1778f4f7e10186084618bb5e  layout_610.h
35e1c844f6fa40c387a1cda9ddfc29f67f256748df28d0a718d920ba4a8f61fa  prep_610.c
47a7aa339b12fc67b99047b5c1666cd66bcba1bf7621cb5c0b9df9094cda2fa4  prep_610.h
602a488ea200bbee67e90c600fda89cb6092d17709b314dbdf75b215f4a45253  vbios_610.c
17990fa3a2464dbc1446ef5f349fe1486b39d9edf859ab1d8896857f65b3b7e0  vbios_610.h
```

`gfw_eval.c` equals the audited-kernel pin recorded in
`tools/rtx4090-unlock/manifest.json:upstream` (`5a366662…`).
