# BUCEX 1.8.7 release validation

Validated on 24 September 2026 from the credited BUCEX 1.8.6.1 source tree.

## Locked reference

- Reference archive: `uccle_copula_20260923T222238_406159Z.zip`
- Archive SHA-256: `2e337e0fbe09f8587ab1782e10c861507b35b73677163b7904e7d1e16d1ac64a`
- Resolved reference-config SHA-256: `9a0f9c082d92831bf1185b8e51b0093d20c2677f3bfee03988acbbab6c958c44`
- Daily-source SHA-256: `a08cefa73732dc3ed7e40b3bdf79c22090b16931b6ae71c502de642d842f6e38`

After removal of provenance comments and the output path, the resolved
`reference_20260923.json` is identical to the archived run configuration. The
final configuration changes only the Monte Carlo budget and output path:
four chains, 3,000 warm-up iterations and 8,000 retained iterations per chain.

## Checks completed

- The full Python test suite passed.
- Release, reporting and seasonal-workflow tests passed after the final edits.
- Every Python module compiled and every research JSON file parsed.
- Seasonal preflight recovered 538 complete blocks, MAM 1892--JJA 2026, with
  31 August 2026 as the last included day.
- Prior calibration recovered 30-year marginal standard deviations of
  0.379, 0.197 and 0.155 degrees Celsius for level, slope and seasonal
  innovations, and 0.30 degrees Celsius per decade for the initial rate.
- A four-chain smoke fit completed through reporting. It is an execution
  check only and is not inferential output.
- The core and optional manuscript-figure routes were rendered and inspected.
- The strict final-run gate rejects the earlier flagged run, as intended.

The actual final fit is deliberately not bundled. Run it with
`research/seasonal/config/final.json`, apply `research.seasonal.check_final`,
and build figures only after that gate passes. Exact commands are in
`FINAL_RUN.md`.
