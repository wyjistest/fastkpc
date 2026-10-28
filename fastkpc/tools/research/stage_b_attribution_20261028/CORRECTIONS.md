# Diagnostic declaration correction

The initial README says 79 L2 waves. This is an arithmetic typo:
ceil(49,919 / 640) = **78**, comprising 77 full waves and a 639-component tail.
L3 has 86 waves, comprising 85 full waves and a 478-component tail.

This correction is recorded while the authenticated canonical capture is still
running, before any captured replay timing or attribution result. Component
counts, inputs, order, contexts, width, native implementation and all gates are
unchanged. The observer already validates the correct expression
`(planned + 639) / 640`; it records the native's actual 78/86 counts.

Retain the originally frozen README/analyzer in attempt3/execution-source.
After the frozen campaign finishes, correct only the README's L2 count and the
analyzer's expected L2 marker count. Preserve both versions and bind the final
analysis code separately. No run is discarded or repeated for this correction.

The first offline analysis looked for kernel correlations only among driver API
records; it explicitly reported all 13,104 kernels as unmatched. This CUPTI
version associates these activities with runtime API IDs. The corrected analysis
matches exact IDs against both API domains and checks that IDs are unambiguous.
All kernel and memcpy correlations now resolve; no ordering/time heuristic is
used. Original analysis source and JSON remain as analysis-v1.py/json. All
interval unions, gaps, kernel resources and selected-wave results are asserted
unchanged. This is an offline analysis correction, without another GPU run.
