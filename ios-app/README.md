# MentalHealthCompanion (iOS)

SwiftUI chat app that runs `MentalHealthCompanionInt4.mlpackage` fully on-device
via Core ML (stateful KV-cache).

## Setup

The compiled model + tokenizer files are gitignored (GBs, regenerate instead of
committing). Before building:

```bash
mkdir -p MentalHealthCompanion/Resources
cp -R ../models/MentalHealthCompanionInt4.mlpackage MentalHealthCompanion/Resources/
cp ../models/fused-hf-model/tokenizer.json \
   ../models/fused-hf-model/tokenizer_config.json \
   ../models/fused-hf-model/special_tokens_map.json \
   ../models/fused-hf-model/chat_template.jinja \
   MentalHealthCompanion/Resources/
xcodegen generate
```

Then open `MentalHealthCompanion.xcodeproj` (or `xcodebuild`/simulator as usual).

## Why a custom generation loop

`ChatViewModel.swift` implements its own prefill+decode loop against `MLModel`
directly rather than using swift-transformers' `LanguageModel.generate()`.
That library's `LanguageModelWithStatefulKVCache.predictNextTokenScores()`
hardcodes the `causalMask` tensor's query-length dimension to `1` even during
prefill of a multi-token prompt (see `Sources/Models/LanguageModel.swift` in
swift-transformers 1.3.3) — correct for single-token decode steps, but it
breaks causal masking between prompt positions during prefill and produces
incoherent/off-tone output. `coreml-debug-test/` is the isolated harness used
to diagnose and verify the fix (run outside the simulator for fast iteration).

## KNOWN ISSUE: app doesn't run correctly on iOS (Simulator or real device)

**Status: the model and generation logic are correct — verified extensively —
but the app currently cannot run correctly on iOS at all, only on macOS.**

### What works

`coreml-debug-test/` (a standalone Swift package, runs on macOS natively, not
iOS) loads `MentalHealthCompanionInt4.mlpackage` and runs the exact same
prefill+decode loop as `ChatViewModel.swift`, with `computeUnits = .all`. It
produces correct, on-tone, coherent Indonesian output for every test prompt.
This proves the *model* (training, fusion, Core ML conversion, INT4
quantization) and the *generation logic* (causal mask construction, KV-cache
state handling) are both correct.

### What's broken

The identical model, loaded and run the identical way on iOS — both the
Simulator and a physical device (tested: iPhone, iOS 26.6.1) — fails or
produces garbage:

| Platform | `computeUnits` | Result |
|---|---|---|
| macOS (native) | `.all` | ✅ Correct, coherent output |
| iOS Simulator | `.cpuOnly` | Loads, but degenerate/repetitive output |
| iOS Simulator | `.cpuAndNeuralEngine` | Loads, but degenerate/repetitive output |
| iOS Simulator | `.all` / `.cpuAndGPU` | ❌ Fails to load — error `-14` |
| Physical device | `.cpuOnly` | ❌ Fails to load — error `-14` |
| Physical device | `.all` | ❌ Fails to load — error `-14` |
| Physical device | (runtime-compiled `.mlpackage` instead of Xcode's precompiled `.mlmodelc`) | Same `-14` — ruled out build-time vs. runtime compilation as the cause |

Full error text:
```
Failed to build the model execution plan using a model architecture file
'.../MentalHealthCompanionInt4.mlmodelc/model.mil' with error code: -14.
```

### Root cause

This matches a confirmed, unfixed Apple bug:
[apple/coremltools#2548](https://github.com/apple/coremltools/issues/2548) —
*"Stateful MIL to CoreML breaks when fixed and flexible inputs are present."*
Our model mixes:
- **Fixed-shape state** (`keyCache`/`valueCache`, declared with a static
  `kv_cache_shape` tuple in `scripts/convert_coreml.py`)
- **Flexible-shape inputs** (`inputIds`/`causalMask`, declared with
  `ct.RangeDim` for the sequence dimensions)

The error message ("mixture of enumerated and range shape flexibility")
matches this exactly. iOS's Core ML runtime rejects this combination when
actually building the execution plan; macOS's Core ML runtime does not — same
`.mlpackage`, same `ct.convert()` call, platform-dependent result.

### What a real fix requires

Removing `ct.RangeDim` from the causal-mask/input-ids shapes entirely (e.g.
fixed-width `causalMask` at `max_context_size` with the actual valid length
encoded in the mask's `-inf`/`0` pattern instead of the tensor's shape). This
is more than a config change: the current `StatefulLlamaForCausalLM.forward()`
derives the KV-cache slice bounds (`end_step = causal_mask.shape[-1]`) from
the mask's *shape* — a fixed-width mask needs that derived from the mask's
*values* or a separate position input instead, and `torch.jit.trace` (used in
`scripts/convert_coreml.py`) cannot correctly capture a runtime-varying value
used for slice bounds — only shapes are traced symbolically, not tensor
values. Making this work would likely require switching the export from
`torch.jit.trace` to `torch.export` (a different coremltools conversion API,
`ct.convert(exported_program, ...)`), which is a substantial rework, not a
parameter tweak — and isn't guaranteed to sidestep the underlying Apple bug.

### Alternative not yet tried

Skip Core ML for on-device inference entirely and convert to GGUF for
`llama.cpp` (a completely different runtime/ecosystem, unaffected by this
Core ML–specific bug). Would need a new conversion script and a different
iOS integration approach (e.g. `llama.cpp`'s Swift/ObjC bindings) — the LoRA
training and `models/fused-hf-model` checkpoint are reusable as-is; only the
Core ML–specific conversion/deployment work (`scripts/convert_coreml.py`,
`ios-app/`) would need replacing.
