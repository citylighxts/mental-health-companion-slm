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
