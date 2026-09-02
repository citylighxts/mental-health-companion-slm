import Foundation
import CoreML
import Tokenizers

/// Custom prefill+decode loop replacing swift-transformers' own
/// LanguageModelWithStatefulKVCache.predictNextTokenScores(), which hardcodes
/// causalMask shape to query_length=1 even during prefill (see
/// Sources/Models/LanguageModel.swift in swift-transformers 1.3.3). That's
/// correct for decode steps but WRONG for prefill of a multi-token prompt.
/// Mirrors the causal-mask construction validated against the reference HF
/// model in Python (scripts/convert_coreml.py's build_causal_mask).
func buildCausalMask(queryLength: Int, endStep: Int) -> [Float16] {
    var mask = [Float16](repeating: -Float16.infinity, count: queryLength * endStep)
    let past = endStep - queryLength
    for i in 0..<queryLength {
        for j in 0...(past + i) {
            mask[i * endStep + j] = 0
        }
    }
    return mask
}

func predictLogits(
    model: MLModel,
    state: MLState,
    fullSequence: [Int32],
    isPrefill: Bool
) async throws -> MLTensor {
    let queryTokens: [Int32] = isPrefill ? fullSequence : [fullSequence.last!]
    let queryLength = queryTokens.count
    let endStep = fullSequence.count

    let inputIdsTensor = MLTensor(shape: [1, queryLength], scalars: queryTokens)
    let maskValues = buildCausalMask(queryLength: queryLength, endStep: endStep)
    let causalMaskTensor = MLTensor(shape: [1, 1, queryLength, endStep], scalars: maskValues)

    let inputDict: [String: MLTensor] = [
        "inputIds": inputIdsTensor,
        "causalMask": causalMaskTensor,
    ]
    let outputs = try await model.prediction(from: inputDict, using: state)
    guard let logits = outputs["logits"] else {
        throw NSError(domain: "CoreMLDebugTest", code: 1, userInfo: [NSLocalizedDescriptionKey: "no logits output"])
    }
    return logits
}

func argmaxLastPosition(_ logits: MLTensor, queryLength: Int) async throws -> Int32 {
    let lastIndex = queryLength - 1
    let lastPosition = logits[0, lastIndex, 0...]
    let ids = try await lastPosition.argmax().shapedArray(of: Int32.self)
    return ids.scalars[0]
}

func generate(
    model: MLModel,
    tok: Tokenizer,
    promptTokens: [Int32],
    maxNewTokens: Int,
    eosTokenId: Int32
) async throws -> [Int32] {
    let state = model.makeState()
    var sequence = promptTokens
    var generated: [Int32] = []

    var logits = try await predictLogits(model: model, state: state, fullSequence: sequence, isPrefill: true)
    var nextToken = try await argmaxLastPosition(logits, queryLength: sequence.count)

    while generated.count < maxNewTokens {
        generated.append(nextToken)
        sequence.append(nextToken)
        if nextToken == eosTokenId { break }

        logits = try await predictLogits(model: model, state: state, fullSequence: sequence, isPrefill: false)
        nextToken = try await argmaxLastPosition(logits, queryLength: 1)
    }
    return generated
}

@main
struct CoreMLDebugTest {
    static func main() async {
        do {
            let mlpackageURL = URL(fileURLWithPath: "/Users/hanaazizah/Developer/genz-mentalhealth-slm/models/MentalHealthCompanionInt4.mlpackage")
            let tokenizerFolder = URL(fileURLWithPath: "/Users/hanaazizah/Developer/genz-mentalhealth-slm/models/fused-hf-model")

            print("[\(Date())] Compiling model for macOS...")
            let compiledURL = try await MLModel.compileModel(at: mlpackageURL)

            let config = MLModelConfiguration()
            config.computeUnits = .all
            let model = try MLModel(contentsOf: compiledURL, configuration: config)

            print("[\(Date())] Loading tokenizer...")
            let tok = try await AutoTokenizer.from(modelFolder: tokenizerFolder)

            let prompts = [
                "aku capek banget hari ini, rasanya pengen nyerah aja",
                "aku ngerasa cemas banget besok mau presentasi",
                "hari ini seru banget nonton film sama temen",
            ]

            for prompt in prompts {
                let messages = [["role": "user", "content": prompt]]
                let promptTokens = try tok.applyChatTemplate(messages: messages).map { Int32($0) }
                print("[\(Date())] Generating for prompt: \(prompt)")
                let generated = try await generate(
                    model: model, tok: tok, promptTokens: promptTokens, maxNewTokens: 80, eosTokenId: 128009
                )
                let text = tok.decode(tokens: generated.map { Int($0) }, skipSpecialTokens: true)
                print("RESPONSE: \(text)")
                print("---")
            }
        } catch {
            print("ERROR: \(error)")
        }
    }
}
