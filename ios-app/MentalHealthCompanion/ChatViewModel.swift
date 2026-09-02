import Foundation
import CoreML
import Tokenizers

/// Custom prefill+decode loop — NOT using swift-transformers' own
/// `LanguageModel.generate()` / `LanguageModelWithStatefulKVCache`. That
/// implementation hardcodes causalMask shape to query_length=1 even during
/// prefill of a multi-token prompt (see Sources/Models/LanguageModel.swift in
/// swift-transformers 1.3.3), which breaks causal masking between prompt
/// positions and produces garbage/off-topic output. This mirrors the
/// causal-mask construction validated against the reference HF model in
/// Python (scripts/convert_coreml.py's build_causal_mask), and against the
/// same fix verified standalone in coreml-debug-test/.
private func buildCausalMask(queryLength: Int, endStep: Int) -> [Float16] {
    var mask = [Float16](repeating: -Float16.infinity, count: queryLength * endStep)
    let past = endStep - queryLength
    for i in 0..<queryLength {
        for j in 0...(past + i) {
            mask[i * endStep + j] = 0
        }
    }
    return mask
}

private func predictLogits(
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
        throw NSError(domain: "ChatViewModel", code: 1, userInfo: [NSLocalizedDescriptionKey: "no logits output"])
    }
    return logits
}

private func argmaxLastPosition(_ logits: MLTensor, queryLength: Int) async throws -> Int32 {
    let lastIndex = queryLength - 1
    let lastPosition = logits[0, lastIndex, 0...]
    let ids = try await lastPosition.argmax().shapedArray(of: Int32.self)
    return ids.scalars[0]
}

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var isGenerating = false
    @Published var isModelLoading = true
    @Published var loadError: String?

    private var mlModel: MLModel?
    private var tokenizer: Tokenizer?

    /// Conversation history in chat-template format (role/content pairs).
    private var history: [[String: String]] = []

    /// Llama-3 Instruct end-of-turn token id (<|eot_id|>) — see
    /// models/fused-hf-model/generation_config.json.
    private static let eotTokenId: Int32 = 128009
    private static let maxNewTokens = 150

    func loadModel() async {
        isModelLoading = true
        loadError = nil
        do {
            guard let modelURL = Bundle.main.url(
                forResource: "MentalHealthCompanionInt4", withExtension: "mlmodelc"
            ) else {
                loadError = "Model file (MentalHealthCompanionInt4.mlmodelc) not found in app bundle."
                isModelLoading = false
                return
            }
            guard let resourceURL = Bundle.main.resourceURL else {
                loadError = "App bundle resource folder not found."
                isModelLoading = false
                return
            }
            let tok = try await AutoTokenizer.from(modelFolder: resourceURL)
            let config = MLModelConfiguration()
            // .cpuOnly is required on the iOS Simulator — .cpuAndGPU / .all fails to
            // build an execution plan there for this stateful INT4 model (error -14).
            // On a real device (with ANE), switch to .all for much better throughput.
            config.computeUnits = .cpuOnly
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            tokenizer = tok
            mlModel = model
        } catch {
            loadError = "Gagal load model: \(error.localizedDescription)"
        }
        isModelLoading = false
    }

    func send(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isGenerating, let model = mlModel, let tok = tokenizer else { return }

        isGenerating = true
        messages.append(ChatMessage(role: .user, text: trimmed))
        history.append(["role": "user", "content": trimmed])

        let assistantIndex = messages.count
        messages.append(ChatMessage(role: .assistant, text: ""))
        let historySnapshot = history

        // Detach from MainActor explicitly — model.prediction(from:using:) does
        // real CPU work and should not run on/block the UI thread.
        Task.detached { [weak self] in
            do {
                let promptTokens = try tok.applyChatTemplate(messages: historySnapshot).map { Int32($0) }
                let state = model.makeState()
                var sequence = promptTokens
                var generated: [Int32] = []

                var logits = try await predictLogits(model: model, state: state, fullSequence: sequence, isPrefill: true)
                var nextToken = try await argmaxLastPosition(logits, queryLength: sequence.count)

                while generated.count < Self.maxNewTokens {
                    generated.append(nextToken)
                    sequence.append(nextToken)
                    if nextToken == Self.eotTokenId { break }

                    let partialText = tok.decode(tokens: generated.map { Int($0) }, skipSpecialTokens: true)
                    Task { @MainActor in
                        guard let self, assistantIndex < self.messages.count else { return }
                        self.messages[assistantIndex].text = partialText
                    }

                    logits = try await predictLogits(model: model, state: state, fullSequence: sequence, isPrefill: false)
                    nextToken = try await argmaxLastPosition(logits, queryLength: 1)
                }

                let finalText = tok.decode(tokens: generated.map { Int($0) }, skipSpecialTokens: true)
                await MainActor.run {
                    guard let self else { return }
                    self.messages[assistantIndex].text = finalText
                    self.history.append(["role": "assistant", "content": finalText])
                    self.isGenerating = false
                }
            } catch {
                await MainActor.run {
                    guard let self else { return }
                    self.messages[assistantIndex].text = "⚠️ Gagal generate respons: \(error.localizedDescription)"
                    self.isGenerating = false
                }
            }
        }
    }
}
