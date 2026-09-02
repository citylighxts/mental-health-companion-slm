// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "CoreMLDebugTest",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3")
    ],
    targets: [
        .executableTarget(
            name: "CoreMLDebugTest",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers")
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        )
    ]
)
