import Foundation

enum LocalWhisperError: Error, LocalizedError {
    case notInstalled
    case failed(String)
    case noOutput

    var errorDescription: String? {
        switch self {
        case .notInstalled:
            return "No se encontró mlx_whisper. Instálalo con: pip install mlx-whisper"
        case .failed(let msg):
            return "Whisper local falló: \(msg)"
        case .noOutput:
            return "Whisper local no devolvió texto."
        }
    }
}

/// Transcripción en la propia Mac con `mlx_whisper` (Whisper sobre MLX, Apple Silicon).
/// No usa API key ni red: llama al CLI ya instalado y reusa los modelos que tenga en la caché de
/// Hugging Face (~/.cache/huggingface/hub). Requiere `ffmpeg` (lo usa mlx_whisper para leer el audio).
final class LocalWhisperClient {
    static let shared = LocalWhisperClient()
    static let defaultModel = "mlx-community/whisper-large-v3-turbo"

    /// Una app lanzada desde Finder no hereda el PATH de la terminal: buscamos el CLI (y ffmpeg)
    /// en las rutas típicas de pip --user, pipx y Homebrew.
    private static let searchDirs: [String] = {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return ["\(home)/.local/bin", "/opt/homebrew/bin", "/usr/local/bin"]
    }()

    var executableURL: URL? {
        Self.searchDirs
            .map { URL(fileURLWithPath: $0).appendingPathComponent("mlx_whisper") }
            .first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    var isInstalled: Bool { executableURL != nil }

    /// Modelos Whisper de mlx-community ya descargados (no requieren red). Siempre incluye el default.
    static func installedModels() -> [String] {
        let hub = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        let names = (try? FileManager.default.contentsOfDirectory(atPath: hub.path)) ?? []
        let found = names
            .filter { $0.hasPrefix("models--mlx-community--whisper") }
            .map { "mlx-community/" + $0.dropFirst("models--mlx-community--".count) }
        return Array(Set(found + [defaultModel])).sorted()
    }

    func transcribe(audioURL: URL, language: String?, model: String) async throws -> String {
        guard let exe = executableURL else { throw LocalWhisperError.notInstalled }

        let outDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("klip-whisper-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: outDir) }

        var args = [audioURL.path,
                    "--model", model.isEmpty ? Self.defaultModel : model,
                    "--output-dir", outDir.path,
                    "--output-name", "out",
                    "--output-format", "txt",
                    "--verbose", "False"]
        // Solo códigos ISO cortos ("es", "en"…); vacío = autodetección.
        if let language, (2...3).contains(language.count), language.allSatisfy({ $0.isLetter }) {
            args += ["--language", language.lowercased()]
        }

        let process = Process()
        process.executableURL = exe
        process.arguments = args
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = (Self.searchDirs + ["/usr/bin", "/bin"]).joined(separator: ":")
        process.environment = env
        // stderr a archivo (no a Pipe): la barra de progreso podría llenar el buffer y bloquear el proceso.
        let errURL = outDir.appendingPathComponent("stderr.log")
        FileManager.default.createFile(atPath: errURL.path, contents: nil)
        let errHandle = try FileHandle(forWritingTo: errURL)
        defer { try? errHandle.close() }
        process.standardError = errHandle
        process.standardOutput = FileHandle.nullDevice

        let status: Int32 = try await withCheckedThrowingContinuation { cont in
            process.terminationHandler = { cont.resume(returning: $0.terminationStatus) }
            do { try process.run() } catch {
                process.terminationHandler = nil
                cont.resume(throwing: LocalWhisperError.failed(error.localizedDescription))
            }
        }

        guard status == 0 else {
            let log = (try? String(contentsOf: errURL, encoding: .utf8)) ?? ""
            let lastLine = log.split(whereSeparator: \.isNewline).last.map(String.init) ?? "código \(status)"
            throw LocalWhisperError.failed(lastLine)
        }
        guard let text = try? String(contentsOf: outDir.appendingPathComponent("out.txt"), encoding: .utf8) else {
            throw LocalWhisperError.noOutput
        }
        return text
    }
}
