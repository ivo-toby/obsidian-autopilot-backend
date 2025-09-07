// sckit-capture.swift
// Minimal ScreenCaptureKit CLI to record per-app audio to WAV with optional segmentation.
// Build: xcrun swiftc -O -o sckit-capture sckit-capture.swift

import Foundation
import ScreenCaptureKit
import CoreMedia
import AudioToolbox

// MARK: - WAV Writer

final class WAVWriter {
    private let fileHandle: FileHandle
    private let sampleRate: UInt32
    private let channels: UInt16
    private let bitsPerSample: UInt16
    private let formatCode: UInt16 // 1=PCM, 3=IEEE float
    private var dataBytesWritten: UInt32 = 0

    init?(url: URL, sampleRate: Int, channels: Int, bitsPerSample: Int = 16, formatCode: Int = 1) {
        self.sampleRate = UInt32(sampleRate)
        self.channels = UInt16(channels)
        self.bitsPerSample = UInt16(bitsPerSample)
        self.formatCode = UInt16(formatCode)
        do {
            FileManager.default.createFile(atPath: url.path, contents: nil)
            self.fileHandle = try FileHandle(forWritingTo: url)
        } catch {
            fputs("Failed to open file: \(error)\n", stderr)
            return nil
        }
        writeHeaderPlaceholder()
    }

    private func writeHeaderPlaceholder() {
        // 44-byte PCM WAV header placeholder.
        // We'll patch sizes on close.
        let byteRate = sampleRate * UInt32(channels) * UInt32(bitsPerSample / 8)
        let blockAlign = UInt16(channels) * (bitsPerSample / 8)

        var header = Data()
        header.append("RIFF".data(using: .ascii)!)
        header.append(uint32(36)) // placeholder chunk size
        header.append("WAVE".data(using: .ascii)!)
        header.append("fmt ".data(using: .ascii)!)
        header.append(uint32(16)) // PCM fmt chunk size
        header.append(uint16(formatCode))  // format (1=PCM, 3=IEEE float)
        header.append(uint16(channels))
        header.append(uint32(sampleRate))
        header.append(uint32(byteRate))
        header.append(uint16(blockAlign))
        header.append(uint16(bitsPerSample))
        header.append("data".data(using: .ascii)!)
        header.append(uint32(0)) // placeholder data chunk size
        fileHandle.write(header)
    }

    func writePCM(_ data: Data) {
        fileHandle.write(data)
        dataBytesWritten &+= UInt32(data.count)
    }

    func close() {
        // Patch sizes in header.
        do {
            try fileHandle.seek(toOffset: 4)
            let riffSize = 36 &+ dataBytesWritten
            fileHandle.write(uint32(riffSize))
            try fileHandle.seek(toOffset: 40)
            fileHandle.write(uint32(dataBytesWritten))
            try fileHandle.close()
        } catch {
            fputs("Error finalizing WAV header: \(error)\n", stderr)
        }
    }

    private func uint16(_ v: UInt16) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
    private func uint32(_ v: UInt32) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
}

// MARK: - Audio Sink

final class AudioSink: NSObject, SCStreamOutput {
    private let sampleRate: Int
    private let channels: Int
    private let outDir: URL
    private let segmentSeconds: Int?
    private var writer: WAVWriter?
    private var startTime: CMTime?
    private var samplesWritten: Int64 = 0
    private var fileIndex: Int = 0
    // Input format (detected from CMSampleBuffer)
    private var inputChannels: Int = 0
    private var inputIsFloat32: Bool = true
    private var inputIsNonInterleaved: Bool = false
    private var formatInitialized: Bool = false


    init(outDir: URL, sampleRate: Int, channels: Int, segmentSeconds: Int?) {
        self.outDir = outDir
        self.sampleRate = sampleRate
        self.channels = channels
        self.segmentSeconds = segmentSeconds
    }

    private func newFileURL() -> URL {
        let ts = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let name = String(format: "meeting_audio_%@_%03d.wav", ts, fileIndex)
        fileIndex += 1
        return outDir.appendingPathComponent(name)
    }

    private func ensureWriter(actualSampleRate: Int) {
        if writer == nil {
            let url = newFileURL()
            FileManager.default.createFile(atPath: url.path, contents: nil)
            // We always write PCM16 on disk
            writer = WAVWriter(url: url, sampleRate: actualSampleRate, channels: channels, bitsPerSample: 16, formatCode: 1)
            samplesWritten = 0
        }
    }

    private func rotateIfNeeded() {
        guard let seg = segmentSeconds, seg > 0 else { return }
        let seconds = Double(samplesWritten) / Double(sampleRate)
        if seconds >= Double(seg) {
            closeWriter()
        }
    }

    func closeWriter() {
        if let w = writer {
            w.close()
            writer = nil
        }
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of outputType: SCStreamOutputType) {
        guard outputType == .audio else { return }
        guard CMSampleBufferDataIsReady(sampleBuffer) else { return }

        // Detect input format on first buffer
        if !formatInitialized {
            if let fmt = CMSampleBufferGetFormatDescription(sampleBuffer), let asbdPtr = CMAudioFormatDescriptionGetStreamBasicDescription(fmt) {
                let asbd = asbdPtr.pointee
                inputChannels = Int(asbd.mChannelsPerFrame)
                inputIsFloat32 = (asbd.mFormatFlags & kAudioFormatFlagIsFloat) != 0
                inputIsNonInterleaved = (asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0
                let actualSR = Int(asbd.mSampleRate)
                ensureWriter(actualSampleRate: actualSR > 0 ? actualSR : sampleRate)
                formatInitialized = true

            } else {
                // Fallback if no format description
                inputChannels = max(1, channels)
                inputIsFloat32 = true
                inputIsNonInterleaved = false
                ensureWriter(actualSampleRate: sampleRate)
                formatInitialized = true

            }
        }

        rotateIfNeeded()

        // Prefer contiguous CMBlockBuffer when available (interleaved)
        if let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) {
            let length = CMBlockBufferGetDataLength(blockBuffer)
            if length > 0 {
                var raw = Data(count: length)
                let copied = raw.withUnsafeMutableBytes { (ptr: UnsafeMutableRawBufferPointer) -> OSStatus in
                    guard let base = ptr.baseAddress else { return -1 }
                    return CMBlockBufferCopyDataBytes(blockBuffer, atOffset: 0, dataLength: length, destination: base)
                }
                if copied == noErr {
                    let srcChannels = max(inputChannels, 1)
                    var pcm16 = Data()

                    if inputIsFloat32 {
                        let totalFloats = raw.count / MemoryLayout<Float32>.size
                        var frames = max(0, totalFloats / srcChannels)
                        if inputIsNonInterleaved {
                            // De-planarize: raw contains [ch0 frames][ch1 frames]...
                            let perChan = totalFloats / max(srcChannels,1)
                            frames = perChan

                            pcm16.reserveCapacity(frames * channels * 2)
                            raw.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
                                let f = ptr.bindMemory(to: Float32.self)
                                for i in 0..<frames {
                                    if channels == 1 {
                                        var acc: Float32 = 0
                                        for c in 0..<srcChannels { acc += f[c*perChan + i] }
                                        let avg = acc / Float32(srcChannels)
                                        let s = Int16(max(-1.0, min(1.0, avg)) * 32767.0)
                                        pcm16.append(contentsOf: withUnsafeBytes(of: s.littleEndian) { Data($0) })
                                    } else {
                                        for c in 0..<channels {
                                            let v = f[min(c, srcChannels-1)*perChan + i]
                                            let s = Int16(max(-1.0, min(1.0, v)) * 32767.0)
                                            pcm16.append(contentsOf: withUnsafeBytes(of: s.littleEndian) { Data($0) })
                                        }
                                    }
                                }
                            }
                        } else {
                            // Interleaved Float32 -> PCM16

                            pcm16.reserveCapacity(frames * channels * 2)
                            raw.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
                                let f = ptr.bindMemory(to: Float32.self)
                                for i in 0..<frames {
                                    if channels == 1 {
                                        var acc: Float32 = 0
                                        for c in 0..<srcChannels { acc += f[i*srcChannels + c] }
                                        let avg = acc / Float32(srcChannels)
                                        let s = Int16(max(-1.0, min(1.0, avg)) * 32767.0)
                                        pcm16.append(contentsOf: withUnsafeBytes(of: s.littleEndian) { Data($0) })
                                    } else {
                                        for c in 0..<channels {
                                            let v = f[i*srcChannels + min(c, srcChannels-1)]
                                            let s = Int16(max(-1.0, min(1.0, v)) * 32767.0)
                                            pcm16.append(contentsOf: withUnsafeBytes(of: s.littleEndian) { Data($0) })
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // Interleaved PCM16
                        if srcChannels == channels {
                            pcm16 = raw
                        } else {
                            let totalSamples = raw.count / 2
                            let frames = max(0, totalSamples / srcChannels)
                            pcm16.reserveCapacity(frames * channels * 2)
                            raw.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
                                let s = ptr.bindMemory(to: Int16.self)
                                for i in 0..<frames {
                                    if channels == 1 {
                                        var acc: Int = 0
                                        for c in 0..<srcChannels { acc += Int(s[i*srcChannels + c]) }
                                        let avg = Int16(acc / srcChannels)
                                        pcm16.append(contentsOf: withUnsafeBytes(of: avg.littleEndian) { Data($0) })
                                    } else {
                                        for c in 0..<channels {
                                            let v = s[i*srcChannels + min(c, srcChannels-1)]
                                            pcm16.append(contentsOf: withUnsafeBytes(of: v.littleEndian) { Data($0) })
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if !pcm16.isEmpty {
                        writer?.writePCM(pcm16)
                        let frameSize = channels * 2
                        if frameSize > 0 { samplesWritten += Int64(pcm16.count / frameSize) }
                    }
                }
            }
            return
        }

        // Fallback: Extract AudioBufferList when no contiguous block buffer is present
        var blockBuffer: CMBlockBuffer? = nil
        // Allocate space large enough for AudioBufferList with up to 8 buffers
        let ablCapacity = MemoryLayout<AudioBufferList>.size + (MemoryLayout<AudioBuffer>.size * 7)
        let ablPtr = UnsafeMutablePointer<AudioBufferList>.allocate(capacity: 1)
        defer { ablPtr.deallocate() }
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: nil,
            bufferListOut: ablPtr,
            bufferListSize: ablCapacity,
            blockBufferAllocator: kCFAllocatorDefault,
            blockBufferMemoryAllocator: kCFAllocatorDefault,
            flags: 0,
            blockBufferOut: &blockBuffer
        )
        if status != noErr { return }

        let abl = UnsafeMutableAudioBufferListPointer(ablPtr)
        // Gather channel data; if planar (one buffer per channel), we will interleave or mix
        var pcm16 = Data()
        if inputIsFloat32 {
            // Convert Float32 to PCM16, interleaving or mixing as needed
            // Determine number of frames from first buffer
            guard let firstBuf = abl.first else { return }
            let frames = Int(firstBuf.mDataByteSize) / MemoryLayout<Float32>.size

            if abl.count >= 2 {
                // Planar: one buffer per channel
                let srcChannels = abl.count
                let dstChannels = channels
                var interleaved = Data()
                interleaved.reserveCapacity(frames * dstChannels * 2)
                // Pointers to each channel's float samples
                var chanPtrs: [UnsafePointer<Float32>] = []
                for i in 0..<srcChannels {
                    let b = abl[i]
                    guard let base = b.mData else { continue }
                    chanPtrs.append(base.assumingMemoryBound(to: Float32.self))
                }
                for f in 0..<frames {
                    if dstChannels == 1 {
                        var acc: Float32 = 0
                        for c in 0..<srcChannels { acc += chanPtrs[c][f] }
                        let avg = acc / Float32(srcChannels)
                        let s = Int16(max(-1.0, min(1.0, avg)) * 32767.0)
                        interleaved.append(contentsOf: withUnsafeBytes(of: s.littleEndian) { Data($0) })
                    } else {
                        for c in 0..<dstChannels {
                            let v = chanPtrs[min(c, srcChannels - 1)][f]
                            let s = Int16(max(-1.0, min(1.0, v)) * 32767.0)
                            interleaved.append(contentsOf: withUnsafeBytes(of: s.littleEndian) { Data($0) })
                        }
                    }
                }
                pcm16 = interleaved
            } else {
                // Single buffer: interleaved or mono
                let b = abl[0]
                guard let base = b.mData else { return }
                let fptr = base.assumingMemoryBound(to: Float32.self)
                let srcChannels = max(inputChannels, 1)
                if srcChannels == 1 && channels == 1 {
                    var out = Data(); out.reserveCapacity(frames * 2)
                    for i in 0..<frames {
                        let s = Int16(max(-1.0, min(1.0, fptr[i])) * 32767.0)
                        out.append(contentsOf: withUnsafeBytes(of: s.littleEndian) { Data($0) })
                    }
                    pcm16 = out
                } else if srcChannels >= 2 && channels == 1 {
                    var out = Data(); out.reserveCapacity(frames * 2)
                    for i in 0..<frames {
                        var acc: Float32 = 0
                        for c in 0..<srcChannels { acc += fptr[i * srcChannels + c] }
                        let avg = acc / Float32(srcChannels)
                        let s = Int16(max(-1.0, min(1.0, avg)) * 32767.0)
                        out.append(contentsOf: withUnsafeBytes(of: s.littleEndian) { Data($0) })
                    }
                    pcm16 = out
                } else {
                    var out = Data(); out.reserveCapacity(frames * channels * 2)
                    for i in 0..<frames {
                        for c in 0..<channels {
                            let v = fptr[i * srcChannels + min(c, srcChannels - 1)]
                            let s = Int16(max(-1.0, min(1.0, v)) * 32767.0)
                            out.append(contentsOf: withUnsafeBytes(of: s.littleEndian) { Data($0) })
                        }
                    }
                    pcm16 = out
                }
            }
        } else {
            // Assume Int16 LPCM
            guard let firstBuf = abl.first else { return }
            let frames = Int(firstBuf.mDataByteSize) / 2

            if abl.count >= 2 {
                // Planar PCM16
                let srcChannels = abl.count
                var out = Data(); out.reserveCapacity(frames * channels * 2)
                var chanPtrs: [UnsafePointer<Int16>] = []
                for i in 0..<srcChannels {
                    let b = abl[i]
                    guard let base = b.mData else { continue }
                    chanPtrs.append(base.assumingMemoryBound(to: Int16.self))
                }
                for f in 0..<frames {
                    if channels == 1 {
                        var acc: Int = 0
                        for c in 0..<srcChannels { acc += Int(chanPtrs[c][f]) }
                        let avg = Int16(acc / srcChannels)
                        out.append(contentsOf: withUnsafeBytes(of: avg.littleEndian) { Data($0) })
                    } else {
                        for c in 0..<channels {
                            let v = chanPtrs[min(c, srcChannels - 1)][f]
                            out.append(contentsOf: withUnsafeBytes(of: v.littleEndian) { Data($0) })
                        }
                    }
                }
                pcm16 = out
            } else {
                // Interleaved or mono buffer
                let b = abl[0]
                guard let base = b.mData else { return }
                let sptr = base.assumingMemoryBound(to: Int16.self)
                let srcChannels = max(inputChannels, 1)
                var out = Data(); out.reserveCapacity(frames * channels * 2)
                if srcChannels == channels {
                    pcm16 = Data(bytes: sptr, count: frames * 2)
                } else if srcChannels >= 2 && channels == 1 {
                    for i in 0..<(frames / srcChannels) {
                        var acc: Int = 0
                        for c in 0..<srcChannels { acc += Int(sptr[i * srcChannels + c]) }
                        let avg = Int16(acc / srcChannels)
                        out.append(contentsOf: withUnsafeBytes(of: avg.littleEndian) { Data($0) })
                    }
                    pcm16 = out
                } else {
                    for i in 0..<(frames / srcChannels) {
                        for c in 0..<channels {
                            let v = sptr[i * srcChannels + min(c, srcChannels - 1)]
                            out.append(contentsOf: withUnsafeBytes(of: v.littleEndian) { Data($0) })
                        }
                    }
                    pcm16 = out
                }
            }
        }

        if !pcm16.isEmpty {
            writer?.writePCM(pcm16)
            let frameSize = channels * 2
            if frameSize > 0 { samplesWritten += Int64(pcm16.count / frameSize) }
        }
    }

    // Convert interleaved Float32 [-1,1] to interleaved Int16, with optional down/up-mix
    private func convertFloat32ToPCM16Interleaved(_ data: Data, srcChannels: Int, dstChannels: Int) -> Data {
        let count = data.count / MemoryLayout<Float32>.size
        var out = Data()
        out.reserveCapacity((count / max(srcChannels,1)) * max(dstChannels,1) * 2)
        data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
            let f = ptr.bindMemory(to: Float32.self)
            let frames = count / max(srcChannels, 1)
            for i in 0..<frames {
                if srcChannels == 1 && dstChannels == 1 {
                    let s = Int16(max(-1.0, min(1.0, f[i])) * 32767.0)
                    out.append(contentsOf: withUnsafeBytes(of: s.littleEndian) { Data($0) })
                } else if srcChannels >= 2 && dstChannels == 1 {
                    // Downmix average
                    var acc: Float32 = 0
                    for c in 0..<srcChannels { acc += f[i*srcChannels + c] }
                    let avg = acc / Float32(srcChannels)
                    let s = Int16(max(-1.0, min(1.0, avg)) * 32767.0)
                    out.append(contentsOf: withUnsafeBytes(of: s.littleEndian) { Data($0) })
                } else if srcChannels == 1 && dstChannels == 2 {
                    let v = max(-1.0, min(1.0, f[i]))
                    let s = Int16(v * 32767.0)
                    let le = withUnsafeBytes(of: s.littleEndian) { Data($0) }
                    out.append(le); out.append(le)
                } else {
                    // Equal channel count or more complex: copy first dstChannels
                    for c in 0..<dstChannels {
                        let v = max(-1.0, min(1.0, f[i*srcChannels + min(c, srcChannels-1)]))
                        let s = Int16(v * 32767.0)
                        out.append(contentsOf: withUnsafeBytes(of: s.littleEndian) { Data($0) })
                    }
                }
            }
        }
        return out
    }

    // Remap PCM16 interleaved channels
    private func remapPCM16(_ data: Data, srcChannels: Int, dstChannels: Int) -> Data {
        let frames = (data.count / 2) / max(srcChannels, 1)
        var out = Data()
        out.reserveCapacity(frames * max(dstChannels,1) * 2)
        data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
            let s = ptr.bindMemory(to: Int16.self)
            for i in 0..<frames {
                if srcChannels >= 2 && dstChannels == 1 {
                    var acc: Int = 0
                    for c in 0..<srcChannels { acc += Int(s[i*srcChannels + c]) }
                    let avg = Int16(acc / srcChannels)
                    out.append(contentsOf: withUnsafeBytes(of: avg.littleEndian) { Data($0) })
                } else if srcChannels == 1 && dstChannels == 2 {
                    let v = s[i]
                    let le = withUnsafeBytes(of: v.littleEndian) { Data($0) }
                    out.append(le); out.append(le)
                } else {
                    for c in 0..<dstChannels {
                        let v = s[i*srcChannels + min(c, srcChannels-1)]
                        out.append(contentsOf: withUnsafeBytes(of: v.littleEndian) { Data($0) })
                    }
                }
            }
        }
        return out
    }
}

// MARK: - Argument parsing

struct Args {
    var bundleId: String
    var outDir: URL
    var sampleRate: Int = 16_000
    var channels: Int = 1
    var stopKey: String = "q"
    var segmentSeconds: Int? = nil
}

func parseArgs() -> Args? {
    var args = CommandLine.arguments.dropFirst()
    func next() -> String? { args.isEmpty ? nil : args.removeFirst() }
    var result = Args(bundleId: "", outDir: URL(fileURLWithPath: FileManager.default.currentDirectoryPath))

    while let a = next() {
        switch a {
        case "--bundle-id": result.bundleId = next() ?? ""
        case "--out-dir": if let p = next() { result.outDir = URL(fileURLWithPath: p) }
        case "--samplerate": if let v = next(), let i = Int(v) { result.sampleRate = i }
        case "--channels": if let v = next(), let i = Int(v) { result.channels = i }
        case "--stop-key": if let v = next() { result.stopKey = v }
        case "--segment-seconds": if let v = next(), let i = Int(v) { result.segmentSeconds = i }
        default: break
        }
    }
    guard !result.bundleId.isEmpty else { return nil }
    return result
}

// MARK: - Main (async)

@main
struct Runner {
    static func main() async {
        guard let args = parseArgs() else {
            fputs("Usage: sckit-capture --bundle-id BUNDLE --out-dir DIR [--samplerate 16000] [--channels 1] [--stop-key q] [--segment-seconds 3600]\n", stderr)
            exit(2)
        }

        do {
            let content = try await SCShareableContent.current
            guard let app = content.applications.first(where: { $0.bundleIdentifier == args.bundleId }) else {
                fputs("Application not found or not running: \(args.bundleId)\n", stderr)
                exit(1)
            }
            guard let display = content.displays.first else {
                fputs("No displays found for content filter\n", stderr)
                exit(1)
            }

            let cfg = SCStreamConfiguration()
            cfg.capturesAudio = true
            // Force 48k to avoid pitch shifts; we'll resample later if needed
            cfg.sampleRate = 48_000

            let filter = SCContentFilter(display: display, including: [app], exceptingWindows: [])
            let stream = SCStream(filter: filter, configuration: cfg, delegate: nil)
            // Align sink's expected sample rate with SC configuration (48k)
            let sink = AudioSink(outDir: args.outDir, sampleRate: 48_000, channels: args.channels, segmentSeconds: args.segmentSeconds)

            do {
                try stream.addStreamOutput(sink, type: SCStreamOutputType.audio, sampleHandlerQueue: DispatchQueue.main)
            } catch {
                fputs("Failed to add stream output: \(error)\n", stderr)
                exit(1)
            }

            do {
                try await stream.startCapture()
            } catch {
                fputs("Failed to start capture: \(error)\n", stderr)
                exit(1)
            }

            // Stop when stopKey is received on stdin (async, no blocking wait in async context)
            let stopKeyLower = args.stopKey.lowercased()
            await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                DispatchQueue.global().async {
                    while let line = readLine() {
                        if line.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == stopKeyLower {
                            break
                        }
                    }
                    cont.resume()
                }
            }

            do {
                try await stream.stopCapture()
            } catch {
                fputs("Failed to stop capture: \(error)\n", stderr)
                exit(1)
            }
            // Ensure final WAV header is patched
            sink.closeWriter()
            exit(0)
        } catch {
            fputs("Error initializing capture: \(error)\n", stderr)
            exit(1)
        }
    }
}


