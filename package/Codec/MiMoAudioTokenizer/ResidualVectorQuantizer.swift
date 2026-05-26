// Copyright © 2025 Xiaomi LLM-Core-Team (original model architecture)
// Copyright © 2025 ailuntx (Python MLX port)
// Copyright © OpenTTS contributors (Swift port)
// License: licenses/mimo.txt (Apache-2.0) + licenses/mlx-audio.txt (MIT)

import Foundation
import MLX
import MLXNN

// Encode-only Residual Vector Quantizer for the MiMo audio tokenizer.
// Port of `mlx_audio/stt/models/mimo_v2_asr/quantization.py`. Training-time
// state (EMA, kmeans init, commitment loss) is omitted; only the inference
// `encode` path is implemented. All types are MiMo-prefixed to avoid
// collisions with Marvis/Mimi's `EuclideanCodebook`, `VectorQuantization`,
// and `ResidualVectorQuantizer` types.

/// Single codebook with Euclidean nearest-neighbour lookup.
class MiMoEuclideanCodebook: Module {
    @ParameterInfo(key: "embed") var embed: MLXArray

    init(dim: Int, codebookSize: Int) {
        _embed.wrappedValue = MLXArray.zeros([codebookSize, dim])
    }

    /// Find the nearest codebook index for each row of `x`.
    func quantize(_ x: MLXArray) -> MLXArray {
        let embedT = embed.T
        let dot = MLX.matmul(x, embedT)
        let eSq = (embed * embed).sum(axis: -1)
        let negDist = 2.0 * dot - eSq
        return MLX.argMax(negDist, axis: -1).asType(.int32)
    }

    func decode(_ ids: MLXArray) -> MLXArray { embed[ids] }
}

/// Identity placeholder used when codebook_dim == dim.
private class MiMoIdentityLayer: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray { x }
}

/// Single VQ layer wrapping a codebook with optional in/out projections.
class MiMoVectorQuantization: Module {
    @ModuleInfo(key: "project_in") var projectIn: UnaryLayer
    @ModuleInfo(key: "project_out") var projectOut: UnaryLayer
    @ModuleInfo(key: "codebook") var codebook: MiMoEuclideanCodebook

    init(dim: Int, codebookSize: Int, codebookDim: Int? = nil) {
        let cbDim = codebookDim ?? dim
        if cbDim != dim {
            _projectIn.wrappedValue = Linear(dim, cbDim, bias: false)
            _projectOut.wrappedValue = Linear(cbDim, dim, bias: false)
        } else {
            _projectIn.wrappedValue = MiMoIdentityLayer()
            _projectOut.wrappedValue = MiMoIdentityLayer()
        }
        _codebook.wrappedValue = MiMoEuclideanCodebook(dim: cbDim, codebookSize: codebookSize)
    }

    func encode(_ x: MLXArray) -> MLXArray { codebook.quantize(projectIn(x)) }
    func decode(_ ids: MLXArray) -> MLXArray { projectOut(codebook.decode(ids)) }
}

/// Stack of VQ layers applied residually.
class MiMoResidualVectorQuantization: Module {
    @ModuleInfo var layers: [MiMoVectorQuantization]

    init(numQuantizers: Int, codebookSize: Int, dim: Int) {
        _layers.wrappedValue = (0 ..< numQuantizers).map { _ in
            MiMoVectorQuantization(dim: dim, codebookSize: codebookSize)
        }
    }

    /// Iteratively quantize residuals across `nQ` layers.
    /// - Returns: stacked int32 indices of shape `(nQ, N)`.
    func encode(_ x: MLXArray, nQ: Int? = nil, start: Int = 0) -> MLXArray {
        let effective = nQ ?? layers.count
        var residual = x
        var indicesList: [MLXArray] = []
        for layer in layers[start ..< effective] {
            let idx = layer.encode(residual)
            let quant = layer.decode(idx)
            residual = residual - quant
            indicesList.append(idx)
        }
        return MLX.stacked(indicesList, axis: 0)
    }
}

/// Top-level RVQ module matching safetensors layout `vq.layers.{i}.codebook.embed`.
class MiMoResidualVectorQuantizer: Module {
    @ModuleInfo var vq: MiMoResidualVectorQuantization
    let nQ: Int
    let dimension: Int

    init(dimension: Int = 1280, nQ: Int = 20, codebookSize: Int = 1024) {
        self.nQ = nQ
        self.dimension = dimension
        _vq.wrappedValue = MiMoResidualVectorQuantization(
            numQuantizers: nQ,
            codebookSize: codebookSize,
            dim: dimension
        )
    }

    func encode(_ x: MLXArray, nQ: Int? = nil, start: Int = 0) -> MLXArray {
        vq.encode(x, nQ: nQ, start: start)
    }
}
