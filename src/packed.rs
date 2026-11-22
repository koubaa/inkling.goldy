//! Packed Inkling-2-Mini checkpoint tensors.
//!
//! The published shard stores expert `w13` interleaved `(gate, up, ...)` and
//! expert `w2` already in matmul order. NVFP4 layers (0–19) stay U8 codes plus
//! F8_E4M3 scales. This loader memory-maps the file, permutes `w13` rows into
//! concatenated `[gate | up]`, and uploads the packed bytes. It does not expand
//! them to FP32. Layers 20–23 are MXFP8 and are rejected.

use anyhow::{bail, Result};
use goldy::{Runtime, Tensor, TensorDType, TensorShape};

use ammon::checkpoint::MappedCheckpoint;

/// Which expert projection a checkpoint tensor is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertProj {
    /// Fused gate and up. Stored interleaved; uploaded as `[gate | up]`.
    W13,
    /// Down projection. No interleave.
    W2,
}

impl ExpertProj {
    fn name(self) -> &'static str {
        match self {
            Self::W13 => "w13",
            Self::W2 => "w2",
        }
    }

    fn deinterleave(self) -> bool {
        matches!(self, Self::W13)
    }
}

/// Device-side NVFP4 experts. Codes and scales stay packed.
pub struct PackedExpertWeight {
    pub codes: Tensor,
    pub scales: Tensor,
    pub factors: Tensor,
}

/// `model.llm.layers.{layer}.mlp.experts.{proj}_weight{suffix}`.
pub fn expert_name(layer: u32, proj: ExpertProj, suffix: &str) -> String {
    format!(
        "model.llm.layers.{layer}.mlp.experts.{}_weight{suffix}",
        proj.name()
    )
}

/// Turn stored `(gate, up, gate, up, ...)` rows into concatenated `[gate | up]`.
///
/// This is `Interleave(dim=1)` from transformers `core_model_loading.py`: reshape
/// `[experts, rows/2, 2, row]`, transpose the pair axis, and reshape back.
///
/// The permute stays on the host, over the packed bytes. Goldy's elementwise
/// kernels index `src[i]` densely and do not honor a stride, so a stride-2
/// view of the stored pairs would read the wrong bytes. A strided elementwise
/// op would be a new kernel, not a view.
pub fn deinterleave_dim1(
    src: &[u8],
    experts: usize,
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<u8>> {
    if rows % 2 != 0 {
        bail!("interleaved rows {rows} are not even");
    }
    let need = experts
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(row_bytes))
        .ok_or_else(|| anyhow::anyhow!("interleave size overflow"))?;
    if src.len() != need {
        bail!("interleave byte length {} != {need}", src.len());
    }
    let half = rows / 2;
    let mut out = vec![0u8; src.len()];
    for expert in 0..experts {
        for pair in 0..half {
            let base = expert * rows;
            copy_row(src, &mut out, base + 2 * pair, base + pair, row_bytes);
            copy_row(
                src,
                &mut out,
                base + 2 * pair + 1,
                base + half + pair,
                row_bytes,
            );
        }
    }
    Ok(out)
}

fn copy_row(src: &[u8], dst: &mut [u8], src_row: usize, dst_row: usize, row_bytes: usize) {
    let from = src_row * row_bytes;
    let to = dst_row * row_bytes;
    dst[to..to + row_bytes].copy_from_slice(&src[from..from + row_bytes]);
}

/// Load one NVFP4 expert tensor. `w13` is deinterleaved. Factors are
/// `amax / outer_scale_denominator` (a few FP32 values, not the weight).
pub fn load_nvfp4(
    checkpoint: &MappedCheckpoint,
    runtime: &Runtime,
    layer: u32,
    proj: ExpertProj,
) -> Result<PackedExpertWeight> {
    let codes_name = expert_name(layer, proj, "");
    let scale_name = expert_name(layer, proj, "_scale");
    let amax_name = expert_name(layer, proj, "_amax");
    let denom_name = expert_name(layer, proj, "_outer_scale_denominator");
    let codes = checkpoint.get(&codes_name)?;
    let scales = checkpoint.get(&scale_name)?;
    if codes.dtype != "U8" || scales.dtype != "F8_E4M3" {
        bail!(
            "`{codes_name}` is {} / {} scales, not NVFP4 U8 + F8_E4M3 (layers 20–23 are MXFP8)",
            codes.dtype,
            scales.dtype
        );
    }
    if codes.shape.len() != 3 || scales.shape.len() != 3 {
        bail!("`{codes_name}` and its scale must be rank 3");
    }
    let experts = codes.shape[0];
    let rows = codes.shape[1];
    let packed_k = codes.shape[2];
    if scales.shape[0] != experts || scales.shape[1] != rows || packed_k % 8 != 0 {
        bail!("`{scale_name}` does not match `{codes_name}`");
    }
    if scales.shape[2] * 16 != packed_k * 2 {
        bail!("`{scale_name}` is not one scale per 16 unpacked columns");
    }
    let code_bytes = checkpoint.bytes(codes);
    let scale_bytes = checkpoint.bytes(scales);
    let (codes_host, scales_host) = if proj.deinterleave() {
        (
            deinterleave_dim1(code_bytes, experts, rows, packed_k)?,
            deinterleave_dim1(scale_bytes, experts, rows, scales.shape[2])?,
        )
    } else {
        (code_bytes.to_vec(), scale_bytes.to_vec())
    };
    let factors = expert_factors(checkpoint, &amax_name, &denom_name, experts)?;
    let code_shape = shape3(experts, rows, packed_k)?;
    let scale_shape = shape3(experts, rows, scales.shape[2])?;
    Ok(PackedExpertWeight {
        codes: Tensor::from_bytes(runtime, code_shape, TensorDType::U8, &codes_host)?,
        scales: Tensor::from_bytes(runtime, scale_shape, TensorDType::F8E4M3, &scales_host)?,
        factors: Tensor::from_f32(runtime, TensorShape::vector(experts as u32), &factors)?,
    })
}

fn shape3(a: usize, b: usize, c: usize) -> Result<TensorShape> {
    let dims = [a, b, c]
        .into_iter()
        .map(|dim| u32::try_from(dim).map_err(|_| anyhow::anyhow!("shape does not fit u32")))
        .collect::<Result<Vec<_>>>()?;
    Ok(TensorShape::from_dims(&dims)?)
}

fn expert_factors(
    checkpoint: &MappedCheckpoint,
    amax_name: &str,
    denom_name: &str,
    experts: usize,
) -> Result<Vec<f32>> {
    let amax = f32_slice(checkpoint, amax_name)?;
    let denom = f32_slice(checkpoint, denom_name)?;
    if amax.len() != experts {
        bail!(
            "`{amax_name}` has {} values, expected {experts}",
            amax.len()
        );
    }
    if denom.len() != 1 && denom.len() != experts {
        bail!("`{denom_name}` must be one value or one per expert");
    }
    Ok(amax
        .iter()
        .enumerate()
        .map(|(i, value)| value / denom[if denom.len() == 1 { 0 } else { i }])
        .collect())
}

fn f32_slice(checkpoint: &MappedCheckpoint, name: &str) -> Result<Vec<f32>> {
    let tensor = checkpoint.get(name)?;
    if tensor.dtype != "F32" {
        bail!("`{name}` is {}, expected F32", tensor.dtype);
    }
    let bytes = checkpoint.bytes(tensor);
    if bytes.len() % 4 != 0 {
        bail!("`{name}` is not a whole number of f32 values");
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deinterleave_pairs_gates_then_ups() {
        let stored = [1u8, 2, 3, 4];
        let got = deinterleave_dim1(&stored, 1, 4, 1).unwrap();
        assert_eq!(got, vec![1, 3, 2, 4]);
    }
}
