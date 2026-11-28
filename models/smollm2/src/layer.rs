//! One decoder layer's weights on the packed slabs. Every layer records through
//! the model's shared [`ammon::CausalAttentionBlock`] and [`ammon::SwiGluBlock`];
//! a layer is only where its weights live.

use crate::weights::WeightSlice;

#[derive(Clone, Debug)]
pub struct LayerSpec {
    pub attn_norm: WeightSlice,
    /// Rows interleaved per head for adjacent-pair RoPE.
    pub query: WeightSlice,
    /// Rows interleaved per head for adjacent-pair RoPE.
    pub key: WeightSlice,
    pub value: WeightSlice,
    pub output: WeightSlice,
    pub mlp_norm: WeightSlice,
    pub gate: WeightSlice,
    pub up: WeightSlice,
    pub down: WeightSlice,
}

#[cfg(feature = "gpu")]
impl LayerSpec {
    pub fn attention<'a>(
        &self,
        slabs: &'a [goldy::Tensor],
    ) -> anyhow::Result<ammon::CausalAttentionBlockWeights<'a>> {
        Ok(ammon::CausalAttentionBlockWeights {
            norm: self.attn_norm.view(slabs)?,
            query: self.query.view(slabs)?,
            key: self.key.view(slabs)?,
            value: self.value.view(slabs)?,
            output: self.output.view(slabs)?,
        })
    }

    pub fn mlp<'a>(
        &self,
        slabs: &'a [goldy::Tensor],
    ) -> anyhow::Result<ammon::SwiGluBlockWeights<'a>> {
        Ok(ammon::SwiGluBlockWeights {
            norm: self.mlp_norm.view(slabs)?,
            gate: self.gate.view(slabs)?,
            up: self.up.view(slabs)?,
            down: self.down.view(slabs)?,
        })
    }
}
