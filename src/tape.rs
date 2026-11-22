//! The only owner of Inkling's depth tape and the loop schedule.
//!
//! Mini is 24 layers, block size 6, layers 6..18 looped 3 times: 48 visits.
//! A boundary layer's input is appended once, after that layer's attention mix
//! and before its MLP mix. Later layers see that row. The layer does not keep
//! a tape of its own.

/// Loop grid. `mini` matches the published text config.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopSchedule {
    pub layers: u32,
    pub block: u32,
    pub loop_start: u32,
    pub loop_end: u32,
    pub passes: u32,
}

impl LoopSchedule {
    pub fn mini() -> Self {
        Self {
            layers: 24,
            block: 6,
            loop_start: 6,
            loop_end: 18,
            passes: 3,
        }
    }

    pub fn n_loop(self) -> u32 {
        self.loop_end - self.loop_start
    }

    /// Pre-loop layers, then each pass of the loop, then post-loop layers.
    pub fn visits(self) -> u32 {
        let pre = self.loop_start;
        let post = self.layers - self.loop_end;
        pre + self.passes * self.n_loop() + post
    }

    pub fn is_boundary(self, layer: u32) -> bool {
        layer % self.block == 0
    }

    pub fn is_full(self, layer: u32) -> bool {
        (layer + 1) % self.block == 0
    }

    /// Cache slot for this physical layer on this pass. Pass 0 uses `layer`.
    /// Later passes use the offset from the reference forward.
    pub fn cache_slot(self, layer: u32, pass: u32) -> u32 {
        if pass == 0 {
            layer
        } else {
            self.layers + (pass - 1) * self.n_loop() - self.loop_start + layer
        }
    }
}

/// First block-row index a later pass uses as a message key.
///
/// The reference drops every pre-loop block and one extra block (the hidden
/// state that entered the loop). The previous pass's exit hidden state is a
/// separate partial, not a row of this slice.
pub fn message_key_start(schedule: LoopSchedule) -> u32 {
    (0..schedule.loop_start)
        .filter(|layer| schedule.is_boundary(*layer))
        .count() as u32
        + 1
}

/// How many block rows are visible at a visit, before that visit's own append.
pub fn blocks_before(schedule: LoopSchedule, layer: u32, pass: u32) -> u32 {
    let mut n = 0u32;
    for earlier in 0..layer {
        if schedule.is_boundary(earlier) {
            n += 1;
        }
    }
    if pass > 0 && layer >= schedule.loop_start && layer < schedule.loop_end {
        // Later passes restart from the pre-loop snapshot, then append only
        // the boundaries already visited in this pass.
        n = (0..schedule.loop_start)
            .filter(|i| schedule.is_boundary(*i))
            .count() as u32;
        for earlier in schedule.loop_start..layer {
            if schedule.is_boundary(earlier) {
                n += 1;
            }
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mini_has_forty_eight_visits_and_the_reference_cache_offset() {
        let mini = LoopSchedule::mini();
        assert_eq!(mini.visits(), 48);
        assert!(mini.is_boundary(0) && mini.is_boundary(6) && !mini.is_boundary(5));
        assert!(mini.is_full(5) && mini.is_full(11) && !mini.is_full(6));
        assert_eq!(mini.cache_slot(6, 0), 6);
        assert_eq!(mini.cache_slot(6, 1), 24);
        assert_eq!(mini.cache_slot(17, 2), 24 + 12 + 11);
        assert_eq!(blocks_before(mini, 6, 0), 1);
        assert_eq!(blocks_before(mini, 12, 1), 2);
        // Mini's pre-loop contributes the layer-0 block. The slice starts one
        // past that and past the first loop boundary, so layer 12 survives.
        assert_eq!(message_key_start(mini), 2);
    }
}

/// Device block tape. The decoder layer writes a boundary row into
/// [`Self::blocks`] at [`Self::len`]; [`Self::commit_written_row`] counts it
/// and does not copy it. Snapshot and restore are how a later loop pass
/// returns to the pre-loop prefix.
#[cfg(feature = "vulkan")]
pub struct DepthTape {
    blocks: goldy::Tensor,
    len: u32,
    capacity: u32,
    snapshot: Option<u32>,
}

#[cfg(feature = "vulkan")]
impl DepthTape {
    pub fn new(runtime: &goldy::Runtime, capacity: u32, hidden: u32) -> anyhow::Result<Self> {
        let blocks = goldy::Tensor::zeros(
            runtime,
            goldy::TensorShape::from_dims(&[capacity, hidden])?,
            goldy::TensorDType::F32,
        )?;
        Ok(Self {
            blocks,
            len: 0,
            capacity,
            snapshot: None,
        })
    }

    pub fn len(&self) -> u32 {
        self.len
    }

    pub fn blocks(&self) -> goldy::TensorView<'_> {
        self.blocks.view()
    }

    /// Remember the current prefix. A later pass restores this length.
    pub fn snapshot_prefix(&mut self) {
        self.snapshot = Some(self.len);
    }

    /// Count the row the layer wrote at `len`. No second copy is stored.
    pub fn commit_written_row(&mut self) -> anyhow::Result<()> {
        if self.len >= self.capacity {
            anyhow::bail!("depth tape is full ({})", self.capacity);
        }
        self.len += 1;
        Ok(())
    }

    /// Drop rows committed after [`Self::snapshot_prefix`]. The bytes stay;
    /// readers stop at [`Self::len`].
    pub fn restore_prefix(&mut self) -> anyhow::Result<()> {
        self.len = self
            .snapshot
            .ok_or_else(|| anyhow::anyhow!("depth tape has no prefix snapshot"))?;
        Ok(())
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod owner {
    use super::*;

    #[test]
    fn commit_counts_the_layer_row_without_copying_another() {
        let device = ammon::gpu::create_runtime().expect("runtime");
        let mut tape = DepthTape::new(&device, 4, 2).unwrap();
        tape.snapshot_prefix();
        tape.commit_written_row().unwrap();
        tape.commit_written_row().unwrap();
        assert_eq!(tape.len(), 2);
        tape.restore_prefix().unwrap();
        assert_eq!(tape.len(), 0);
        tape.commit_written_row().unwrap();
        assert_eq!(tape.len(), 1);
        assert_eq!(tape.blocks().shape().dims(), [4, 2]);
    }
}
