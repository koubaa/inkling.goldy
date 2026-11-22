//! Packed Mini weights: synthetic CI tensors, and real rows when
//! `INKLING_REAL_PACK` points at a directory written by
//! `tools/check_packed_parity.py`.

#![cfg(feature = "vulkan")]

use ammon::checkpoint::MappedCheckpoint;
use ammon::gpu::create_runtime;
use ammon::Linear;
use goldy::{Nvfp4Experts, Runtime, Scheme, Tensor, TensorDType, TensorKernels, TensorShape};
use inkling_goldy::packed::{load_nvfp4, ExpertProj};
use safetensors::tensor::{serialize, TensorView};
use safetensors::Dtype;

fn runtime() -> Runtime {
    create_runtime().expect("goldy runtime")
}

fn read_f32(scheme: &mut Scheme, buf: &goldy::Buffer) -> Vec<f32> {
    let mut sub = scheme.submit().expect("submit");
    let bytes = (&mut sub >> buf).take::<u8>().expect("host take");
    bytemuck::cast_slice(&bytes).to_vec()
}

fn write_file(name: &str, views: Vec<(&str, TensorView<'_>)>) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(name);
    std::fs::write(&path, serialize(views, None).unwrap()).unwrap();
    path
}

#[test]
fn interleaved_w13_matches_concatenated_dequant() {
    let experts = 1usize;
    let rows = 4usize;
    let k = 16usize;
    let mut stored = vec![0u8; experts * rows * (k / 2)];
    // Stored row 0 is gate 0. Nibble 2 is E2M1 1.0, scale 1, factor 1.
    stored[0] = 0x02;
    let scales = vec![0x38u8; rows];
    let amax = [1.0f32];
    let denom = [1.0f32];
    let path = write_file(
        "inkling-w13-interleaved.safetensors",
        vec![
            (
                "model.llm.layers.0.mlp.experts.w13_weight",
                TensorView::new(Dtype::U8, vec![experts, rows, k / 2], &stored).unwrap(),
            ),
            (
                "model.llm.layers.0.mlp.experts.w13_weight_scale",
                TensorView::new(Dtype::F8_E4M3, vec![experts, rows, 1], &scales).unwrap(),
            ),
            (
                "model.llm.layers.0.mlp.experts.w13_weight_amax",
                TensorView::new(Dtype::F32, vec![experts], bytemuck::cast_slice(&amax)).unwrap(),
            ),
            (
                "model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator",
                TensorView::new(Dtype::F32, vec![1], bytemuck::cast_slice(&denom)).unwrap(),
            ),
        ],
    );
    let mapped = MappedCheckpoint::open(&path).unwrap();
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let weight = load_nvfp4(&mapped, &device, 0, ExpertProj::W13).unwrap();
    let mut x_host = vec![0.0f32; k];
    x_host[0] = 1.0;
    let x = Tensor::from_f32(&device, TensorShape::vector(k as u32), &x_host).unwrap();
    let ids = Tensor::from_u32(&device, TensorShape::vector(1), &[0]).unwrap();
    let out = Tensor::zeros(&device, TensorShape::vector(rows as u32), TensorDType::F32).unwrap();
    let kernels = TensorKernels::new(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernels
        .recorder(&mut scheme)
        .matmul_nvfp4_indexed_into(
            "w13",
            Nvfp4Experts {
                codes: weight.codes.view(),
                scales: weight.scales.view(),
                factors: weight.factors.view(),
                ids: ids.view(),
                slot: 0,
            },
            x.view(),
            out.view(),
        )
        .unwrap();
    let got = read_f32(&mut scheme, out.buffer());
    // After Interleave(dim=1), gate 0 is row 0 and the other three rows are zero.
    assert!((got[0] - 1.0).abs() < 1e-5, "{got:?}");
    assert!(got[1..].iter().all(|v| v.abs() < 1e-6), "{got:?}");
    let _ = std::fs::remove_file(path);
}

fn pattern(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect()
}

fn f32_file(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn max_abs(got: &[f32], expect: &[f32]) -> f32 {
    got.iter()
        .zip(expect)
        .map(|(g, e)| (g - e).abs())
        .fold(0.0, f32::max)
}

#[test]
fn real_mini_layers_when_present() {
    let Some(dir) = std::env::var_os("INKLING_REAL_PACK") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let device = runtime();
    let ctx = device.create_context().unwrap();
    let kernels = TensorKernels::new(&device).unwrap();
    let x_host = pattern(2304);
    let x = Tensor::from_f32(&device, TensorShape::vector(2304), &x_host).unwrap();
    let ids = Tensor::from_u32(&device, TensorShape::vector(1), &[0]).unwrap();

    for layer in [0u32, 10, 19] {
        let layer_dir = dir.join(format!("l{layer}"));
        let codes = std::fs::read(layer_dir.join("codes.bin")).unwrap();
        let scales = std::fs::read(layer_dir.join("scales.bin")).unwrap();
        let amax = std::fs::read(layer_dir.join("amax.bin")).unwrap();
        let denom = std::fs::read(layer_dir.join("denom.bin")).unwrap();
        let expect = f32_file(&layer_dir.join("expected.bin"));
        let names = [
            format!("model.llm.layers.{layer}.mlp.experts.w13_weight"),
            format!("model.llm.layers.{layer}.mlp.experts.w13_weight_scale"),
            format!("model.llm.layers.{layer}.mlp.experts.w13_weight_amax"),
            format!("model.llm.layers.{layer}.mlp.experts.w13_weight_outer_scale_denominator"),
        ];
        let views = vec![
            TensorView::new(Dtype::U8, vec![1, 2304, 1152], &codes).unwrap(),
            TensorView::new(Dtype::F8_E4M3, vec![1, 2304, 144], &scales).unwrap(),
            TensorView::new(Dtype::F32, vec![1], &amax).unwrap(),
            TensorView::new(Dtype::F32, vec![1], &denom).unwrap(),
        ];
        let owned: Vec<(&str, TensorView<'_>)> = names
            .iter()
            .zip(views)
            .map(|(name, view)| (name.as_str(), view))
            .collect();
        let path = write_file(&format!("inkling-real-l{layer}.safetensors"), owned);
        let mapped = MappedCheckpoint::open(&path).unwrap();
        let weight = load_nvfp4(&mapped, &device, layer, ExpertProj::W13).unwrap();
        let out = Tensor::zeros(&device, TensorShape::vector(2304), TensorDType::F32).unwrap();
        let mut scheme = Scheme::new(&ctx);
        kernels
            .recorder(&mut scheme)
            .matmul_nvfp4_indexed_into(
                "real_w13",
                Nvfp4Experts {
                    codes: weight.codes.view(),
                    scales: weight.scales.view(),
                    factors: weight.factors.view(),
                    ids: ids.view(),
                    slot: 0,
                },
                x.view(),
                out.view(),
            )
            .unwrap();
        let got = read_f32(&mut scheme, out.buffer());
        let err = max_abs(&got, &expect);
        eprintln!("layer {layer} expert0 w13 max abs {err}");
        assert!(err < 1e-3, "layer {layer} max abs {err}");
        let _ = std::fs::remove_file(path);
    }

    let wq = std::fs::read(dir.join("wq_rows.bin")).unwrap();
    let expect = f32_file(&dir.join("wq_expected.bin"));
    let path = write_file(
        "inkling-real-wq.safetensors",
        vec![(
            "model.llm.layers.0.attn.wq_du.weight",
            TensorView::new(Dtype::BF16, vec![4, 2304], &wq).unwrap(),
        )],
    );
    let mapped = MappedCheckpoint::open(&path).unwrap();
    let weight = mapped
        .upload(&device, "model.llm.layers.0.attn.wq_du.weight")
        .unwrap();
    assert_eq!(weight.dtype(), TensorDType::BF16);
    let out = Tensor::zeros(&device, TensorShape::vector(4), TensorDType::F32).unwrap();
    let linear = Linear::new(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    linear
        .record_bf16(&mut scheme, weight.view(), x.view(), out.view())
        .unwrap();
    let got = read_f32(&mut scheme, out.buffer());
    let err = max_abs(&got, &expect);
    eprintln!("layer 0 wq first 4 rows max abs {err}");
    assert!(err < 1e-2, "wq max abs {err}");
    let _ = std::fs::remove_file(path);
}
