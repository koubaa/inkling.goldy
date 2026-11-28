fn main() -> anyhow::Result<()> {
    #[cfg(not(feature = "gpu"))]
    {
        anyhow::bail!("build with --features vulkan, cuda, or metal");
    }

    #[cfg(feature = "gpu")]
    {
        run()
    }
}

#[cfg(feature = "gpu")]
fn run() -> anyhow::Result<()> {
    use ammon::HfTokenizer;
    use anyhow::{bail, Context};
    use smollm2_goldy::generate::generate;
    use smollm2_goldy::model::{Model, DEFAULT_SEQ_LEN};
    use std::path::PathBuf;

    fn print_usage() {
        eprintln!(
            "Usage: smollm2-goldy [prompt] [--assets DIR] [-n TOKENS] [--seq POSITIONS]\n\
             Defaults: assets/ beside the crate, 32 tokens, {DEFAULT_SEQ_LEN} positions,\n\
             prompt \"The capital of France is\""
        );
    }

    let mut assets = smollm2_goldy::assets_dir();
    let mut prompt = "The capital of France is".to_string();
    let mut n_tokens: u32 = 32;
    let mut seq_len = DEFAULT_SEQ_LEN;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            "--assets" => assets = PathBuf::from(args.next().context("missing --assets path")?),
            "-n" | "--n" => n_tokens = args.next().context("missing -n value")?.parse()?,
            "--seq" => seq_len = args.next().context("missing --seq value")?.parse()?,
            flag if flag.starts_with('-') => bail!("unknown flag {flag}"),
            other => prompt = other.to_string(),
        }
    }

    eprintln!("loading {}", assets.display());
    let tokenizer = HfTokenizer::from_file(assets.join("tokenizer.json"))?;
    let mut model = Model::load_dir(&assets, seq_len)?;
    let c = model.config;
    eprintln!(
        "hidden={} layers={} heads={}/{} kv head_dim={} vocab={} rope_theta={} seq={seq_len}",
        c.hidden, c.layers, c.q_heads, c.kv_heads, c.head_dim, c.vocab, c.rope_theta
    );
    let out = generate(&mut model, &tokenizer, &prompt, n_tokens)?;
    println!("{prompt}{}", out.text);
    eprintln!(
        "{} prompt tokens, {} generated, {:.1} tok/s, worker records={}",
        out.prompt_ids.len(),
        out.tokens.len(),
        out.tokens_per_second,
        model.replay_stats().records
    );
    Ok(())
}
