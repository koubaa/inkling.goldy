fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("generate") => generate(args),
        Some(other) => {
            eprintln!("unknown command {other}. Try `generate`.");
            std::process::exit(2);
        }
        None => {
            let mini = inkling_goldy::tape::LoopSchedule::mini();
            println!(
                "inkling-goldy: Inkling 2 on Ammon. Mini schedule is {} visits. \
                 `inkling-goldy generate --weights <safetensors> --header <mini_forward.bin> --prompt 30,13 --max-new 2`",
                mini.visits()
            );
        }
    }
}

fn generate(mut args: impl Iterator<Item = String>) {
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = &mut args;
        eprintln!("generate needs `cargo run --features vulkan -- generate ...`");
        std::process::exit(1);
    }
    #[cfg(feature = "vulkan")]
    {
        let mut weights = None;
        let mut header = None;
        let mut config = None;
        let mut prompt = None;
        let mut max_new = 2u32;
        let mut tokenizer = None;
        let mut text = None;
        while let Some(arg) = args.next() {
            let value = args.next();
            match arg.as_str() {
                "--weights" => weights = value,
                "--header" => header = value,
                "--config" => config = value,
                "--prompt" => prompt = value,
                "--text" => text = value,
                "--tokenizer" => tokenizer = value,
                "--max-new" => {
                    let Some(raw) = value else {
                        eprintln!("--max-new needs a number");
                        std::process::exit(2);
                    };
                    max_new = raw.parse().unwrap_or_else(|_| {
                        eprintln!("--max-new {raw} is not a whole number");
                        std::process::exit(2);
                    });
                }
                other => {
                    eprintln!("unknown argument {other}");
                    std::process::exit(2);
                }
            }
        }
        let weights = weights.expect("--weights");
        let file = std::fs::File::open(&weights).unwrap_or_else(|err| {
            eprintln!("open {weights}: {err}");
            std::process::exit(1);
        });
        let mapped = unsafe { memmap2::Mmap::map(&file) }.unwrap_or_else(|err| {
            eprintln!("map {weights}: {err}");
            std::process::exit(1);
        });
        let (spec, tokens) = if let Some(config) = config {
            let text = std::fs::read_to_string(&config).unwrap_or_else(|err| {
                eprintln!("read {config}: {err}");
                std::process::exit(1);
            });
            let spec = inkling_goldy::model::spec_from_config(&text).unwrap_or_else(|err| {
                eprintln!("config: {err}");
                std::process::exit(1);
            });
            (spec, Vec::new())
        } else if let Some(header) = header {
            let header_bytes = std::fs::read(&header).unwrap_or_else(|err| {
                eprintln!("read {header}: {err}");
                std::process::exit(1);
            });
            let (spec, tokens, _) = inkling_goldy::model::read_golden(&header_bytes)
                .unwrap_or_else(|err| {
                    eprintln!("header: {err}");
                    std::process::exit(1);
                });
            (spec, tokens)
        } else {
            eprintln!("generate needs --config <config.json> or --header <mini_forward.bin>");
            std::process::exit(2);
        };
        let prompt_tokens = if let Some(text) = text {
            let path = tokenizer.unwrap_or_else(|| "assets/tokenizer.json".into());
            let tok =
                inkling_goldy::tokenizer::InklingTokenizer::from_file(path).unwrap_or_else(|err| {
                    eprintln!("{err}");
                    std::process::exit(1);
                });
            tok.encode(&text).unwrap_or_else(|err| {
                eprintln!("{err}");
                std::process::exit(1);
            })
        } else if let Some(prompt) = prompt {
            prompt
                .split(',')
                .filter(|part| !part.is_empty())
                .map(|part| part.parse::<u32>().expect("prompt token"))
                .collect()
        } else {
            tokens.into_iter().take(2).collect()
        };
        let device = ammon::gpu::create_serial_runtime().expect("runtime");
        let ctx = device.create_context().expect("context");
        let mut model =
            inkling_goldy::model::TextModel::new(&device, spec, &mapped).expect("model");
        model.bind(&ctx).expect("bind");
        let decoded =
            inkling_goldy::generate::greedy(&mut model, &prompt_tokens, max_new).expect("greedy");
        println!(
            "prompt {:?} new {:?} prefill {:.3} s ({:.2} tok/s) decode {:.3} s ({:.2} tok/s, {:.3} s/tok)",
            decoded.prompt,
            decoded.new_tokens,
            decoded.prefill_secs,
            decoded.prefill_tokens_per_sec(),
            decoded.decode_secs,
            decoded.decode_tokens_per_sec(),
            decoded.decode_latency_secs(),
        );
    }
}
