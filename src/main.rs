fn main() {
    let mini = inkling_goldy::tape::LoopSchedule::mini();
    println!(
        "inkling.goldy: Inkling 2 decoder layer on Ammon. Mini schedule is {} visits (see tools/ref/README.md).",
        mini.visits()
    );
}
