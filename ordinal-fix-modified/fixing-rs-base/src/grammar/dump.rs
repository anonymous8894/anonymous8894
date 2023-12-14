use super::{GrammarSymbolsRef, SymbolMap};
use crate::utils::Pointer;
use serde::Serialize;
use std::io::Write;

#[derive(Serialize)]
struct DumpSymbols<'a> {
    non_terminals: Vec<&'a str>,
    symbolic_terminals: Vec<&'a str>,
    literal_terminals: Vec<&'a str>,
}

fn symbols_to_vec_string<'a>(map: &'a SymbolMap<'a>) -> Vec<&'a str> {
    map.values().map(|s| Pointer::ptr(s).name()).collect()
}

pub fn dump_symbols<'a, W>(
    grammar: &GrammarSymbolsRef<'a>,
    writer: W,
) -> Result<(), serde_json::Error>
where
    W: Write,
{
    let json = DumpSymbols {
        non_terminals: symbols_to_vec_string(&grammar.non_terminals),
        symbolic_terminals: symbols_to_vec_string(&grammar.symbolic_terminals),
        literal_terminals: symbols_to_vec_string(&grammar.literal_terminals),
    };
    serde_json::to_writer(writer, &json)
}
