use crate::grammars::SupportedGrammar;
use clap::Parser;
use fixing_rs_base::grammar::{dump_symbols, Grammar, GrammarArena};
use std::{ffi::OsString, fs::File, io};

#[derive(Parser)]
pub struct DumpSymbols {
    #[clap(value_enum)]
    lang: SupportedGrammar,
    #[arg(long)]
    output_file: Option<OsString>,
}

impl DumpSymbols {
    pub fn run(self) {
        let grammar = self.lang.fixing_info().grammar;
        let grammar_arena = GrammarArena::new();
        let grammar = Grammar::new(&grammar_arena, grammar).unwrap();
        match self.output_file {
            Some(output_file) => {
                let output = File::create(output_file).unwrap();
                dump_symbols(&grammar.get_symbol_ref(), output).unwrap();
            }
            None => {
                dump_symbols(&grammar.get_symbol_ref(), io::stdout()).unwrap();
            }
        }
    }
}
