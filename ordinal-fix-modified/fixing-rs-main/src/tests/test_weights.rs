use crate::mj::fixing::MJ_GRAMMAR;
use fixing_rs_base::{
    edge_weight_generator::MappingEdgeWeightGenerator,
    grammar::{Grammar, GrammarArena},
    utils::{RefArena, StringPool},
};
use std::{fs::File, io::Write};

const EDGE_WEIGHT_20: &str = include_str!("test_weights/edge_weights_20.json");

#[test]
fn test_load_weights() {
    let grammar = MJ_GRAMMAR;
    let grammar_arena = GrammarArena::new();
    let grammar = Grammar::new(&grammar_arena, grammar).unwrap();
    let strings = RefArena::new();
    let str_pool = StringPool::new(&strings);
    let generator = MappingEdgeWeightGenerator::from_json_file(
        EDGE_WEIGHT_20,
        grammar.get_symbol_ref(),
        &str_pool,
    )
    .unwrap();
    let mut dump_file = File::create("../target/dump_test_weights.txt").unwrap();
    write!(dump_file, "{:?}", generator).unwrap();
}
