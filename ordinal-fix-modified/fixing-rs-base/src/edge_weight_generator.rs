use crate::{
    containers::{Map, Set},
    grammar::{GrammarSymbolsRef, SymbolRef, SymbolType},
    utils::{u8_slice_to_i16_slice, StringPool, StringRef},
};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::{
    error::Error,
    fmt::{Debug, Display},
    iter,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GraphEdgeType {
    Original,
    Insertion,
    Update,
    Deletion,
}

pub trait EdgeWeightGenerator<'a, 'b> {
    fn get_weight(
        &self,
        ty: GraphEdgeType,
        index: usize,
        symbol: SymbolRef<'a>,
        literal: Option<&str>,
    ) -> Option<usize>;

    fn get_weight_multi<'c>(
        &'c self,
        ty: GraphEdgeType,
        index: usize,
        symbol: SymbolRef<'a>,
    ) -> impl Iterator<Item = usize> + 'c;

    fn get_literals_with_weight<'c>(
        &'c self,
        ty: GraphEdgeType,
        index: usize,
        symbol: SymbolRef<'a>,
        weight: usize,
    ) -> Option<impl Iterator<Item = StringRef<'b>> + 'c>
    where
        'b: 'c;
}

pub struct DefaultEdgeWeightGenrator;

impl<'a, 'b> EdgeWeightGenerator<'a, 'b> for DefaultEdgeWeightGenrator {
    fn get_weight(
        &self,
        ty: GraphEdgeType,
        _index: usize,
        _symbol: SymbolRef<'a>,
        _literal: Option<&str>,
    ) -> Option<usize> {
        Some(if ty == GraphEdgeType::Original { 0 } else { 1 })
    }

    fn get_weight_multi<'c>(
        &'c self,
        _ty: GraphEdgeType,
        _index: usize,
        _symbol: SymbolRef<'a>,
    ) -> impl Iterator<Item = usize> + 'c {
        iter::once(1)
    }

    fn get_literals_with_weight<'c>(
        &'c self,
        _ty: GraphEdgeType,
        _index: usize,
        _symbol: SymbolRef<'a>,
        _weight: usize,
    ) -> Option<impl Iterator<Item = StringRef<'b>> + 'c>
    where
        'b: 'c,
    {
        None::<iter::Empty<_>>
    }
}

#[derive(Debug)]
pub struct MultipleWeightMappings<'b> {
    map: Map<StringRef<'b>, usize>,
    map_inv: Map<usize, Set<StringRef<'b>>>,
}

impl<'b> MultipleWeightMappings<'b> {
    pub fn new() -> Self {
        Self {
            map: Map::new(),
            map_inv: Map::new(),
        }
    }

    pub fn insert(&mut self, key: StringRef<'b>, value: usize) {
        self.map.insert(key, value);
        self.map_inv
            .entry(value)
            .or_insert_with(Set::new)
            .insert(key);
    }
}

#[derive(Serialize, Deserialize)]
struct TokenDesc {
    is_symbolic: bool,
    name: String,
    value: String,
}

#[derive(Serialize, Deserialize)]
struct JsonEdgeWeights {
    length: usize,
    tokens: Vec<TokenDesc>,
    origin: String,
    insert: String,
    update: String,
    remove: String,
}

#[derive(Debug)]
pub enum JsonLoadError {
    DeserializeError(serde_json::Error),
    NoSuchSymbol(String),
    Base64DecodeErrorr(base64::DecodeError, &'static str),
    InvalidLength(usize, usize, &'static str),
}

impl Display for JsonLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self, f)
    }
}

impl Error for JsonLoadError {}

impl From<serde_json::Error> for JsonLoadError {
    fn from(err: serde_json::Error) -> Self {
        Self::DeserializeError(err)
    }
}

#[derive(Debug)]
pub struct MappingEdgeWeightGenerator<'a, 'b> {
    original: Vec<usize>,
    removal: Vec<Option<usize>>,
    insertion: Vec<Map<SymbolRef<'a>, usize>>,
    insertion_multi: Vec<Map<SymbolRef<'a>, MultipleWeightMappings<'b>>>,
    update: Vec<Map<SymbolRef<'a>, usize>>,
    update_multi: Vec<Map<SymbolRef<'a>, MultipleWeightMappings<'b>>>,
    empty_map: Map<usize, Set<StringRef<'b>>>,
    empty_set: Set<StringRef<'b>>,
}

impl<'a, 'b> MappingEdgeWeightGenerator<'a, 'b> {
    pub fn new(token_length: usize) -> Self {
        Self {
            original: vec![0; token_length],
            removal: vec![None; token_length],
            insertion: (0..=token_length).map(|_| Map::new()).collect(),
            insertion_multi: (0..=token_length).map(|_| Map::new()).collect(),
            update: (0..token_length).map(|_| Map::new()).collect(),
            update_multi: (0..token_length).map(|_| Map::new()).collect(),
            empty_map: Map::new(),
            empty_set: Set::new(),
        }
    }

    pub fn from_json_file(
        file: &str,
        grammar: GrammarSymbolsRef<'a>,
        str_pool: &StringPool<'b>,
    ) -> Result<Self, JsonLoadError> {
        let json: JsonEdgeWeights = serde_json::from_str(file)?;
        let token_length = json.length;

        let mut original = vec![0; token_length];
        let mut removal = vec![None; token_length];
        let mut insertion: Vec<Map<SymbolRef<'a>, usize>> =
            (0..=token_length).map(|_| Map::new()).collect();
        let mut insertion_multi: Vec<Map<SymbolRef<'a>, MultipleWeightMappings<'b>>> =
            (0..=token_length).map(|_| Map::new()).collect();
        let mut update: Vec<Map<SymbolRef<'a>, usize>> =
            (0..token_length).map(|_| Map::new()).collect();
        let mut update_multi: Vec<Map<SymbolRef<'a>, MultipleWeightMappings<'b>>> =
            (0..token_length).map(|_| Map::new()).collect();

        let symbols: Result<Vec<_>, _> = json
            .tokens
            .iter()
            .map(|token| -> Result<_, JsonLoadError> {
                let symbol = if token.is_symbolic {
                    grammar.symbolic_terminals
                } else {
                    grammar.literal_terminals
                }
                .get(token.name.as_str())
                .ok_or_else(|| JsonLoadError::NoSuchSymbol(token.name.clone()))?;
                let symbol = *symbol;
                let multi_valued = symbol.symbol_type() == SymbolType::SymbolicTerminal
                    && symbol.is_multi_valued();
                let value = if multi_valued {
                    Some(str_pool.get_or_add(token.value.as_str()))
                } else {
                    None
                };
                Ok((symbol, multi_valued, value))
            })
            .collect();
        let symbols = symbols?;

        let strs = vec![
            (&json.origin, "origin", GraphEdgeType::Deletion, json.length),
            (&json.remove, "remove", GraphEdgeType::Deletion, json.length),
            (
                &json.insert,
                "insert",
                GraphEdgeType::Insertion,
                (token_length + 1) * symbols.len(),
            ),
            (
                &json.update,
                "update",
                GraphEdgeType::Update,
                token_length * symbols.len(),
            ),
        ];
        let process_result: Result<(), JsonLoadError> = strs
            .into_iter()
            .map(|(s, msg, ty, len)| -> Result<(), JsonLoadError> {
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(s)
                    .map_err(|e| JsonLoadError::Base64DecodeErrorr(e, msg))?;
                if bytes.len() != len * 2 {
                    Err(JsonLoadError::InvalidLength(bytes.len(), len, msg))?
                }
                let values = u8_slice_to_i16_slice(bytes.as_slice());
                match ty {
                    GraphEdgeType::Original | GraphEdgeType::Deletion => {
                        for (i, &v) in values.iter().enumerate() {
                            if v > 0 {
                                if ty == GraphEdgeType::Original {
                                    original[i] = v as usize;
                                } else {
                                    removal[i] = Some(v as usize)
                                }
                            }
                        }
                    }
                    GraphEdgeType::Insertion | GraphEdgeType::Update => {
                        let mut cur: usize = 0;
                        for i in
                            0..token_length + if ty == GraphEdgeType::Insertion { 1 } else { 0 }
                        {
                            for &(symbol, multi_valued, value) in symbols.iter() {
                                let v = values[cur];
                                if v > 0 {
                                    if multi_valued {
                                        let map = if ty == GraphEdgeType::Insertion {
                                            &mut insertion_multi
                                        } else {
                                            &mut update_multi
                                        };
                                        let map = &mut map[i];
                                        map.entry(symbol)
                                            .or_insert_with(MultipleWeightMappings::new)
                                            .insert(value.unwrap(), v as usize);
                                    } else {
                                        let map = if ty == GraphEdgeType::Insertion {
                                            &mut insertion
                                        } else {
                                            &mut update
                                        };
                                        map[i].insert(symbol, v as usize);
                                    }
                                }
                                cur += 1;
                            }
                        }
                    }
                }
                Ok(())
            })
            .collect();
        process_result?;

        Ok(Self {
            original,
            removal,
            insertion,
            insertion_multi,
            update,
            update_multi,
            empty_map: Map::new(),
            empty_set: Set::new(),
        })
    }
}

impl<'a, 'b> EdgeWeightGenerator<'a, 'b> for MappingEdgeWeightGenerator<'a, 'b> {
    fn get_weight(
        &self,
        ty: GraphEdgeType,
        index: usize,
        symbol: SymbolRef<'a>,
        _literal: Option<&str>,
    ) -> Option<usize> {
        match ty {
            GraphEdgeType::Original => Some(self.original[index]),
            GraphEdgeType::Insertion => self.insertion[index].get(&symbol).copied(),
            GraphEdgeType::Update => self.update[index].get(&symbol).copied(),
            GraphEdgeType::Deletion => self.removal[index],
        }
    }

    fn get_weight_multi<'c>(
        &'c self,
        ty: GraphEdgeType,
        index: usize,
        symbol: SymbolRef<'a>,
    ) -> impl Iterator<Item = usize> + 'c {
        match ty {
            GraphEdgeType::Insertion => {
                if let Some(multi) = self.insertion_multi[index].get(&symbol) {
                    &multi.map_inv
                } else {
                    &self.empty_map
                }
            }
            GraphEdgeType::Update => {
                if let Some(multi) = self.update_multi[index].get(&symbol) {
                    &multi.map_inv
                } else {
                    &self.empty_map
                }
            }
            _ => unreachable!(),
        }
        .keys()
        .copied()
    }

    fn get_literals_with_weight<'c>(
        &'c self,
        ty: GraphEdgeType,
        index: usize,
        symbol: SymbolRef<'a>,
        weight: usize,
    ) -> Option<impl Iterator<Item = StringRef<'b>> + 'c>
    where
        'b: 'c,
    {
        let map = match ty {
            GraphEdgeType::Insertion => &self.insertion_multi[index],
            GraphEdgeType::Update => &self.update_multi[index],
            _ => unreachable!(),
        };
        map.get(&symbol)
            .and_then(|m| m.map_inv.get(&weight))
            .or_else(|| Some(&self.empty_set))
            .map(|x| x.iter().copied())
    }
}
