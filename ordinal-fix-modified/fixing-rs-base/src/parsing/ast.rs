use crate::grammar::{unescape_string, StrEscapeError, SymbolType};
use std::{
    cell::{Ref, RefCell},
    ops::Deref,
};
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlternativeNode<'input> {
    pub id: usize,
    pub elements: Vec<Element<'input>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RulesNode<'input> {
    pub sym: &'input str,
    pub types: Vec<&'input str>,
    pub root_symbol: Option<()>,
    pub alternatives: Vec<AlternativeNode<'input>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Element<'input> {
    pub element_type: SymbolType,
    pub element_value: &'input str,
    pub element_literal_value: RefCell<Option<String>>,
}

impl<'input> Element<'input> {
    pub fn new(element_type: SymbolType, element_value: &'input str) -> Self {
        Self {
            element_type,
            element_value,
            element_literal_value: RefCell::new(None),
        }
    }

    pub fn gen_element_literal_value(&self) -> Result<(), StrEscapeError> {
        *self.element_literal_value.borrow_mut() = match self.element_type {
            SymbolType::LiteralTerminal => Some(unescape_string(self.element_value)?),
            _ => None,
        };
        Ok(())
    }

    pub fn get_element_value(&self) -> DerefReturnValues<'_> {
        match self.element_type {
            SymbolType::LiteralTerminal => {
                DerefReturnValues::RefMap(Ref::map(self.element_literal_value.borrow(), |x| {
                    x.as_ref().unwrap()
                }))
            }
            _ => DerefReturnValues::StrRef(self.element_value),
        }
    }
}

pub enum DerefReturnValues<'a> {
    RefMap(Ref<'a, String>),
    StrRef(&'a str),
}

impl<'a> Deref for DerefReturnValues<'a> {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::RefMap(x) => x,
            Self::StrRef(x) => x,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GrammarFile<'input> {
    pub rules: Vec<RulesNode<'input>>,
    pub multivalued_symbols: Vec<&'input str>,
    pub annos: Vec<TerminalAnno<'input>>,
}

#[derive(Clone, Debug)]
pub struct TerminalAnno<'input> {
    pub name: &'input str,
    pub types: Vec<&'input str>,
}
