use super::{grammar::GrammarRuleRef, GrammarRuleLength};
use crate::utils::Pointer;
use serde::Serialize;
use std::{
    cell::{Ref, RefCell},
    fmt::{Debug, Display, Formatter},
};

pub type SymbolRef<'a> = Pointer<'a, Symbol<'a>>;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize)]
pub enum SymbolType {
    LiteralTerminal,
    SymbolicTerminal,
    NonTerminal,
}

impl Display for SymbolType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymbolType::LiteralTerminal => write!(f, "LiteralTerminal"),
            SymbolType::SymbolicTerminal => write!(f, "SymbolicTerminal"),
            SymbolType::NonTerminal => write!(f, "NonTerminal"),
        }
    }
}

pub struct Symbol<'a> {
    id: usize,
    symbol_type: SymbolType,
    name: String,

    entity: RefCell<SymbolEntity<'a>>,
}

pub struct SymbolEntity<'a> {
    rules: Vec<GrammarRuleRef<'a>>,

    ref_one: Vec<GrammarRuleRef<'a>>,
    ref_two_left: Vec<GrammarRuleRef<'a>>,
    ref_two_right: Vec<GrammarRuleRef<'a>>,

    is_multi_valued: bool,
}

impl<'a> Symbol<'a> {
    pub(super) fn new(id: usize, symbol_type: SymbolType, name: &str) -> Self {
        Self {
            id,
            symbol_type,
            name: String::from(name),
            entity: RefCell::new(SymbolEntity {
                rules: Vec::new(),
                ref_one: Vec::new(),
                ref_two_left: Vec::new(),
                ref_two_right: Vec::new(),
                is_multi_valued: false,
            }),
        }
    }

    pub(super) fn add_ref_one(&self, rule: GrammarRuleRef<'a>) {
        assert!(rule.rule_type().length() == GrammarRuleLength::One);
        assert!(std::ptr::eq(&*rule.right1().unwrap(), self));
        self.entity.borrow_mut().ref_one.push(rule);
    }
    pub(super) fn add_ref_two_left(&self, rule: GrammarRuleRef<'a>) {
        assert!(rule.rule_type().length() == GrammarRuleLength::Two);
        assert!(std::ptr::eq(&*rule.right1().unwrap(), self));
        self.entity.borrow_mut().ref_two_left.push(rule);
    }
    pub(super) fn add_ref_two_right(&self, rule: GrammarRuleRef<'a>) {
        assert!(rule.rule_type().length() == GrammarRuleLength::Two);
        assert!(std::ptr::eq(&*rule.right2().unwrap(), self));
        self.entity.borrow_mut().ref_two_right.push(rule);
    }
    pub(super) fn add_rule(&self, rule: GrammarRuleRef<'a>) {
        self.entity.borrow_mut().rules.push(rule);
    }
    pub(super) fn set_multi_valued(&self) {
        self.entity.borrow_mut().is_multi_valued = true;
    }

    pub fn symbol_id(&self) -> usize {
        self.id
    }
    pub fn symbol_type(&self) -> SymbolType {
        self.symbol_type
    }
    pub fn name<'b>(&'b self) -> &'b str {
        &self.name
    }
    pub fn rules<'b>(&'b self) -> Ref<'b, Vec<GrammarRuleRef<'a>>> {
        Ref::map(self.entity.borrow(), |e| &e.rules)
    }
    pub fn ref_one<'b>(&'b self) -> Ref<'b, Vec<GrammarRuleRef<'a>>> {
        Ref::map(self.entity.borrow(), |e| &e.ref_one)
    }
    pub fn ref_two_left<'b>(&'b self) -> Ref<'b, Vec<GrammarRuleRef<'a>>> {
        Ref::map(self.entity.borrow(), |e| &e.ref_two_left)
    }
    pub fn ref_two_right<'b>(&'b self) -> Ref<'b, Vec<GrammarRuleRef<'a>>> {
        Ref::map(self.entity.borrow(), |e| &e.ref_two_right)
    }
    pub fn is_multi_valued(&self) -> bool {
        self.entity.borrow().is_multi_valued
    }

    pub fn fmt_all(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self)?;
        for rule in self.rules().iter() {
            writeln!(f, "  {}", rule)?;
        }
        Ok(())
    }
}

impl Display for Symbol<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Symbol[{} {}]", self.symbol_type, self.name)?;
        Ok(())
    }
}

impl Debug for Symbol<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        <Symbol<'_> as Display>::fmt(&self, f)
    }
}

impl<'a> Display for SymbolRef<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.ptr())
    }
}
