use super::FKeyRef;
use crate::{grammar::SymbolRef, props::UnionProp, utils::LinkedSeq};

#[derive(Debug, Getters, CopyGetters)]
pub struct ResultToken<'a> {
    #[getset(get_copy = "pub")]
    symbol: SymbolRef<'a>,
    #[getset(get = "pub")]
    value: Option<String>,
}

impl<'a> ResultToken<'a> {
    pub fn new(symbol: SymbolRef<'a>, value: Option<String>) -> Self {
        Self { symbol, value }
    }
}

pub struct ResultStorage<'a, 'b, PG, PSI, PSS>
where
    PG: UnionProp,
    PSI: UnionProp,
    PSS: UnionProp,
{
    processing: Vec<(
        LinkedSeq<ResultToken<'a>>,
        LinkedSeq<FKeyRef<'a, 'b, PG, PSI, PSS>>,
    )>,
    finished: Vec<LinkedSeq<ResultToken<'a>>>,
    limit: usize,
}

impl<'a, 'b, PG, PSI, PSS> ResultStorage<'a, 'b, PG, PSI, PSS>
where
    PG: UnionProp,
    PSI: UnionProp,
    PSS: UnionProp,
{
    pub fn new(limit: usize) -> Self {
        Self {
            processing: Vec::new(),
            finished: Vec::new(),
            limit,
        }
    }

    pub fn push_processing(
        &mut self,
        tokens: LinkedSeq<ResultToken<'a>>,
        fkey: LinkedSeq<FKeyRef<'a, 'b, PG, PSI, PSS>>,
    ) {
        if fkey.is_none() {
            if self.finished.len() >= self.limit {
                return;
            }
            self.finished.push(tokens);
        } else {
            if self.processing.len() + self.finished.len() >= self.limit {
                return;
            }
            self.processing.push((tokens, fkey));
        }
    }

    pub fn pop_processing(
        &mut self,
    ) -> Option<(
        LinkedSeq<ResultToken<'a>>,
        LinkedSeq<FKeyRef<'a, 'b, PG, PSI, PSS>>,
    )> {
        self.processing.pop()
    }

    pub fn take_finished(self) -> Vec<LinkedSeq<ResultToken<'a>>> {
        self.finished
    }

    pub fn should_continue(&self) -> bool {
        self.finished.len() < self.limit
    }

    pub fn is_multiple(&self) -> bool {
        self.limit > 1
    }
}
