use super::{Pointer, RefArena};
use std::{cell::RefCell, collections::HashMap};

pub struct StringPool<'a> {
    pool: RefCell<HashMap<&'a str, StringRef<'a>>>,
    arena: &'a RefArena<String>,
}

impl<'a> StringPool<'a> {
    pub fn new(arena: &'a RefArena<String>) -> Self {
        Self {
            pool: RefCell::new(HashMap::new()),
            arena,
        }
    }

    pub fn get_or_add(&self, s: &str) -> StringRef<'a> {
        let mut pool = self.pool.borrow_mut();
        if let Some(r) = pool.get(s) {
            *r
        } else {
            let r = self.arena.alloc(s.to_string());
            pool.insert(r.ptr(), r);
            r
        }
    }

    pub fn get(&self, s: &str) -> Option<StringRef<'a>> {
        match self.pool.borrow().get(s) {
            Some(r) => Some(*r),
            None => {
                panic!("String not found: {}", s)
            }
        }
    }
}

pub type StringRef<'a> = Pointer<'a, String>;
