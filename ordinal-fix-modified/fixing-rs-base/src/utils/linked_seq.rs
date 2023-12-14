use std::rc::Rc;

#[derive(Clone)]
pub struct LinkedSeqNode<T> {
    last: T,
    remaining: LinkedSeq<T>,
}

impl<T> LinkedSeqNode<T> {
    pub fn last(&self) -> &T {
        &self.last
    }

    pub fn remaining(&self) -> &LinkedSeq<T> {
        &self.remaining
    }

    pub fn new(last: T, remaining: &LinkedSeq<T>) -> Self {
        Self {
            last,
            remaining: remaining.clone(),
        }
    }
}

pub struct LinkedSeq<T>(Option<Rc<LinkedSeqNode<T>>>);

impl<T> Clone for LinkedSeq<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> LinkedSeq<T> {
    pub fn new() -> Self {
        Self(None)
    }

    pub fn append(&self, last: T) -> Self {
        Self(Some(Rc::new(LinkedSeqNode::new(last, self))))
    }

    pub fn pop(&self) -> Option<(&T, &Self)> {
        self.0.as_ref().map(|node| (node.last(), node.remaining()))
    }

    pub fn is_none(&self) -> bool {
        self.0.is_none()
    }

    pub fn one(last: T) -> Self {
        Self::new().append(last)
    }
}

impl<'a, T> Iterator for &'a LinkedSeq<T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.pop().map(|(last, remaining)| {
            *self = remaining;
            last
        })
    }
}

impl<T> Default for LinkedSeq<T> {
    fn default() -> Self {
        Self::new()
    }
}
