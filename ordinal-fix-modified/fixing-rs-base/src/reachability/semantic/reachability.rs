use super::{current::SReachabilityCurrent, FKeyRef, ResultStorage, ResultToken, SProcessor};
use crate::{
    grammar::SymbolType,
    props::UnionProp,
    reachability::{GProcessor, SReachabilityArena},
    utils::LinkedSeq,
};
use log::info;

mod cache;
mod edge;
pub use cache::{SReachabilityCache, SReachabilityCacheEntity, SReachabilityCacheEntityRef};
pub use edge::SReachabilityEdges;

pub struct SReachability<'a, 'b, PG, PSI, PSS>
where
    PG: UnionProp,
    PSI: UnionProp,
    PSS: UnionProp,
{
    arena: &'b SReachabilityArena<'a, 'b, PG, PSI, PSS>,
    edges: SReachabilityEdges<'a, 'b, PG, PSI, PSS>,
    cache: SReachabilityCache<'a, 'b, PG, PSI, PSS>,
}

impl<'a, 'b, 'q, PG, PSI, PSS> SReachability<'a, 'b, PG, PSI, PSS>
where
    PG: UnionProp,
    PSI: UnionProp,
    PSS: UnionProp,
{
    pub fn new(arena: &'b SReachabilityArena<'a, 'b, PG, PSI, PSS>, allow_multiple: bool) -> Self {
        Self {
            arena,
            edges: SReachabilityEdges::new(arena, allow_multiple),
            cache: SReachabilityCache::new(arena),
        }
    }

    pub(super) fn split(
        &mut self,
    ) -> (
        &'b SReachabilityArena<'a, 'b, PG, PSI, PSS>,
        &mut SReachabilityEdges<'a, 'b, PG, PSI, PSS>,
        &mut SReachabilityCache<'a, 'b, PG, PSI, PSS>,
    ) {
        (self.arena, &mut self.edges, &mut self.cache)
    }

    pub fn generate_from<'c, 'p, GProc, SProc>(
        &mut self,
        start: FKeyRef<'a, 'b, PG, PSI, PSS>,
        proc: &SProc,
        verbose: bool,
        result: &mut ResultStorage<'a, 'b, PG, PSI, PSS>,
        reachability_current: &'c SReachabilityCurrent<
            'a,
            'b,
            'c,
            'p,
            'q,
            PG,
            PSI,
            PSS,
            GProc,
            SProc,
        >,
    ) where
        GProc: GProcessor<PG = PG>,
        SProc: SProcessor<PG = PG, PSI = PSI, PSS = PSS>,
    {
        result.push_processing(Default::default(), LinkedSeq::one(start));
        loop {
            if !result.should_continue() {
                break;
            }
            if let Some((tokens, current)) = result.pop_processing() {
                let (current, remaining) = current.pop().unwrap();
                if result.is_multiple() {
                    let gkey = self.edges.get_entity(current).unwrap().gkey();
                    let it =
                        reachability_current.query_edge(gkey, current.inh_prop().clone(), self);
                    it.exhausive(self);
                }
                let entity = self.edges.get_entity(current).unwrap();
                if verbose {
                    info!(
                        "Gen: {} {} {} {} {:?} {:?} {:?} {:?}",
                        current.begin(),
                        current.end(),
                        current.symbol().name(),
                        entity.length(),
                        entity.literal(),
                        current.gprop(),
                        current.inh_prop(),
                        current.syn_prop(),
                    );
                }
                match current.symbol().symbol_type() {
                    SymbolType::LiteralTerminal => {
                        let current = tokens.append(ResultToken::new(current.symbol(), None));
                        result.push_processing(current, remaining.clone());
                    }
                    SymbolType::SymbolicTerminal => {
                        let gen = proc.process_symbolic_terminal_gen(
                            current.symbol(),
                            current.gprop(),
                            &current.inh_prop(),
                            current.syn_prop().unwrap_single(),
                            entity.literal().as_deref(),
                        );
                        result.push_processing(
                            tokens.append(ResultToken::new(current.symbol(), Some(gen))),
                            remaining.clone(),
                        );
                    }
                    SymbolType::NonTerminal => {
                        for entity in entity.rules().values() {
                            let mut remaining = remaining.clone();
                            match entity.right2() {
                                Some(key) => {
                                    remaining = remaining.append(key);
                                }
                                None => {}
                            }
                            match entity.right1() {
                                Some(key) => {
                                    remaining = remaining.append(key);
                                }
                                None => {}
                            }
                            result.push_processing(tokens.clone(), remaining);
                        }
                    }
                }
            } else {
                break;
            }
        }
    }

    // fn append(
    //     &mut self,
    //     current: &FKey<'a, PG, PSI, PSS>,
    //     proc: &impl SProcessor<PG = PG, PSI = PSI, PSS = PSS>,
    //     result: &mut ResultStorage<'a>,
    //     current_seq: ResultSequence<'a>,
    //     verbose: bool,
    // ) {
    //     let entity = self.edges.get_entity(current).unwrap();
    //     if verbose {
    //         info!(
    //             "Gen: {} {} {} {} {:?} {:?} {:?} {:?}",
    //             current.begin(),
    //             current.end(),
    //             current.symbol().name(),
    //             entity.length(),
    //             entity.literal(),
    //             current.gprop(),
    //             current.inh_prop(),
    //             current.syn_prop(),
    //         );
    //     }
    //     match current.symbol().symbol_type() {
    //         SymbolType::NonTerminal => {
    //             for entity in entity.rules().values() {
    //                 // let entity = entity.rules().values().next().unwrap();
    //                 // if let Some(key) = entity.right1() {
    //                 //     self.append(key.ptr(), proc, result, verbose);
    //                 // }
    //                 // if let Some(key) = entity.right2() {
    //                 //     self.append(key.ptr(), proc, result, verbose);
    //                 // }
    //             }
    //         }
    //         SymbolType::LiteralTerminal => {
    //             // result.push(current.symbol().name().to_string());
    //         }
    //         SymbolType::SymbolicTerminal => {
    //             let gen = proc.process_symbolic_terminal_gen(
    //                 current.symbol(),
    //                 current.gprop(),
    //                 &current.inh_prop(),
    //                 current.syn_prop().unwrap_single(),
    //                 entity.literal().as_deref(),
    //             );
    //             // result.push(gen);
    //         }
    //     }
    // }
}
