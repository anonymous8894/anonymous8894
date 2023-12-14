use super::{
    mjenv::{MJClsRef, MJEnv},
    mjsymtab::{MJArgs, MJDecl, MJIdSelected, MJIdSelector, MJSymTab},
    syntactic::MJProp,
};
use fixing_rs_base::{
    containers::Set,
    edge_weight_generator::{EdgeWeightGenerator, GraphEdgeType},
    grammar::{GrammarSymbolsRef, SymbolRef},
    props::{PropArray, PropEmpty},
    tokenizer::Token,
    union_prop,
    utils::{StringPool, StringRef},
};
use std::{collections::HashSet, marker::PhantomData};

pub struct MJSProcessor<'a, 'b, G: EdgeWeightGenerator<'a, 'b>> {
    mjenv: &'b MJEnv<'b>,
    str_pool: &'b StringPool<'b>,
    identifiers: Vec<StringRef<'b>>,
    weight_generator: &'b G,
    symbol_identifier: SymbolRef<'a>,
    _photom: PhantomData<&'a ()>,
}

impl<'a, 'b, G: EdgeWeightGenerator<'a, 'b>> MJSProcessor<'a, 'b, G> {
    pub fn new(
        mjenv: &'b MJEnv<'b>,
        str_pool: &'b StringPool<'b>,
        tokens: &Vec<Token<'_, '_>>,
        max_new_id: usize,
        weight_generator: &'b G,
        grammar_ref: GrammarSymbolsRef<'a>,
    ) -> Self {
        let mut all_identifiers = HashSet::new();
        for token in tokens {
            if token.symbol.name() == "IDENTIFIER" {
                all_identifiers.insert(token.literal.to_owned());
            }
        }
        for token in mjenv.iter_names() {
            all_identifiers.insert(token.to_string());
        }
        for i in 0..max_new_id {
            all_identifiers.insert(format!("__new_id_{}", i));
        }

        let mut identifiers = Vec::new();
        for ident in all_identifiers {
            identifiers.push(str_pool.get_or_add(&ident[..]));
        }

        let symbol_identifier = *grammar_ref.symbolic_terminals.get("IDENTIFIER").unwrap();

        Self {
            str_pool,
            mjenv,
            identifiers,
            weight_generator,
            symbol_identifier,
            _photom: PhantomData,
        }
    }
}

union_prop!(
    MJSynProp<'a>,
    Empty,
    {
        Empty(PropEmpty),
        Type(MJClsRef<'a>),
        Decl(MJDecl<'a>),
        Str(StringRef<'a>),
        IdSelected(MJIdSelected<'a>)
    }
);

union_prop!(
    MJInhProp<'a>,
    Empty,
    {
        Empty(PropEmpty),
        SymTab(MJSymTab<'a>),
        IdSelector(MJIdSelector<'a>),
        Args(MJArgs<'a>)
    }
);

#[impl_semantic_processor(
    g_prop = "MJProp",
    si_prop = "MJInhProp<'b>",
    ss_prop = "MJSynProp<'b>",
    grammar_file = "fixing-rs-main/src/mj/middle_weight_java"
)]
#[allow(non_snake_case)]
impl<'a, 'b, G> MJSProcessor<'a, 'b, G>
where
    G: EdgeWeightGenerator<'a, 'b>,
{
    fn rooti(&self) -> MJSymTab<'b> {
        MJSymTab::new()
    }

    //nts statements: 1 statement statements
    fn nti_statements_1_1(
        &self,
        _g: &PropArray<MJProp>,
        inh: &MJSymTab<'b>,
        left: &MJDecl<'b>,
    ) -> MJSymTab<'b> {
        inh.extend_decl(left)
    }

    //nts statement: 0 ';'
    fn nts_statement_0(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        _s1: &PropEmpty,
    ) -> MJDecl<'b> {
        MJDecl::empty()
    }

    //nts statement: 1 declaration
    fn nts_statement_1(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        decl: &MJDecl<'b>,
    ) -> MJDecl<'b> {
        decl.clone()
    }

    //nts statement: 2 pExpression ';'
    fn nts_statement_2(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        _s1: &MJClsRef<'b>,
        _s2: &PropEmpty,
    ) -> MJDecl<'b> {
        MJDecl::empty()
    }

    //nts statement: 4 expression '.' fieldName '=' expression ';'
    fn nts_statement_4(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        _s1: &MJClsRef<'b>,
        _s2: &PropEmpty,
        left: &MJIdSelected<'b>,
        _s4: &PropEmpty,
        right: &MJClsRef<'b>,
        _s6: &PropEmpty,
    ) -> Option<MJDecl<'b>> {
        if self
            .mjenv
            .can_right_assign_to_left(left.unwrap_ty(), *right)
        {
            Some(MJDecl::empty())
        } else {
            None
        }
    }

    //nts statement: 5 identifier '=' expression ';'
    fn nts_statement_5(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        left: &MJIdSelected<'b>,
        _s2: &PropEmpty,
        right: &MJClsRef<'b>,
        _s4: &PropEmpty,
    ) -> Option<MJDecl<'b>> {
        if self
            .mjenv
            .can_right_assign_to_left(left.unwrap_ty(), *right)
        {
            Some(MJDecl::empty())
        } else {
            None
        }
    }

    //nts statement: 6 'return' expression ';'
    fn nts_statement_6(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        _s1: &PropEmpty,
        _s2: &MJClsRef<'b>,
        _s3: &PropEmpty,
    ) -> Option<MJDecl<'b>> {
        None
    }

    //nts statement: 7 'if' '(' expression '==' expression ')' block 'else' block
    fn nts_statement_7(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        _s1: &PropEmpty,
        _s2: &PropEmpty,
        _s3: &MJClsRef<'b>,
        _s4: &PropEmpty,
        _s5: &MJClsRef<'b>,
        _s6: &PropEmpty,
        _s7: &PropEmpty,
        _s8: &PropEmpty,
        _s9: &PropEmpty,
    ) -> MJDecl<'b> {
        MJDecl::empty()
    }

    //nts statement: 8 block
    fn nts_statement_8(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        _s1: &PropEmpty,
    ) -> MJDecl<'b> {
        MJDecl::empty()
    }

    //nts statement: 9 'return' ';'
    fn nts_statement_9(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        _s1: &PropEmpty,
        _s2: &PropEmpty,
    ) -> MJDecl<'b> {
        MJDecl::empty()
    }

    //nti 2 statement: 4 expression '.' fieldName '=' expression ';'
    fn nti_statement_4_2(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        expr: &MJClsRef<'b>,
        _s2: &PropEmpty,
    ) -> MJIdSelector<'b> {
        MJIdSelector::Field(expr.clone())
    }

    //nti 0 statement: 5 identifier '=' expression ';'
    fn nti_statement_5_0(&self, _g: &PropArray<MJProp>, inh: &MJSymTab<'b>) -> MJIdSelector<'b> {
        MJIdSelector::Identifier(inh.clone())
    }

    //nts declaration: 0 className newIdentifier ';'
    fn nts_declaration_0(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        ty: &MJIdSelected<'b>,
        name: &MJIdSelected<'b>,
        _s3: &PropEmpty,
    ) -> MJDecl<'b> {
        MJDecl::new(name.unwrap_newid(), ty.unwrap_ty())
    }

    //nti 0 declaration: 0 className newIdentifier ';'
    fn nti_declaration_0_0(&self, _g: &PropArray<MJProp>, _inh: &MJSymTab<'b>) -> MJIdSelector<'b> {
        MJIdSelector::Class
    }

    //nti 1 declaration: 0 className newIdentifier ';'
    fn nti_declaration_0_1(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        _s1: &MJIdSelected<'b>,
    ) -> MJIdSelector<'b> {
        MJIdSelector::NewIdentifier
    }

    //nts expression: 0 identifier
    fn nts_expression_0(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        id: &MJIdSelected<'b>,
    ) -> MJClsRef<'b> {
        id.unwrap_ty()
    }

    // nts expression: 1 'null'
    fn nts_expression_1(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        _s1: &PropEmpty,
    ) -> MJClsRef<'b> {
        self.mjenv.get_default_null()
    }

    // nts expression: 2 expression '.' fieldName
    fn nts_expression_2(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        _s1: &MJClsRef<'b>,
        _s2: &PropEmpty,
        id: &MJIdSelected<'b>,
    ) -> MJClsRef<'b> {
        id.unwrap_ty()
    }

    // nts expression: 3 '(' className ')' expression
    fn nts_expression_3(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        _s1: &PropEmpty,
        ty: &MJIdSelected<'b>,
        _s3: &PropEmpty,
        _s4: &MJClsRef<'b>,
    ) -> MJClsRef<'b> {
        ty.unwrap_ty()
    }

    // nts expression: 4 pExpression
    fn nts_expression_4(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        pexp: &MJClsRef<'b>,
    ) -> MJClsRef<'b> {
        *pexp
    }

    // nts expression: 5 '(' expression ')'
    fn nts_expression_5(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        _s1: &PropEmpty,
        exp: &MJClsRef<'b>,
        _s3: &PropEmpty,
    ) -> MJClsRef<'b> {
        *exp
    }

    // nti 0 expression: 0 identifier
    fn nti_expression_0_0(&self, _g: &PropArray<MJProp>, inh: &MJSymTab<'b>) -> MJIdSelector<'b> {
        MJIdSelector::Identifier(inh.clone())
    }

    //nti 2 expression: 2 expression '.' fieldName
    fn nti_expression_2_2(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        exp: &MJClsRef<'b>,
        _s2: &PropEmpty,
    ) -> MJIdSelector<'b> {
        MJIdSelector::Field(*exp)
    }

    // nti 1 expression: 3 '(' className ')' expression
    fn nti_expression_3_1(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        _s1: &PropEmpty,
    ) -> MJIdSelector<'b> {
        MJIdSelector::Class
    }

    // nts pExpression: 0 expression '.' methodName '(' argumentList ')'
    fn nts_pExpression_0(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        _s1: &MJClsRef<'b>,
        _s2: &PropEmpty,
        id: &MJIdSelected<'b>,
        _s4: &PropEmpty,
        _s5: &PropEmpty,
        _s6: &PropEmpty,
    ) -> MJClsRef<'b> {
        match id.unwrap_method().ret_ty() {
            Some(ty) => *ty,
            None => self.mjenv.get_default_void(),
        }
    }

    // nts pExpression: 1 'new' className '(' argumentList ')'
    fn nts_pExpression_1(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        _s1: &PropEmpty,
        id: &MJIdSelected<'b>,
        _s3: &PropEmpty,
        _s4: &PropEmpty,
        _s5: &PropEmpty,
    ) -> MJClsRef<'b> {
        id.unwrap_ty()
    }

    // nti 2 pExpression: 0 expression '.' fieldName '(' argumentList ')'
    fn nti_pExpression_0_2(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        exp: &MJClsRef<'b>,
        _s2: &PropEmpty,
    ) -> MJIdSelector<'b> {
        MJIdSelector::Method(*exp)
    }

    // nti 1 pExpression: 1 'new' className '(' argumentList ')'
    fn nti_pExpression_1_1(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJSymTab<'b>,
        _s1: &PropEmpty,
    ) -> MJIdSelector<'b> {
        MJIdSelector::Class
    }

    // nti 4 pExpression: 0 expression '.' fieldName '(' argumentList ')'
    fn nti_pExpression_0_4(
        &self,
        _g: &PropArray<MJProp>,
        inh: &MJSymTab<'b>,
        _s1: &MJClsRef<'b>,
        _s2: &PropEmpty,
        id: &MJIdSelected<'b>,
        _s4: &PropEmpty,
    ) -> MJArgs<'b> {
        MJArgs::Method(id.unwrap_method(), 0, inh.clone())
    }

    // nti 3 pExpression: 1 'new' className '(' argumentList ')'
    fn nti_pExpression_1_3(
        &self,
        _g: &PropArray<MJProp>,
        inh: &MJSymTab<'b>,
        _s1: &PropEmpty,
        id: &MJIdSelected<'b>,
        _s3: &PropEmpty,
    ) -> MJArgs<'b> {
        MJArgs::Constructor(id.unwrap_ty().constructor(), 0, inh.clone())
    }

    // nts argumentList: 0
    fn nts_argumentList_0(&self, _g: &PropArray<MJProp>, inh: &MJArgs<'b>) -> Option<PropEmpty> {
        let m = match inh {
            MJArgs::Method(m, _, _) => m.params(),
            MJArgs::Constructor(m, _, _) => m.params(),
        };
        if m.len() == 0 {
            Some(PropEmpty)
        } else {
            None
        }
    }

    // nti 0 argumentList: 1 argumentListOther
    fn nti_argumentList_1_0(&self, _g: &PropArray<MJProp>, inh: &MJArgs<'b>) -> Option<MJArgs<'b>> {
        let m = match inh {
            MJArgs::Method(m, _, _) => m.params(),
            MJArgs::Constructor(m, _, _) => m.params(),
        };
        if m.len() > 0 {
            Some(inh.clone())
        } else {
            None
        }
    }

    // nts argumentListOther: 0 expression
    fn nts_argumentListOther_0(
        &self,
        _g: &PropArray<MJProp>,
        inh: &MJArgs<'b>,
        exp: &MJClsRef<'b>,
    ) -> Option<PropEmpty> {
        let (m, c) = match inh {
            MJArgs::Method(m, c, _) => (m.params(), c),
            MJArgs::Constructor(m, c, _) => (m.params(), c),
        };
        if self.mjenv.can_right_assign_to_left(m[*c], *exp) {
            Some(PropEmpty)
        } else {
            None
        }
    }

    // nti 0 argumentListOther: 0 expression
    fn nti_argumentListOther_0_0(
        &self,
        _g: &PropArray<MJProp>,
        inh: &MJArgs<'b>,
    ) -> Option<MJSymTab<'b>> {
        let (m, c, i) = match inh {
            MJArgs::Method(m, c, i) => (m.params(), c, i),
            MJArgs::Constructor(m, c, i) => (m.params(), c, i),
        };
        if m.len() == c + 1 {
            Some(i.clone())
        } else {
            None
        }
    }

    // nti 0 argumentListOther: 1 expression ',' argumentListOther
    fn nti_argumentListOther_1_0(
        &self,
        _g: &PropArray<MJProp>,
        inh: &MJArgs<'b>,
    ) -> Option<MJSymTab<'b>> {
        let (m, c, i) = match inh {
            MJArgs::Method(m, c, i) => (m.params(), c, i),
            MJArgs::Constructor(m, c, i) => (m.params(), c, i),
        };
        if m.len() > c + 1 {
            Some(i.clone())
        } else {
            None
        }
    }

    // nti 2 argumentListOther: 1 expression ',' argumentListOther
    fn nti_argumentListOther_1_2(
        &self,
        _g: &PropArray<MJProp>,
        inh: &MJArgs<'b>,
        exp: &MJClsRef<'b>,
        _s2: &PropEmpty,
    ) -> Option<MJArgs<'b>> {
        let (m, c) = match inh {
            MJArgs::Method(m, c, _) => (m.params(), c),
            MJArgs::Constructor(m, c, _) => (m.params(), c),
        };
        if !self.mjenv.can_right_assign_to_left(m[*c], *exp) {
            return None;
        }
        Some(inh.next())
    }

    // nts identifier: 0 IDENTIFIER
    fn nts_identifier_0(
        &self,
        _g: &PropArray<MJProp>,
        inh: &MJIdSelector<'b>,
        id: &StringRef<'b>,
    ) -> MJIdSelected<'b> {
        match inh {
            MJIdSelector::Identifier(sym_tab) => {
                MJIdSelected::Identifier(sym_tab.get(*id).unwrap())
            }
            _ => panic!("not identifier"),
        }
    }

    // nts className: 0 IDENTIFIER
    fn nts_className_0(
        &self,
        _g: &PropArray<MJProp>,
        inh: &MJIdSelector<'b>,
        id: &StringRef<'b>,
    ) -> MJIdSelected<'b> {
        match inh {
            MJIdSelector::Class => MJIdSelected::Identifier(self.mjenv.get_class(*id).unwrap()),
            _ => panic!("not classname"),
        }
    }

    // nts fieldName: 0 IDENTIFIER
    fn nts_fieldName_0(
        &self,
        _g: &PropArray<MJProp>,
        inh: &MJIdSelector<'b>,
        id: &StringRef<'b>,
    ) -> MJIdSelected<'b> {
        match inh {
            MJIdSelector::Field(c) => MJIdSelected::Identifier(
                *c.content().borrow().fields().get(id.as_str()).unwrap().ty(),
            ),
            _ => panic!("not field"),
        }
    }

    // nts methodName: 0 IDENTIFIER
    fn nts_methodName_0(
        &self,
        _g: &PropArray<MJProp>,
        inh: &MJIdSelector<'b>,
        id: &StringRef<'b>,
    ) -> MJIdSelected<'b> {
        match inh {
            MJIdSelector::Method(c) => {
                MJIdSelected::Method(*c.content().borrow().methods().get(id.as_str()).unwrap())
            }
            _ => panic!("not method"),
        }
    }

    // nts newIdentifier: 0 IDENTIFIER
    fn nts_newIdentifier_0(
        &self,
        _g: &PropArray<MJProp>,
        inh: &MJIdSelector<'b>,
        id: &StringRef<'b>,
    ) -> MJIdSelected<'b> {
        match inh {
            MJIdSelector::NewIdentifier => MJIdSelected::NewId(*id),
            _ => panic!("not new"),
        }
    }

    // sts IDENTIFIER
    fn sts_IDENTIFIER(
        &self,
        _g: &PropArray<MJProp>,
        inh: &MJIdSelector<'b>,
        literal: Option<&str>,
        edge_weight: usize,
        edge_type: GraphEdgeType,
        index: usize,
    ) -> Vec<StringRef<'b>> {
        let current_identifiers = match edge_type {
            GraphEdgeType::Original => None,
            _ => self
                .weight_generator
                .get_literals_with_weight(edge_type, index, self.symbol_identifier, edge_weight)
                .map(|s| s.collect::<Set<_>>()),
        };
        let mut result = Vec::new();
        let literal = literal.map(|s| self.str_pool.get(s).unwrap());
        match inh {
            MJIdSelector::Identifier(sym_tab) => match literal {
                Some(literal) => {
                    if let Some(_) = sym_tab.get(literal) {
                        result.push(literal);
                    }
                }
                None => {
                    for (&k, _) in sym_tab.iter() {
                        if current_identifiers.is_none()
                            || current_identifiers.as_ref().unwrap().contains(&k)
                        {
                            result.push(k);
                        }
                    }
                }
            },
            MJIdSelector::Field(c) => match literal {
                Some(literal) => {
                    if let Some(_) = c.content().borrow().fields().get(literal.as_str()) {
                        result.push(literal);
                    }
                }
                None => {
                    for (k, _) in c.content().borrow().fields().iter() {
                        let k = self.str_pool.get(k).unwrap();
                        if current_identifiers.is_none()
                            || current_identifiers.as_ref().unwrap().contains(&k)
                        {
                            result.push(k);
                        }
                    }
                }
            },
            MJIdSelector::Method(c) => match literal {
                Some(literal) => {
                    if let Some(_) = c.content().borrow().methods().get(literal.as_str()) {
                        result.push(literal);
                    }
                }
                None => {
                    for (k, _) in c.content().borrow().methods().iter() {
                        let k = self.str_pool.get(k).unwrap();
                        if current_identifiers.is_none()
                            || current_identifiers.as_ref().unwrap().contains(&k)
                        {
                            result.push(k);
                        }
                    }
                }
            },
            MJIdSelector::Class => match literal {
                Some(literal) => {
                    if let Some(_) = self.mjenv.get_class(literal) {
                        result.push(literal);
                    }
                }
                None => {
                    for c in self.mjenv.iter_class() {
                        result.push(self.str_pool.get(c.name()).unwrap());
                    }
                }
            },
            MJIdSelector::NewIdentifier => match literal {
                Some(literal) => result.push(literal),
                None => {
                    for &k in self.identifiers.iter() {
                        if current_identifiers.is_none()
                            || current_identifiers.as_ref().unwrap().contains(&k)
                        {
                            result.push(k);
                        }
                    }
                }
            },
        }
        result
    }

    // stg IDENTIFIER
    fn stg_IDENTIFIER(
        &self,
        _g: &PropArray<MJProp>,
        _inh: &MJIdSelector<'b>,
        syn: &StringRef<'b>,
        _literal: Option<&str>,
    ) -> String {
        syn.to_string()
    }
}
