use crate::{
    edge_weight_generator::{EdgeWeightGenerator, JsonLoadError},
    fixing_info::FixingInfo,
    grammar::{Grammar, GrammarArena, SymbolType},
    props::UnionProp,
    reachability::{
        find, GProcessor, GReachability, GReachabilityArena, ResultStorage, SProcessor,
        SReachability, SReachabilityArena,
    },
    tokenizer::Token,
};
use log::info;
use serde::Serialize;
use std::{
    error::Error,
    fmt::{Debug, Display},
    fs::{self, File},
    io::{self},
    time::{Duration, Instant},
};

pub struct FixTaskInfo {
    pub input_name: String,
    pub env_name: String,
    pub output_name: Option<String>,
    pub weights: Option<String>,
    pub max_len: usize,
    pub max_new_id: usize,
    pub verbose_gen: bool,
    pub output_nums: usize,
}

#[derive(Serialize, Debug)]
pub struct FixTaskOutputToken {
    pub ty: SymbolType,
    pub name: String,
    pub value: String,
}

#[derive(Debug)]
pub struct FixTaskResult {
    pub time_before_load: Instant,
    pub time_after_load: Instant,
    pub time_after_reachability_built: Instant,
    pub time_after_find: Vec<Instant>,
    pub found_length: Option<usize>,
    pub outputs: Vec<Vec<FixTaskOutputToken>>,
}

#[derive(Debug)]
pub enum FixTaskError<T: Error, E: Error> {
    ReadInputError(io::Error),
    ReadEnvError(io::Error),
    ReadWeightsError(io::Error),
    WriteOutputError(io::Error),
    TokenizerError(T),
    EnvLoadError(E),
    WeightsJsonLoadError(JsonLoadError),
}

impl<T: Debug + Error, E: Debug + Error> Display for FixTaskError<T, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as Debug>::fmt(self, f)
    }
}

impl<T: Error, E: Error> Error for FixTaskError<T, E> {}

impl<T: Error, E: Error> From<JsonLoadError> for FixTaskError<T, E> {
    fn from(e: JsonLoadError) -> Self {
        FixTaskError::WeightsJsonLoadError(e)
    }
}

fn do_fix_impl<'a, 'b, GProc, SProc, PG, T, E, G>(
    grammar: &'a Grammar<'a>,
    tokens: &Vec<Token<'a, '_>>,
    weight_generator: &G,
    gproc: &GProc,
    sproc: &SProc,
    info: &FixTaskInfo,
    time_before_load: Instant,
) -> Result<FixTaskResult, FixTaskError<T, E>>
where
    PG: UnionProp,
    G: EdgeWeightGenerator<'a, 'b>,
    GProc: GProcessor<PG = PG>,
    SProc: SProcessor<PG = PG>,
    T: Error,
    E: Error,
{
    let time_after_load = Instant::now();

    let greachability_arena = GReachabilityArena::new();
    let sreachability_arena = SReachabilityArena::new();
    let mut syntactic_reachability = GReachability::new(
        &grammar,
        &greachability_arena,
        &tokens,
        gproc,
        info.max_len,
        weight_generator,
    );
    let mut sreachability = SReachability::new(&sreachability_arena, false);

    let time_after_reachability_built = Instant::now();
    let mut time_after_find = Vec::new();

    let mut found_length = None;
    let mut outputs = None;
    for current_len in 0..=info.max_len {
        info!("Updating to length {}...", current_len);
        syntactic_reachability.update_until(current_len);
        let has_syn = if let Some(ref e) = syntactic_reachability.get_start_edges().get(current_len)
        {
            e.len() != 0
        } else {
            false
        };
        info!("Has syntactic reachability: {}", has_syn);
        // match
        let mut result_storage = ResultStorage::new(1);
        find(
            sproc,
            &sreachability_arena,
            &syntactic_reachability,
            current_len,
            current_len,
            &mut sreachability,
            &mut result_storage,
            sproc,
            info.verbose_gen,
        );
        if !result_storage.should_continue() {
            time_after_find.push(Instant::now());
            found_length = Some(current_len);
            let result_storage = result_storage.take_finished();
            outputs = Some(
                result_storage
                    .into_iter()
                    .map(|x| {
                        let mut result = x
                            .map(|t| FixTaskOutputToken {
                                ty: t.symbol().symbol_type(),
                                name: t.symbol().name().to_string(),
                                value: t.value().clone().unwrap_or_else(|| String::new()),
                            })
                            .collect::<Vec<_>>();
                        result.reverse();
                        result
                    })
                    .collect(),
            );

            break;
        }
    }

    Ok(FixTaskResult {
        time_before_load,
        time_after_load,
        time_after_reachability_built,
        time_after_find,
        found_length,
        outputs: outputs.unwrap_or_else(|| Vec::new()),
    })
}

mod do_fix_inner {
    pub trait DoFixInner {}
    impl DoFixInner for super::DoFixImpl {}
}

pub trait DoFix: do_fix_inner::DoFixInner {
    fn do_fix<'a, 'b, GProc, SProc, PG, T, E, G>(
        self,
        grammar: &'a Grammar<'a>,
        tokens: &Vec<Token<'a, '_>>,
        weight_generator: &G,
        gproc: &GProc,
        sproc: &SProc,
        info: &FixTaskInfo,
        time_before_load: Instant,
    ) -> Result<FixTaskResult, FixTaskError<T, E>>
    where
        PG: UnionProp,
        GProc: GProcessor<PG = PG>,
        SProc: SProcessor<PG = PG>,
        T: Error,
        E: Error,
        G: EdgeWeightGenerator<'a, 'b>;
}

pub struct DoFixImpl;

impl DoFix for DoFixImpl {
    fn do_fix<'a, 'b, GProc, SProc, PG, T, E, G>(
        self,
        grammar: &'a Grammar<'a>,
        tokens: &Vec<Token<'a, '_>>,
        weight_generator: &G,
        gproc: &GProc,
        sproc: &SProc,
        info: &FixTaskInfo,
        time_before_load: Instant,
    ) -> Result<FixTaskResult, FixTaskError<T, E>>
    where
        PG: UnionProp,
        GProc: GProcessor<PG = PG>,
        SProc: SProcessor<PG = PG>,
        T: Error,
        E: Error,
        G: EdgeWeightGenerator<'a, 'b>,
    {
        do_fix_impl(
            grammar,
            tokens,
            weight_generator,
            gproc,
            sproc,
            info,
            time_before_load,
        )
    }
}

pub trait FixingInputProcessorBase {
    fn info(&self) -> &FixingInfo;
}

pub trait FixingInputProcessor: FixingInputProcessorBase {
    fn process<'a>(
        &self,
        grammar: &'a Grammar<'a>,
        input_str: &str,
        env_str: &str,
        weights_str: Option<&str>,
        info: &FixTaskInfo,
        time_before_load: Instant,
        do_fix: impl DoFix,
    ) -> Result<FixTaskResult, FixTaskError<Self::TokenizerError, Self::EnvLoadError>>;

    type TokenizerError: Error;
    type EnvLoadError: Error;
}

fn fix_in_loop<'a, P, T, E>(
    processor: &P,
    info: &FixTaskInfo,
    grammar: &'a Grammar<'a>,
) -> Result<FixTaskResult, FixTaskError<T, E>>
where
    P: FixingInputProcessor<TokenizerError = T, EnvLoadError = E>,
    T: Error,
    E: Error,
{
    let time_before_load = Instant::now();

    let input = fs::read_to_string(info.input_name.as_str())
        .map_err(|e| FixTaskError::ReadInputError(e))?;
    let env =
        fs::read_to_string(info.env_name.as_str()).map_err(|e| FixTaskError::ReadEnvError(e))?;
    let weights = match info.weights {
        Some(ref weights) => Some(
            fs::read_to_string(weights.as_str()).map_err(|e| FixTaskError::ReadWeightsError(e))?,
        ),
        None => None,
    };

    processor.process(
        &grammar,
        input.as_str(),
        env.as_str(),
        weights.as_ref().map(|x| x.as_str()),
        &info,
        time_before_load,
        DoFixImpl,
    )
}

pub fn fix<P, T, E>(
    inputs: impl Iterator<Item = FixTaskInfo>,
    processor: &P,
) -> Vec<Result<FixTaskResult, FixTaskError<T, E>>>
where
    P: FixingInputProcessor<TokenizerError = T, EnvLoadError = E>,
    T: Error,
    E: Error,
{
    let grammar_arena = GrammarArena::new();
    let grammar = Grammar::new(&grammar_arena, processor.info().grammar).unwrap();
    let mut result = Vec::new();
    for info in inputs {
        let r = fix_in_loop(processor, &info, &grammar);
        match r {
            Ok(ref r) => {
                let time_load = r.time_after_load - r.time_before_load;
                let time_load = time_load.as_secs_f64();
                let time_build = r.time_after_reachability_built - r.time_after_load;
                let time_build = time_build.as_secs_f64();
                let time_find = match r.time_after_find.last() {
                    Some(x) => x.clone() - r.time_after_reachability_built,
                    None => Duration::new(0, 0),
                };
                let time_find = time_find.as_secs_f64();
                println!(
                    "---RESULT---,input_name:{},length:{},time_load:{},time_build:{},time_find:{}",
                    info.input_name,
                    match r.found_length {
                        Some(l) => l.to_string(),
                        None => "-1".to_string(),
                    },
                    time_load,
                    time_build,
                    time_find,
                );
                if let Some(output) = info.output_name {
                    let output = File::create(output).unwrap();
                    serde_json::to_writer(output, &r.outputs).unwrap();
                }
            }
            Err(ref e) => {
                println!("---RESULT---,input_name:{},error:{:?}", info.input_name, e)
            }
        }

        result.push(r);
    }
    result
}
