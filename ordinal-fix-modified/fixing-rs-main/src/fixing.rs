use crate::{
    c::fixing::CFixingInputProcessor, grammars::SupportedGrammar,
    mj::fixing::MJFixingInputProcessor,
};
use clap::{Parser, Subcommand};
use clap_num::number_range;
use csv::Reader;
use fixing_rs_base::fixing::{fix, FixTaskInfo};
use std::ffi::OsString;

fn gen_num_parser(s: &str) -> Result<usize, String> {
    number_range(s, 1, usize::MAX)
}

#[derive(Parser, Clone)]
pub struct FixCmd {
    #[arg(long, value_enum)]
    lang: SupportedGrammar,
    #[arg(long)]
    max_len: usize,
    #[arg(long)]
    max_new_id: usize,
    #[arg(long, default_value_t = 1, value_parser=gen_num_parser)]
    gen_num: usize,
    #[arg(long)]
    verbose_gen: bool,
    #[arg(long)]
    dump_cnf: Option<OsString>,
    #[arg(long)]
    solver_timeout: Option<u64>,
    #[command(subcommand)]
    files: CmdFiles,
}

#[derive(Subcommand, Clone)]
pub enum CmdFiles {
    Single {
        #[arg(long)]
        input: String,
        #[arg(long)]
        env: String,
        #[arg(long)]
        output: Option<String>,
        #[arg(long)]
        weights: Option<String>,
    },
    Multiple {
        #[arg(long)]
        file_list: String,
    },
}

impl FixCmd {
    pub fn run(self) {
        let files = Self::to_files(
            self.max_len,
            self.max_new_id,
            self.files,
            self.verbose_gen,
            self.gen_num,
        );
        match self.lang {
            SupportedGrammar::MJ => {
                fix(files, &MJFixingInputProcessor);
            }
            SupportedGrammar::C => {
                fix(files, &CFixingInputProcessor);
            }
        };
    }

    pub fn to_files(
        max_len: usize,
        max_new_id: usize,
        files: CmdFiles,
        verbose_gen: bool,
        gen_num: usize,
    ) -> impl Iterator<Item = FixTaskInfo> {
        let mut inputs = Vec::new();
        match files {
            CmdFiles::Single {
                input,
                env,
                output,
                weights,
            } => {
                inputs.push(FixTaskInfo {
                    input_name: input,
                    env_name: env,
                    output_name: output,
                    weights,
                    max_len,
                    max_new_id,
                    verbose_gen,
                    output_nums: gen_num,
                });
            }
            CmdFiles::Multiple { file_list } => {
                let input_file = std::fs::File::open(file_list).unwrap();
                let mut reader = Reader::from_reader(input_file);
                for item in reader.records() {
                    let item = item.unwrap();
                    if item.is_empty() {
                        continue;
                    }
                    let len = item.len();
                    if len != 2 && len != 3 && len != 4 {
                        panic!("Input file wrong record length.");
                    }
                    let input = item.get(0).unwrap().to_string();
                    let env = item.get(1).unwrap().to_string();
                    let output = match item.get(2) {
                        Some(x) => {
                            if x.len() != 0 {
                                Some(x.to_string())
                            } else {
                                None
                            }
                        }
                        None => None,
                    };
                    let weights = match item.get(3) {
                        Some(x) => {
                            if x.len() != 0 {
                                Some(x.to_string())
                            } else {
                                None
                            }
                        }
                        None => None,
                    };
                    inputs.push(FixTaskInfo {
                        input_name: input,
                        env_name: env,
                        output_name: output,
                        weights,
                        max_len,
                        max_new_id,
                        verbose_gen,
                        output_nums: gen_num,
                    });
                }
            }
        }
        inputs.into_iter()
    }
}
