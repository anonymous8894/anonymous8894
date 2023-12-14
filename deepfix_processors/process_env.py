import multiprocessing
import sys

import tqdm

import process_all
import processed_loader
import pycparser

KNOWN_NAMES = {
    "int",
    "short",
    "long",
    "signed",
    "unsigned",
    "char",
    "float",
    "double",
    "void",
}


def names_to_types_no_const(names):
    if "void" in names:
        return "void"
    if "float" in names or "double" in names:
        return "float"
    return "int"


def names_to_type(names):
    names = set(names)
    for name in names:
        if name not in KNOWN_NAMES:
            raise Exception("Unknown type name: {}".format(name))
    base = names_to_types_no_const(names)
    if "const" in names:
        return f"{base}.const"
    else:
        return base


def add_qual(qual, appending):
    if "const" in qual:
        appending.append("const")


def get_var_decl_type_inner(decl, appending):
    c_ast = pycparser.c_parser.c_ast
    if isinstance(decl, c_ast.FuncDecl):
        raise Exception("FuncDecl not supported")
    elif isinstance(decl, c_ast.TypeDecl):
        add_qual(decl.quals, appending)
        get_var_decl_type_inner(decl.type, appending)
    elif isinstance(decl, c_ast.IdentifierType):
        names = decl.names
        names = names_to_type(names)
        appending.append(names)
    elif isinstance(decl, c_ast.ArrayDecl):
        appending.append("[]")
        get_var_decl_type_inner(decl.type, appending)
    elif isinstance(decl, c_ast.PtrDecl):
        appending.append("*")
        add_qual(decl.quals, appending)
        get_var_decl_type_inner(decl.type, appending)
    elif isinstance(decl, c_ast.Struct):
        appending.append("void")
    else:
        raise Exception("Unknown type: {}".format(decl))


def get_var_decl_type(decl):
    appending = []
    get_var_decl_type_inner(decl, appending)
    return ".".join(appending[::-1])


def process_param(param):
    c_ast = pycparser.c_parser.c_ast
    if isinstance(param, c_ast.Decl):
        param_name = param.name
        param_type = get_var_decl_type(param.type)
        return f"{param_name}:{param_type}"
    elif isinstance(param, c_ast.EllipsisParam):
        return f"..."
    elif isinstance(param, c_ast.Typename):
        param_type = get_var_decl_type(param.type)
        return f":{param_type}"
    elif isinstance(param, c_ast.ID):
        param_name = param.name
        return f"{param_name}:int"
    else:
        raise Exception(f"not decl: {param}")


def process_decl(decl, outputs):
    c_ast = pycparser.c_parser.c_ast
    if isinstance(decl.type, c_ast.FuncDecl):
        func_name = decl.name
        func_type = decl.type
        func_ret = get_var_decl_type(func_type.type)
        func_args = (
            map(process_param, func_type.args.params)
            if func_type.args is not None
            else []
        )
        func_args = ",\n".join(func_args)
        out_name = func_name
        out = f"=FN {func_name}:{func_ret}-\n{func_args};\n"
    else:
        var_name = decl.name
        var_type = get_var_decl_type(decl.type)
        out_name = var_name
        out = f"=VAR {var_name}:{var_type};\n"
    if out_name in outputs[1]:
        outputs[0][outputs[1][out_name]] = ""
    outputs[1][out_name] = len(outputs[0])
    outputs[0].append(out)


def processfile(env_file):
    code_text = process_all.preprocess_file(env_file, None)
    if isinstance(code_text, str):
        raise Exception("Failed to process env file: {}".format(env_file))
    code_text = "".join(code_text)
    parser = pycparser.CParser()
    ast = parser.parse(code_text)
    c_ast = pycparser.c_parser.c_ast
    outputs = [], {}
    for ext in ast.ext:
        try:
            if isinstance(ext, c_ast.Typedef):
                pass
            elif isinstance(ext, c_ast.FuncDef):
                process_decl(ext.decl, outputs)
            elif isinstance(ext, c_ast.Decl):
                process_decl(ext, outputs)
            else:
                raise Exception(f"Unknown type: {type(ext)}")
        except Exception:
            print(env_file)
            print(ext)
            raise
    with open(f"{env_file}.p", "w") as fout:
        for o in outputs[0]:
            fout.write(o)
            fout.write("\n")


def main(envs=None):
    if envs is None:
        envs = [x.env_file for x in processed_loader.get_running_samples()]
    pool = multiprocessing.Pool(process_all.NUM_PROCESSES)
    t = tqdm.tqdm(pool.imap_unordered(processfile, envs), total=len(envs))
    for _ in t:
        pass


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    else:
        main(sys.argv[1:])
