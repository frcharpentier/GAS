import inspect
import sys
import ast
import traceback
import types
import itertools
from functools import partial, update_wrapper
import re


DEFAULT_LOOP_LABEL = None

# have to make our own partial in case someone wants to use reloading as a iterator without any arguments
# they would get a partial back because a call without a iterator argument is assumed to be a decorator.
# getting a "TypeError: 'functools.partial' object is not iterable"
# which is not really descriptive.
# hence we overwrite the iter to make sure that the error makes sense.
class no_iter_partial(partial):
    def __iter__(self):
        raise TypeError("Nothing to iterate over. Please pass an iterable to reloading.")
    

class BreakOutOfLoopException(Exception):
    def __init__(self, label, cont=False):
        self.label = label
        self.cont = True if cont else False


def break_loop(label):
    raise BreakOutOfLoopException(label, False)

def continue_loop(label):
    raise BreakOutOfLoopException(label, True)

class reload_ctxt:
    def __init__(self, label=None):
        self.label = None

    def __enter__(self):
        pass

    def __exit__(self):
        pass

def isolate_labeled_loop(tree, loop_label, nom_breakBool, nom_contBool, nom_breakExcept):
    """Modifies tree inplace as unclear how to create ast.Module.
    Returns itervars"""
    candidate_nodes = []
    for node in ast.walk(tree):
        #if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
        #    fonction = node.iter.func
        #    fonction = ast.Expression(body=fonction)
        #    fonction = compile(fonction, filename="", mode="eval")
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "reload_loop"):

            labels = [X.value for X in node.iter.keywords if X.arg == "label" and isinstance(X.value, ast.Constant)]
            labels = [X.value for X in labels]
            labels = [X for X in labels if X==loop_label]

            if len(labels) == 1:
                candidate_nodes.append(node)

    if len(candidate_nodes) > 1:
        raise LookupError(
            "The reloading loop is ambiguous. Use distinct labels to name each reloading loop."
        )

    if len(candidate_nodes) < 1:
        raise LookupError(
            "Could not locate reloading loop. The loop’s label disappeared"
        )

    bloc  = """while True:
    if {n_cont_bool:s}:
        {n_broken_bool:s} = False
        break
    {n_cont_bool:s}   = True
    # insérer code ici
    {n_broken_bool:s} = False
    break
if {n_broken_bool:s}:
    raise {n_excep:s}({n_label:s}, False)
elif {n_cont_bool:s}:
    raise {n_excep:s}({n_label:s}, True)"""
    
    bloc = bloc.format(n_cont_bool=nom_contBool, n_broken_bool=nom_breakBool, n_excep=nom_breakExcept, n_label=repr(loop_label))
    tree_1 = ast.parse(bloc)

    loop_node = candidate_nodes[0]
    body = [X for X in tree_1.body[0].body[:2]]
    body.extend([X for X in loop_node.body])
    body.extend([X for X in tree_1.body[0].body[2:]])
    tree_1.body[0].body = body
    return tree_1, loop_node.target

def get_loop_code_break(fpath, loop_label, nom_breakBool, nom_contBool, nom_breakExcept):
        
    while True:
        tree = parse_file_until_successful(fpath)
        try:
            tree_1, itervars = isolate_labeled_loop(tree, loop_label, nom_breakBool, nom_contBool, nom_breakExcept)
            
            return compile(tree_1, filename=fpath, mode="exec"), format_itervars(itervars)
        except LookupError:
            handle_exception(fpath)


def isolate_labeled_cond(tree, loop_label):
    candidate_nodes = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Name)
            and node.test.func.id == "reload_cond"):

            labels = [X.value for X in node.test.keywords if X.arg == "label" and isinstance(X.value, ast.Constant)]
            labels = [X.value for X in labels]
            labels = [X for X in labels if X==loop_label]

            if len(labels) == 1:
                candidate_nodes.append(node)

    if len(candidate_nodes) > 1:
        raise LookupError(
            "The reloading cond is ambiguous. Use distinct labels to name each reloading cond."
        )

    if len(candidate_nodes) < 1:
        raise LookupError(
            "Could not locate reloading cond. The loop’s label disappeared"
        )

    tree_1 = ast.parse("a=8\nb=9")

    cond_node = candidate_nodes[0]
    tree_1.body = cond_node.body
    return tree_1


def get_cond_code(fpath, loop_label):
    
    while True:
        tree = parse_file_until_successful(fpath)
        try:
            tree_1 = isolate_labeled_cond(tree, loop_label)
            return compile(tree_1, filename=fpath, mode="exec")
        except LookupError:
            handle_exception(fpath)


def reload_cond(label=None):
    if label is None:
        return True
    
    try:
        cond_frame_info = inspect.stack()[1] #Dans cette copie de la fonction _reloading_loop(), la profondeur d’appel n’est pas la même
        fpath = cond_frame_info.filename
        #numlig = cond_frame_info.lineno
        frame = cond_frame_info.frame
        caller_globals = frame.f_globals
        caller_locals = frame.f_locals
    finally:
        del frame
        del cond_frame_info

    # create a unique name in the caller namespace that we can safely write
    # the values of the iteration variables into
    nouveaux_noms = unique_name(itertools.chain(caller_locals.keys(), caller_globals.keys()))
    #nom_breakBool = nouveaux_noms + "_break_bool"
    #nom_breakExcept = nouveaux_noms + "_break_except"

    #caller_locals[nom_breakExcept] = BreakOutOfLoopException
    #caller_locals[nom_breakBool] = True
    compiled_body = get_cond_code(fpath, label)  #, nom_breakBool, nom_breakExcept)
    try:
        # run conditional body
        exec(compiled_body, caller_globals, caller_locals)
    except Exception as excep:
        if isinstance(excep, BreakOutOfLoopException):
            raise excep
        else:
            handle_exception(fpath)
            
    return False

def reload_loop(seq=None, every=1, forever=None, label=None):
    global DEFAULT_LOOP_LABEL
    if seq is None:
        assert forever == True
        seq = itertools.count(0)
    if label is None:
        return seq
    try:
        loop_frame_info = inspect.stack()[1] #Dans cette copie de la fonction _reloading_loop(), la profondeur d’appel n’est pas la même
        fpath = loop_frame_info.filename
        numlig = loop_frame_info.lineno
        frame = loop_frame_info.frame
        caller_globals = frame.f_globals
        caller_locals = frame.f_locals
    finally:
        del frame
        del loop_frame_info

    # create a unique name in the caller namespace that we can safely write
    # the values of the iteration variables into
    nouveaux_noms = unique_name(itertools.chain(caller_locals.keys(), caller_globals.keys()))
    nom_itvalue = nouveaux_noms + "_itervar"
    nom_breakBool = nouveaux_noms + "_break_bool"
    nom_contBool  = nouveaux_noms + "_cont_bool"
    nom_breakExcept = nouveaux_noms + "_break_except"

    caller_locals[nom_breakExcept] = BreakOutOfLoopException

    old_loop_label = DEFAULT_LOOP_LABEL
    DEFAULT_LOOP_LABEL = label
    for i, itervar_values in enumerate(seq):
        if i % every == 0:
            compiled_body, itervars = get_loop_code_break(fpath, label, nom_breakBool, nom_contBool, nom_breakExcept)


        caller_locals[nom_itvalue] = itervar_values
        caller_locals[nom_breakBool] = True
        caller_locals[nom_contBool]  = False
        exec(itervars + " = " + nom_itvalue, caller_globals, caller_locals)
        try:
            # run main loop body
            exec(compiled_body, caller_globals, caller_locals)
        except BreakOutOfLoopException as excep:
            if excep.label == label and excep.cont:
                continue
            elif excep.label == label and not (excep.cont):
                break
            else:
                raise excep
        except Exception:
            handle_exception(fpath)
    
    del caller_locals[nom_itvalue]
    del caller_locals[nom_breakBool]
    del caller_locals[nom_contBool]
    del caller_locals[nom_breakExcept]

    DEFAULT_LOOP_LABEL = old_loop_label
    return []


def reloading(fn_or_seq=None, every=1, forever=None):
    """Wraps a loop iterator or decorates a function to reload the source code
    before every loop iteration or function invocation.

    When wrapped around the outermost iterator in a `for` loop, e.g.
    `for i in reloading(range(10))`, causes the loop body to reload from source
    before every iteration while keeping the state.
    When used as a function decorator, the decorated function is reloaded from
    source before each execution.

    Pass the integer keyword argument `every` to reload the source code
    only every n-th iteration/invocation.

    Args:
        fn_or_seq (function | iterable): A function or loop iterator which should
            be reloaded from source before each invocation or iteration,
            respectively
        every (int, Optional): After how many iterations/invocations to reload
        forever (bool, Optional): Pass `forever=true` instead of an iterator to
            create an endless loop

    """
    if fn_or_seq:
        if isinstance(fn_or_seq, types.FunctionType):
            return _reloading_function(fn_or_seq, every=every)
        return _reloading_loop(fn_or_seq, every=every)
    if forever:
        return _reloading_loop(iter(int, 1), every=every)

    # return this function with the keyword arguments partialed in,
    # so that the return value can be used as a decorator
    decorator = update_wrapper(no_iter_partial(reloading, every=every), reloading)
    return decorator


def unique_name(used):
    # get the longest element of the used names and append a "0"
    return max(used, key=len) + "0"


def format_itervars(ast_node):
    """Formats an `ast_node` of loop iteration variables as string, e.g. 'a, b'"""

    # handle the case that there only is a single loop var
    if isinstance(ast_node, ast.Name):
        return ast_node.id

    names = []
    for child in ast_node.elts:
        if isinstance(child, ast.Name):
            names.append(child.id)
        elif isinstance(child, ast.Tuple) or isinstance(child, ast.List):
            # if its another tuple, like "a, (b, c)", recurse
            names.append("({})".format(format_itervars(child)))

    return ", ".join(names)


def load_file(path):
    src = ""
    # while loop here since while saving, the file may sometimes be empty.
    while (src == ""):
        with open(path, "r") as f:
            src = f.read()
    return src + "\n"


def parse_file_until_successful(path):
    source = load_file(path)
    while True:
        try:
            tree = ast.parse(source)
            return tree
        except SyntaxError:
            handle_exception(path)
            source = load_file(path)


def isolate_loop_body_and_get_itervars(tree, lineno, loop_id):
    """Modifies tree inplace as unclear how to create ast.Module.
    Returns itervars"""
    candidate_nodes = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and node.iter.func.id == "reloading"
            and (
                    (loop_id is not None and loop_id == get_loop_id(node))
                    or getattr(node, "lineno", None) == lineno
                )
            ):
            candidate_nodes.append(node)

    if len(candidate_nodes) > 1:
        raise LookupError(
            "The reloading loop is ambigious. Use `reloading` only once per line and make sure that the code in that line is unique within the source file."
        )

    if len(candidate_nodes) < 1:
        raise LookupError(
            "Could not locate reloading loop. Please make sure the code in the line that uses `reloading` doesn't change between reloads."
        )

    loop_node = candidate_nodes[0]
    tree.body = loop_node.body
    return loop_node.target, get_loop_id(loop_node)





def get_loop_id(ast_node):
    """Generates a unique identifier for an `ast_node` of type ast.For to find the loop in the changed source file
    """
    return ast.dump(ast_node.target) + "__" + ast.dump(ast_node.iter)


def get_loop_code(loop_frame_info, loop_id):
    fpath = loop_frame_info.filename
    while True:
        tree = parse_file_until_successful(fpath)
        try:
            itervars, found_loop_id = isolate_loop_body_and_get_itervars(tree, lineno=loop_frame_info[2], loop_id=loop_id)
            return compile(tree, filename=fpath, mode="exec"), format_itervars(itervars), found_loop_id
        except LookupError:
            handle_exception(fpath)






def handle_exception(fpath):
    exc = traceback.format_exc()
    exc = exc.replace('File "<string>"', 'File "{}"'.format(fpath))
    sys.stderr.write(exc + "\n")
    print("Edit {} and press return to continue".format(fpath))
    sys.stdin.readline()


def _reloading_loop(seq, every=1):
    loop_frame_info = inspect.stack()[2]
    fpath = loop_frame_info.filename
    frame = loop_frame_info.frame
    caller_globals = frame.f_globals
    caller_locals = frame.f_locals

    # create a unique name in the caller namespace that we can safely write
    # the values of the iteration variables into
    nouveaux_noms = unique_name(itertools.chain(caller_locals.keys(), caller_globals.keys()))
    nom_itvalue = nouveaux_noms + "_itervar"
    #nom_breakBool = nouveaux_noms + "_break_bool"
    #nom_breakExcept = nouveaux_noms + "_break_except"

    #caller_locals[nom_breakExcept] = BreakOutOfLoopException
    loop_id = None

    for i, itervar_values in enumerate(seq):
        if i % every == 0:
            compiled_body, itervars, loop_id = get_loop_code(loop_frame_info, loop_id)

        caller_locals[nom_itvalue] = itervar_values
        #caller_locals[nom_breakBool] = True
        exec(itervars + " = " + nom_itvalue, caller_globals, caller_locals)
        try:
            # run main loop body
            exec(compiled_body, caller_globals, caller_locals)
        except BreakOutOfLoopException:
            break
        except Exception:
            handle_exception(fpath)

    return []


def get_decorator_name_or_none(dec_node):
    if hasattr(dec_node, "id"):
        return dec_node.id
    elif hasattr(dec_node.func, "id"):
        return dec_node.func.id
    elif hasattr(dec_node.func.value, "id"):
        return dec_node.func.value.id
    else:
        return None


def strip_reloading_decorator(func):
    """Remove the 'reloading' decorator and all decorators before it"""
    decorator_names = [get_decorator_name(dec) for dec in func.decorator_list]
    reloading_idx = decorator_names.index("reloading")
    func.decorator_list = func.decorator_list[reloading_idx + 1:]


def isolate_function_def(funcname, tree):
    """Strip everything but the function definition from the ast in-place.
    Also strips the reloading decorator from the function definition"""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == funcname
            and "reloading" in [
                get_decorator_name_or_none(dec)
                for dec in node.decorator_list
            ]
        ):
            strip_reloading_decorator(node)
            tree.body = [ node ]
            return True
    return False


def get_function_def_code(fpath, fn):
    tree = parse_file_until_successful(fpath)
    found = isolate_function_def(fn.__name__, tree)
    if not found:
        return None
    compiled = compile(tree, filename="", mode="exec")
    return compiled


def get_reloaded_function(caller_globals, caller_locals, fpath, fn):
    code = get_function_def_code(fpath, fn)
    if code is None:
        return None
    # need to copy locals, otherwise the exec will overwrite the decorated with the undecorated new version
    # this became a need after removing the reloading decorator from the newly defined version
    caller_locals_copy = caller_locals.copy()
    exec(code, caller_globals, caller_locals_copy)
    func = caller_locals_copy[fn.__name__]
    return func


def _reloading_function(fn, every=1):
    stack = inspect.stack()
    frame, fpath = stack[2][:2]
    caller_locals = frame.f_locals
    caller_globals = frame.f_globals

    # crutch to use dict as python2 doesn't support nonlocal
    state = {
        "func": None,
        "reloads": 0,
    }

    def wrapped(*args, **kwargs):
        if state["reloads"] % every == 0:
            state["func"] = get_reloaded_function(caller_globals, caller_locals, fpath, fn) or state["func"]
        state["reloads"] += 1
        while True:
            try:
                result = state["func"](*args, **kwargs)
                return result
            except Exception:
                handle_exception(fpath)
                state["func"] = get_reloaded_function(caller_globals, caller_locals, fpath, fn) or state["func"]

    caller_locals[fn.__name__] = wrapped
    return wrapped