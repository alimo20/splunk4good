#!/usr/bin/env python

import re


def is_truthy(s):
    return str(s).lower() in [
        '1', 't', 'true', 'y', 'yes', 'enable', 'enabled'
    ]


def is_falsy(s):
    return str(s).lower() in [
        '0', 'f', 'false', 'n', 'no', 'disable', 'disabled'
    ]


def booly(s):
    if is_truthy(s):
        return True
    elif is_falsy(s):
        return False

    raise ValueError('Failed to convert "%s" to a boolean value' % str(s))


def unquote_arg(arg):
    if len(arg) > 0 and (arg[0] == "'" or arg[0] == '"') and arg[0] == arg[-1]:
        return arg[1:-1]
    return arg


def convert_params(params, floats=[], ints=[], strs=[], bools=[], aliases={}, ignore_extra=False):
    out_params = {}
    for p in params:
        op = aliases.get(p, p)
        if p in floats:
            try:
                out_params[op] = float(params[p])
            except:
                raise ValueError("Invalid value for %s: must be a float" % p)
        elif p in ints:
            try:
                out_params[op] = int(params[p])
            except:
                raise ValueError("Invalid value for %s: must be an int" % p)
        elif p in strs:
            out_params[op] = str(unquote_arg(params[p]))
            if len(out_params[op]) == 0:
                raise ValueError("Invalid value for %s: must be a non-empty string" % p)
        elif p in bools:
            try:
                out_params[op] = booly(params[p])
            except ValueError:
                raise ValueError("Invalid value for %s: must be a boolean" % p)
        elif not ignore_extra:
            raise ValueError("Unexpected parameter: %s" % p)

    return out_params


def parse_args(argv):
    options = {}

    from_seen = False

    params_re = re.compile("([_a-zA-Z][_a-zA-Z0-9]*)\s*=\s*(.*)")
    while argv:
        arg = argv.pop(0)
        if arg.lower() == 'into':
            if 'model_name' in options:
                raise ValueError('Syntax error: you may specify "into" only once')

            try:
                options['model_name'] = unquote_arg(argv.pop(0))
                assert len(options['model_name']) > 0
            except:
                raise ValueError('Syntax error: "into" keyword requires argument')
        elif arg.lower() == 'by':
            if 'split_by' in options:
                raise ValueError('Syntax error: you may specify "by" only once')

            try:
                options['split_by'] = unquote_arg(argv.pop(0))
                assert len(options['split_by']) > 0
            except:
                raise ValueError('Syntax error: "by" keyword requires argument')
        elif arg.lower() == 'as':
            if 'output_name' in options:
                raise ValueError('Syntax error: you may specify "as" only once')

            try:
                options['output_name'] = unquote_arg(argv.pop(0))
                assert len(options['output_name']) > 0
            except:
                raise ValueError('Syntax error: "as" keyword requires argument')
        elif arg.lower() == 'from' or arg == "~":
            if from_seen:
                raise ValueError('Syntax error: you may specify "from" only once')

            from_seen = True
            continue
        else:
            m = params_re.match(arg)
            if m:
                params = options.setdefault('params', {})
                params[m.group(1)] = m.group(2)
            else:
                arg = unquote_arg(arg)
                if len(arg) == 0: continue
                args = options.setdefault('args', [])
                args.append(arg)

                if from_seen:
                    variables = options.setdefault('explanatory_variables', [])
                else:
                    variables = options.setdefault('variables', [])
                    if isinstance(arg, unicode):
                        arg = arg.encode('utf-8')

                variables.append(arg)

    return options
