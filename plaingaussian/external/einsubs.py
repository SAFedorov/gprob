#!/usr/bin/env python
# coding: utf-8
#
# SAF: This is a reduced version the subscript parser from opt-einsum package
# https://github.com/dgasmith/opt_einsum.
#
# The original can found at
# https://optimized-einsum.readthedocs.io/en/stable/_modules/opt_einsum/parser.html
#
#
# Copyright (c) 2014 Daniel Smith


_einsum_symbols_base = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def get_symbol(i):
    """Get the symbol corresponding to int ``i`` - runs through the usual 52
    letters before resorting to unicode characters, starting at ``chr(192)``.

    Examples
    --------
    >>> get_symbol(2)
    'c'

    >>> get_symbol(200)
    'Ŕ'

    >>> get_symbol(20000)
    '京'
    """
    if i < 52:
        return _einsum_symbols_base[i]
    
    return chr(i + 140)


def gen_unused_symbols(used, n):
    """Generate ``n`` symbols that are not already in ``used``.

    Examples
    --------
    >>> list(gen_unused_symbols("abd", 2))
    ['c', 'e']
    """
    i = cnt = 0
    while cnt < n:
        s = get_symbol(i)
        i += 1
        if s in used:
            continue
        yield s
        cnt += 1


def find_output_str(subscripts):
    """
    Find the output string for the inputs ``subscripts`` under canonical 
    einstein summation rules. That is, repeated indices are summed over 
    by default.

    Examples
    --------
    >>> find_output_str("ab,bc")
    'ac'

    >>> find_output_str("a,b")
    'ab'

    >>> find_output_str("a,a,b,b")
    ''
    """
    tmp_subscripts = subscripts.replace(",", "")
    return "".join(s for s in sorted(set(tmp_subscripts)) if tmp_subscripts.count(s) == 1)


def parse(subscripts, shapes):
    """
    Returns
    -------
    input_strings : list
        List of parsed input strings
    output_string : str
        Parsed output string

    Examples
    --------
    The operand list is simplified to reduce printing:

    >>> a = np.random.rand(4, 4)
    >>> b = np.random.rand(4, 4, 4)
    >>> parse('...a,...a->...', (a.shape, b.shape))
    (['da', 'cda'], 'cd')
    """

    if not isinstance(subscripts, str):
        raise ValueError("Subscripts must be a string.")

    subscripts = subscripts.replace(" ", "")

    # Check for proper "->"
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    # Parse ellipses
    if "." in subscripts:
        used = subscripts.replace(".", "").replace(",", "").replace("->", "")
        ellipse_inds = "".join(gen_unused_symbols(used, max(len(x) for x in shapes)))
        longest = 0

        # Do we have an output to account for?
        if "->" in subscripts:
            input_tmp, output_sub = subscripts.split("->")
            split_subscripts = input_tmp.split(",")
            out_sub = True
        else:
            split_subscripts = subscripts.split(',')
            out_sub = False

        for num, sub in enumerate(split_subscripts):
            if "." in sub:
                if (sub.count(".") != 3) or (sub.count("...") != 1):
                    raise ValueError("Invalid Ellipses.")

                # Take into account numerical values
                if shapes[num] == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(len(shapes[num]), 1) - (len(sub) - 3)

                if ellipse_count > longest:
                    longest = ellipse_count

                if ellipse_count < 0:
                    raise ValueError("Ellipses lengths do not match.")
                elif ellipse_count == 0:
                    split_subscripts[num] = sub.replace('...', '')
                else:
                    split_subscripts[num] = sub.replace('...', ellipse_inds[-ellipse_count:])

        subscripts = ",".join(split_subscripts)

        # Figure out output ellipses
        if longest == 0:
            out_ellipse = ""
        else:
            out_ellipse = ellipse_inds[-longest:]

        if out_sub:
            subscripts += "->" + output_sub.replace("...", out_ellipse)
        else:
            # Special care for outputless ellipses
            output_subscript = find_output_str(subscripts)
            normal_inds = ''.join(sorted(set(output_subscript) - set(out_ellipse)))

            subscripts += "->" + out_ellipse + normal_inds

    # Build output string if does not exist
    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")
    else:
        input_subscripts, output_subscript = subscripts, find_output_str(subscripts)

    # Make sure output subscripts are in the input
    for char in output_subscript:
        if char not in input_subscripts:
            raise ValueError(f"Output character '{char}' did not appear in the input")

    return input_subscripts.split(','), output_subscript