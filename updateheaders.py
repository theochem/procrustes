# -*- coding: utf-8 -*-
# The Procrustes library provides a set of functions for transforming
# a matrix to make it as similar as possible to a target matrix.
#
# Copyright (C) 2017-2022 The QC-Devs Community
#
# This file is part of Procrustes.
#
# Procrustes is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Procrustes is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --


import os
from fnmatch import fnmatch
from glob import glob


def strip_header(lines, closing):
    # search for the header closing line, i.e. '# --\n'
    counter = 0
    found = False
    for line in lines:
        counter += 1
        if line == closing:
            found = True
            break
    if found:
        del lines[:counter]
        # If the header closing is not found, we assume it is not present.
    # add a header closing line
    lines.insert(0, closing)


def fix_python(fn, lines, header_lines):
    # check if a shebang is present
    do_shebang = lines[0].startswith('#!')
    # remove the current header
    strip_header(lines, '# --\n')
    # add a pylint line for test files:
    # if os.path.basename(fn).startswith('test_'):
    #     if not lines[1].startswith('#pylint: skip-file'):
    #         lines.insert(1, '#pylint: skip-file\n')
    # add new header (insert in reverse order)
    for hline in header_lines[::-1]:
        lines.insert(0, ('# ' + hline).strip() + '\n')
    # add a source code encoding line
    lines.insert(0, '# -*- coding: utf-8 -*-\n')
    if do_shebang:
        lines.insert(0, '#!/usr/bin/env python\n')


def fix_c(fn, lines, header_lines):
    # check for an exception line
    for line in lines:
        if 'no_update_headers' in line:
            return
    # remove the current header
    strip_header(lines, '//--\n')
    # add new header (insert must be in reverse order)
    for hline in header_lines[::-1]:
        lines.insert(0, ('// ' + hline).strip() + '\n')


def fix_rst(fn, lines, header_lines):
    # check for an exception line
    for line in lines:
        if 'no_update_headers' in line:
            return
    # remove the current header
    strip_header(lines, '    : --\n')
    # add an empty line after header if needed
    if len(lines[1].strip()) > 0:
        lines.insert(1, '\n')
    # add new header (insert must be in reverse order)
    for hline in header_lines[::-1]:
        lines.insert(0, ('    : ' + hline).rstrip() + '\n')
    # add comment instruction
    lines.insert(0, '..\n')


def iter_subdirs(root):
    for dn, _, _ in os.walk(root):
        yield dn


def main():
    source_dirs = ['.', 'docs', 'scripts', 'tools'] + \
                  list(iter_subdirs('procrustes'))

    fixers = [
        ('*.py', fix_python),
        ('*.pxd', fix_python),
        ('*.pyx', fix_python),
        ('*.txt', fix_python),
        ('*.c', fix_c),
        ('*.cpp', fix_c),
        ('*.h', fix_c),
        ('*.rst', fix_rst),
    ]

    f = open('HEADER')
    header_lines = f.readlines()
    f.close()

    for sdir in source_dirs:
        print('Scanning:', sdir)
        for fn in glob(sdir + '/*.*'):
            if not os.path.isfile(fn):
                continue
            for pattern, fixer in fixers:
                if fnmatch(fn, pattern):
                    print(' Fixing:', fn)
                    with open(fn) as f:
                        lines = f.readlines()
                    fixer(fn, lines, header_lines)
                    with open(fn, 'w') as f:
                        f.writelines(lines)
                    break


if __name__ == '__main__':
    main()
