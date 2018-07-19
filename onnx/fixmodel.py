"""onnx model fixer

This utility will fix up common model errors.

1. Add a domain to the model object if there isn't one.
2. Fix names of nodes, graphs, values, etc. to conform to the standard.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import onnx
from onnx import defs
from onnx.defs import OpSchema
import onnx.shape_inference

import argparse

def replaceInvalidChars(name):

    global modified

    if name == '':
        return name

    n = name

    if name[0].isnumeric():
        n = "_" + n
    n = n.replace("/","_").replace("-","_")

    if n != name:
        modified = True

    return n

def examine(model, domain):

    global modified

    modified = False

    if model.domain == None or model.domain == '':
        model.domain = domain
        modified = True

    model.graph.name = replaceInvalidChars(model.graph.name)

    for node in model.graph.node:
        if node.name != None or node.name != '':
            node.name = replaceInvalidChars(node.name)        

        for vi in range(len(node.input)):
            if node.input[vi] != None or node.input[vi] != '':
                node.input[vi] = replaceInvalidChars(node.input[vi])        
        for vi in range(len(node.output)):
            if node.output[vi] != None or node.output[vi] != '':
                node.output[vi] = replaceInvalidChars(node.output[vi])        
    
    for vi in model.graph.initializer:
        if vi.name != None or vi.name != '':
            vi.name = replaceInvalidChars(vi.name)

    for vi in model.graph.value_info:
        if vi.name != None or vi.name != '':
            vi.name = replaceInvalidChars(vi.name)        

    for vi in model.graph.input:
        if vi.name != None or vi.name != '':
            vi.name = replaceInvalidChars(vi.name)        

    for vi in model.graph.output:
        if vi.name != None or vi.name != '':
            vi.name = replaceInvalidChars(vi.name)        

    return modified

def main():  # type: () -> None
    parser = argparse.ArgumentParser(
        description='Generates .proto file variations from .in.proto')
    parser.add_argument('-d', '--domain', default='org.tempuri.onnx',
                        help='domain name to use if missing in the model'
                        ' (default: %(default)s)')
    parser.add_argument('-o', '--output',
                        default=os.path.realpath(os.path.curdir),
                        help='output directory (default: %(default)s)')
    parser.add_argument('files', nargs='*',
                        help='list of ONNX files ')
    args = parser.parse_args()

    for file in args.files:
        m = onnx.load(file)
        print('\n==== Examining ' + file + ' ====\n')
        outpath = args.output + os.path.sep + file.replace(".onnx", ".new.onnx")
        if examine(m, args.domain):
            onnx.save(m, outpath)
            print('Wrote modified file to: ' + outpath)
        else:
            print('No modifications were made.')

    print('\n')

if __name__ == '__main__':
    main()