import csv
import datetime
import logging
import os
import sys
import yaml

import Lexer
import Parser
import LLVMGenerator


class Compiler(object):

    def __init__(self, input_file, output_file, emit_ast, 
                       emit_llvm, emit_tokens, jit, use_optimizations,
                       emit_stats, extra_args):
        self.input_file = input_file
        self.output_file = output_file
        self.emit_ast = emit_ast
        self.emit_llvm = emit_llvm
        self.emit_tokens = emit_tokens
        self.jit = jit
        self.use_optimizations = use_optimizations
        self.emit_stats = emit_stats
        self.extra_args = extra_args

    def compile(self):
        """
        Entry point for compilation process
        """
        logging.info('Parsing file: ' + self.input_file)
        l = Lexer.Lexer()
        data = self._get_input_data()

        lexer = l.get_lexer()
        if self.emit_tokens:
            self.write_tokens(lexer, data)

        parser = Parser.Parser()
        tree = parser.parse(data, lexer)

        if self.emit_ast:
            self.write_ast(tree)

        generator = LLVMGenerator.LLVMGenerator()
        if self.emit_llvm or self.jit:
            self.write_llvm(generator, tree)

        if self.emit_stats:
            self.write_stats(l, parser, generator)

    def _get_input_data(self):
        """
        Reads input EK text file 

        return: str 
        """
        with open(self.input_file, 'r') as f:
            return f.read()

    def write_tokens(self, lexer, data):
        """
        Writes tokens to an output file
        """
        lexer.input(data)
        with open(self.output_file, 'w') as f:
            for token in lexer:
                f.write(str(token) + '\n')

    def write_ast(self, tree):
        """
        Outputs abstract syntax tree to an output file
        """
        logging.info('Writing output to: ' + self.output_file)

        # The yaml library requires this to format neatly
        def noop(self, *args, **kw):
            pass

        yaml.emitter.Emitter.process_tag = noop

        with open(self.output_file, 'w') as f:
            yaml.dump(tree, f, default_flow_style=False)

    def write_llvm(self, generator, tree):
        """
        Generates the LLVM intermediate representation
        """
        logging.info('Creating LLVM IR')
        generator.generate(tree, self.emit_llvm, self.jit, self.use_optimizations,
                           self.extra_args, self.input_file)

    def write_stats(self, lexer, parser, generator):
        """
        Records statistics of JIT runtime in order to measure performance optimizations
        """
        header = ['Lex', 'Parse', 'LLVM', 'Optimize', 'Run']
        microsecond = datetime.timedelta(microseconds=1)
        opt_time = 0 if generator.optimize_time is None else (generator.optimize_time / microsecond)

        # Use input filename as a key
        key = os.path.basename(self.input_file).replace('.ek', '')

        # Use given optimizations as part of key
        if type(self.use_optimizations) == list:
            key += '_O_'
            if len(self.use_optimizations) > 0:
                key += '_'.join(self.use_optimizations)

        output = [key
                , lexer.lex_time / microsecond
                , parser.parse_time / microsecond
                , generator.llvm_time / microsecond
                , opt_time
                , generator.run_time / microsecond]

        with open('../stats/stats.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(output)

        logging.info('Lex time:', lexer.lex_time)
        logging.info('Parse time:', parser.parse_time)
        logging.info('LLVM time:', generator.llvm_time)
        logging.info('Opt time:', generator.optimize_time)
        logging.info('Run time:', generator.run_time)
