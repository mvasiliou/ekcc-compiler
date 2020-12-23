import logging

import Lexer
import Parser
import LLVMGenerator
import yaml
import sys
import os


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

    def _get_input_data(self):
        with open(self.input_file, 'r') as f:
            return f.read()

    def write_lexer_output(self):
        # This method is not actually called, it is for outputting the set of
        # tokens when debugging
        l = Lexer.Lexer()
        lexer = l.get_lexer()
        data = self._get_input_data()

        with open('lexer.out', 'w') as f:
            lexer.input(data)

            while True:
                token = lexer.token()
                if not token: 
                    break
                f.write(str(token) + '\n')

    def compile(self):
        logging.info('Parsing file: ' + self.input_file)
        self.write_lexer_output()
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

    def write_ast(self, tree):
        logging.info('Writing output to: ' + self.output_file)
        def noop(self, *args, **kw):
            pass

        yaml.emitter.Emitter.process_tag = noop

        with open(self.output_file, 'w') as f:
            yaml.dump(tree, f, default_flow_style=False)

    def write_llvm(self, generator, tree):
        logging.info('Creating LLVM IR')
        generator.generate(tree, self.emit_llvm, self.jit, self.use_optimizations,
                   self.extra_args, self.input_file)

    def write_tokens(self, lexer, data):
        lexer.input(data)
        with open(self.output_file, 'w') as f:
            for token in lexer:
                f.write(str(token) + '\n')

    def write_stats(self, lexer, parser, generator):
        import csv
        import datetime
        header = ['Lex', 'Parse', 'LLVM', 'Optimize', 'Run']
        
        microsecond = datetime.timedelta(microseconds=1)
        opt_time = 0 if generator.optimize_time is None else (generator.optimize_time / microsecond)

        key = os.path.basename(self.input_file).replace('.ek', '')

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

        print('Lex time:', lexer.lex_time)
        print('Parse time:', parser.parse_time)
        print('LLVM time:', generator.llvm_time)
        print('Opt time:', generator.optimize_time)
        print('Run time:', generator.run_time)
