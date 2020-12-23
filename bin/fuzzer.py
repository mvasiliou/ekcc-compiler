import argparse
import logging
import Compiler
import afl

afl.init()

def main():
    parser = argparse.ArgumentParser(description='Extended Kaleidoscope Compiler')
    parser.add_argument('-?', dest='show_help', default=False, action='store_true')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', 
        help='Turn on verbose logging')

    parser.add_argument('--emit-ast', default=False, action='store_true')
    parser.add_argument('--emit-llvm', default=False, action='store_true')
    parser.add_argument('-O', dest='use_optimizations', default=False, 
                        action='store_true', help='Enable optimizations')
    parser.add_argument('-o', dest='output_file', help='Output file path',
                        default='output.yaml')
    parser.add_argument('input_file', help='Source file to compile')


    args = parser.parse_args()

    if args.show_help:
        parser.print_help()
    
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level,
                        format='%(asctime)s [%(levelname)s]: %(message)s',
                        datefmt='%Y%m%d %H:%M:%S')

    c = Compiler.Compiler(args.input_file, args.output_file, args.emit_ast, 
                          args.emit_llvm, args.use_optimizations)
    c.compile()


if __name__ == "__main__":
    main()


