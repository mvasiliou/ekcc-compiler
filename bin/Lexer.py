import logging
import ply.lex
import sys
import datetime


class Lexer(object):

    tokens = ['EXTERN'
            , 'DEF'
            , 'RETURN'
            , 'IF'
            , 'WHILE'
            , 'PRINT'
            , 'ELSE'
            , 'REF'
            , 'NOALIAS'

            , 'TYPE'
            , 'IDENTIFIER'
            , 'LITERAL'
            , 'COMMENT'

            , 'OPEN_PAREN'
            , 'CLOSE_PAREN'
            , 'OPEN_BRACE'
            , 'CLOSE_BRACE'
            , 'SEMICOLON'
            , 'COMMA'
            , 'DOLLARSIGN'

            # BINARY OPS
            , 'EQUAL_SIGN'
            , 'OPEN_SQUARE'
            , 'CLOSE_SQUARE'

            # ARITHMETIC OPS
            , 'PLUS'
            , 'MINUS'
            , 'MULTIPLY'
            , 'DIVIDE'

            # LOGIC OPS
            , 'DOUBLE_EQUAL'
            , 'LESS_THAN'
            , 'GREATER_THAN'
            , 'DOUBLE_AND'
            , 'DOUBLE_PIPE'
        

            # Unary Ops
            , 'EXCLAMATION'            
            , 'SLIT'
             ]    

    #t_INT = r'int'
    #t_FLOAT = r'float'
    #t_VOID = r'void'
    #t_CINT = r'cint'
    #t_BOOL = r'bool'
    t_LITERAL = r'[0-9]+(\.[0-9]+)?'
    #t_COMMENT = r'\#.*\n'

    t_OPEN_PAREN = r'\('
    t_CLOSE_PAREN = r'\)'
    t_OPEN_BRACE = r'{'
    t_CLOSE_BRACE = r'}'
    t_SEMICOLON = r';'
    t_COMMA = r','
    t_DOLLARSIGN = r'\$'

    # BINARY OPS
    t_EQUAL_SIGN = '='
    t_OPEN_SQUARE = '\['
    t_CLOSE_SQUARE = '\]'

    # ARITHMETIC OPS
    t_MULTIPLY = r'\*'
    t_PLUS = r'\+'
    t_MINUS = r'\-'
    t_DIVIDE = r'\/'


    # LOGIC OPS
    t_DOUBLE_EQUAL = r'=='
    t_LESS_THAN = r'<'
    t_GREATER_THAN = r'>'
    # bitwise AND only for bools
    t_DOUBLE_AND = r'&&'
    # bitwise OR only for bools
    t_DOUBLE_PIPE = r'\|\|'

    # UNARY OPS
    t_EXCLAMATION = r'!'


    t_SLIT = '"[^"\n\r]*"'

    t_ignore = ' \t'

    reserved = {
        'extern' : 'EXTERN',
        'def':      'DEF',
        'return':   'RETURN',
        'while':    'WHILE',
        'if':       'IF',
        'else':     'ELSE',
        'print':    'PRINT',
        'ref':      'REF',
        'noalias':  'NOALIAS' 
     }
  
    def t_IDENTIFIER(self, t):
        r'[a-zA-Z_]+[a-zA-Z0-9_]*'
        if t.value == 'true' or t.value == 'false':
            t.type = 'LITERAL'
            return t

        a = self.reserved.get(t.value)    # Check for reserved words
        if a is not None:
            t.type = a
            return t

        if t.value in self.types:
            t.type = 'TYPE'
        else:
            t.type = 'IDENTIFIER'

        return t

    types = {
        'int', 'float', 'void', 'cint', 'bool'
    }

    def t_COMMENT(self, t):
        r'\#.*\n'
        t.lexer.lineno += 1

    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)

    def t_error(self, t):
        line = t.value.split('\n')[0]
        logging.info('Unlexable input at line: {i} character: {c} \n {o}'
            .format(i=t.lexer.lineno, c=t.lexer.lexpos, o=line))
        sys.exit(0)

    def get_lexer(self):
        start = datetime.datetime.utcnow()
        l = ply.lex.lex(module=self)
        end = datetime.datetime.utcnow()
        self.lex_time = end - start
        return l
 