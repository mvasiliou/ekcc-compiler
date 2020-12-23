import ply.yacc
import Lexer
from collections import namedtuple
import yaml
import logging
import sys
import llvmlite.ir as ir
import datetime

FUNCTION_VAR_TYPES = {}


class Node(object):

    INT_MIN = ir.Constant(ir.IntType(32), -2147483648)

    def _get_ir_type(self, ast_type):
        if 'ref' in ast_type:
            base_type = self._get_ir_type(ast_type.replace('ref ', ''))
            return ir.PointerType(base_type)
        if ast_type == 'int' or ast_type == 'cint':
            return ir.IntType(32)
        if ast_type == 'void':
            return ir.VoidType()
        if ast_type == 'bool':
            return ir.IntType(1)
        if ast_type == 'float':
            return ir.FloatType()
        raise Exception('Unable to find LLVM type for: ' + ast_type)

    def _exit_program(self, module, builder, scope):
        func = scope.get('exit')
        args = [ir.Constant(ir.IntType(32), 1)]
        builder.call(func, args)


class Program(Node):

    def __init__(self, externs, funcs):
        self.externs = externs
        self.funcs = funcs
        self._name = 'prog'

        found_run = False
        for func in self.funcs.funcs:
            if func.globid == 'run':
                found_run = True
                if func.ret_type != 'int':
                    logging.error('run method must return an int type!')
                    sys.exit(1)
                elif len(func.vdecls.vars) != 0:
                    logging.error('run method must not take arguments!')
                    sys.exit(1)
                else:
                    break

        if not found_run:
            logging.error('Program must define a run function!')
            sys.exit(1)

    def visit(self, module, scope, extra_args):
        if type(self.externs) != list:
            self.externs.visit(module, scope, extra_args)
        self.funcs.visit(module, scope)

    def get_functions(self):
        return self.funcs.funcs


class ExternList(Node):
    name = 'externs'

    def __init__(self, extern):
        self.externs = [extern]
        self._name = 'externs'

    def add_extern(self, extern):
        self.externs.append(extern)

    def visit(self, module, scope, extra_args):
        for extern in self.externs:
            extern.visit(module, scope, extra_args)

class Extern(Node):

    def __init__(self, ret_type, globid, tdecls):
        self.ret_type = ret_type
        self.globid = globid
        self.tdecls = tdecls
        self._name = 'extern'

    def visit(self, module, scope, extra_args):
        if self.globid == 'arg':
            self._add_arg_function(module, scope, extra_args)
        if self.globid == 'argf':
            self._add_argf_function(module, scope, extra_args)

    def _add_arg_function(self, module, scope, extra_args):
        for arg in extra_args:
            if '.' in arg:
                logging.error('Attempting to pass in float while using arg')
                sys.exit(1)

        arg_ty = ir.IntType(32)
        f_ty = ir.FunctionType(ir.IntType(32), [arg_ty], var_arg=True)
        arg = ir.Function(module, f_ty, name="arg")
        scope['arg'] = arg

        block = arg.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        for i, x in enumerate(extra_args):
            index = ir.Constant(ir.IntType(32), i)
            cond = builder.icmp_signed('==', arg.args[0], index, name='equal')

            with builder.if_then(cond) as conditional:
                builder.ret(ir.Constant(ir.IntType(32), int(x)))
        builder.ret(ir.Constant(ir.IntType(32), 0))

    def _add_argf_function(self, module, scope, extra_args):
        arg_ty = ir.IntType(32)
        f_ty = ir.FunctionType(ir.FloatType(), [arg_ty], var_arg=True)
        arg = ir.Function(module, f_ty, name="argf")
        scope['argf'] = arg

        block = arg.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        for i, x in enumerate(extra_args):
            index = ir.Constant(ir.IntType(32), i)
            cond = builder.icmp_signed('==', arg.args[0], index, name='equal')

            with builder.if_then(cond) as conditional:
                builder.ret(ir.Constant(ir.FloatType(), float(x)))
        builder.ret(ir.Constant(ir.FloatType(), 0.0))

class FuncList(Node):

    def __init__(self, func):
        self.funcs = [func]

    def add_func(self, func):
        self.funcs.append(func)

    def visit(self, module, scope):
        for func in self.funcs:
            func.visit(module, scope)


class Function(Node):

    def __init__(self, ret_type, globid, blk, vdecls):
        self.ret_type = ret_type
        self.globid = globid
        self.blk = blk
        self.vdecls = vdecls
        self._name = 'func'

        if 'ref' in ret_type:
            logging.error('A function may not return a ref type.')
            sys.exit(1)

        if self.globid in FUNCTION_VAR_TYPES:
            logging.error('Function ' + self.globid + ' has already been defined')
            sys.exit(1)
        else:
            FUNCTION_VAR_TYPES[self.globid] = {}

    def visit(self, module, scope):
        llvm_ret_type = self._get_ir_type(self.ret_type)
        arguments = [arg.llvm_vtype for arg in self.vdecls.vars]
        fnty = ir.FunctionType(llvm_ret_type, arguments)
        func = ir.Function(module, fnty, name=self.globid)
        scope[self.globid] = func
        local_scope = {k:v for k,v in scope.items()}
        block = func.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        for x, arg in enumerate(self.vdecls.vars):
            local_scope[arg.var] = func.args[x]

        self.blk.visit(builder, module, func, local_scope)
        if self.ret_type == 'void':
            builder.ret_void()


class VDeclList(Node):

    def __init__(self, vdecl=None):
        if vdecl is None:
            self.vars = []
        else:
            self.vars = [vdecl]
        self._name = 'vdecls'

    def add_vdecl(self, vdecl):
        self.vars.append(vdecl)


class VariableDecl(Node):

    def __init__(self, vtype, var):
        self.vtype = vtype
        self.var = var
        self._name = 'vdecl'

        if self.vtype == 'void':
            logging.error('In <vdecl>, the type may not be void!')
            sys.exit(1)

    @property
    def llvm_vtype(self):
        return self._get_ir_type(self.vtype)


class StatementList(Node):

    def __init__(self, stmt):
        self.stmts = [stmt]
        self._name = 'stmts'

    def add_stmt(self, stmt):
        self.stmts.append(stmt)

    def visit(self, builder, module, block, scope):
        for statement in self.stmts:
            statement.visit(builder, module, block, scope)


class Block(Node):
    count = 0
    def __init__(self, statements):
        self.contents = statements
        self._name = 'blk'

    def visit(self, builder, module, block, scope):
        self.count += 1
        local_scope = {k:v for k,v in scope.items()}        
        if type(self.contents) == list and len(self.contents) == 0:
            return
        self.contents.visit(builder, module, block, local_scope)


class IfStatement(Node):

    def __init__(self, cond, stmt):
        self.cond = cond
        self.stmt = stmt
        self._name = 'if'

    def visit(self, builder, module, block, scope):
        condition = self.cond.visit(builder, module, block, scope)

        with builder.if_then(condition) as conditional:
            self.stmt.visit(builder, module, block, scope)

class IfElseStatement(Node):

    def __init__(self, cond, stmt, else_stmt):
        self.cond = cond
        self.stmt = stmt
        self.else_stmt = else_stmt
        self._name = 'if'

    def visit(self, builder, module, block, scope):
        condition = self.cond.visit(builder, module, block, scope)

        with builder.if_else(condition) as (then, otherwise):
            with then:
                self.stmt.visit(builder, module, block, scope)
            with otherwise:
                self.else_stmt.visit(builder, module, block, scope)


class WhileStatement(Node):

    def __init__(self, cond, stmt):
        self.cond = cond
        self.stmt = stmt
        self._name = 'while'

    def visit(self, builder, module, block, scope):
        new_block = builder.append_basic_block(name="while")
        block_builder = ir.IRBuilder(new_block)
        builder.branch(new_block)
        
        condition = self.cond.visit(block_builder, module, block, scope)
        with block_builder.if_then(condition) as conditional:
            self.stmt.visit(block_builder, module, new_block, scope)
            block_builder.branch(new_block)
            builder.position_at_end(conditional)


class BinOp(Node):

    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs
        self._name = 'binop'

    def get_vtype(self, local_scope):
        left = self.lhs.get_vtype(local_scope).replace('ref ', '').replace('noalias ', '')
        right = self.rhs.get_vtype(local_scope).replace('ref ', '').replace('noalias ', '')

        self.left_vtype = left
        self.right_vtype = right

        if self.op in ('lt', 'gt', 'eq', 'and', 'or'):
            self.vtype = 'bool'
            return self.vtype

        if (left == 'cint' and right == 'int') or (left == 'int' and right == 'cint'):
            self.vtype = 'cint'
        elif left != right:
            logging.error('Type of left and right expr are not equal!')
            logging.error('Left: ' + left)
            logging.error('Right: ' + right)
            sys.exit(1)
        else:
            self.vtype = left
        return self.vtype

    def visit(self, builder, module, block, scope):
        left = self.lhs.visit(builder, module, block, scope)
        right = self.rhs.visit(builder, module, block, scope)

        if getattr(left.type, 'pointee', None) is not None:
            left = builder.load(left)

        if getattr(right.type, 'pointee', None) is not None:
            right = builder.load(right)

        # These could be floats
        if self.op == 'lt':
            if self.left_vtype == 'float':
                func = builder.fcmp_ordered
            else:
                func = builder.icmp_signed
            return func('<', left, right, name='less_than')

        if self.op == 'gt':
            if self.left_vtype == 'float':
                func = builder.fcmp_ordered
            else:
                func = builder.icmp_signed
            return func('>', left, right, name='great_than')

        if self.op == 'eq':
            if self.left_vtype == 'float':
                func = builder.fcmp_ordered
            else:
                func = builder.icmp_signed
            return func('==', left, right, name='equal_to')

        if self.op == 'and':
            return builder.and_(left, right, name='and')
            
        if self.op == 'or':
            return builder.or_(left, right, name='or')

        if self.op == 'mul':
            if self.vtype == 'float':
                func = builder.fmul
            elif self.vtype == 'cint':
                func = builder.smul_with_overflow
                func_proper = builder.mul
                return self._checked_overflow_arith(module, scope, builder, left, right, 
                                                    func, func_proper, 'mul')
            else:
                func = builder.mul
                
            return func(left, right, name='mul')

        if self.op == 'div':
            zero = ir.Constant(ir.IntType(32), 0)
            condition = builder.icmp_signed('==', right, zero, name='oflow_check')
            with builder.if_then(condition) as conditional:
                self._exit_program(module, builder, scope)

            if self.vtype == 'float':
                func = builder.fdiv
            else:
                neg1 = ir.Constant(ir.IntType(32), -1)
                neg_condition = builder.icmp_signed('==', right, neg1, name='neg1_check')
                int_min_cond = builder.icmp_signed('==', left, self.INT_MIN, name='int_min_check')

                condition = builder.and_(neg_condition, int_min_cond)

                with builder.if_then(condition) as conditional:
                    self._exit_program(module, builder, scope)


                with builder.if_then(condition) as conditional:
                    self._exit_program(module, builder, scope)

                func = builder.sdiv

            return func(left, right, name='div')

        if self.op == 'add':
            if self.vtype == 'float':
                func = builder.fadd
            elif self.vtype == 'cint':
                func = builder.sadd_with_overflow
                func_proper = builder.add
                return self._checked_overflow_arith(module, scope, builder, left, right, 
                                                    func, func_proper, 'add')
            else:
                func = builder.add
            return func(left, right, name='add')

        if self.op == 'sub':
            if self.vtype == 'float':
                func = builder.fsub
            elif self.vtype == 'cint':
                func = builder.ssub_with_overflow
                func_proper = builder.sub
                return self._checked_overflow_arith(module, scope, builder, left, right, 
                                                    func, func_proper, 'sub')
            else:
                func = builder.sub
            return func(left, right, name='sub')

        if self.op == '=':
            pointer = scope[self.lhs.var]
            return builder.store(right, pointer)

        raise Exception('Cannot define BinOp LLVM method: ' + self.op)

    def _checked_overflow_arith(self, module, scope, builder, left, right, func, func_proper, name):
        oflow_type = ir.LiteralStructType((ir.IntType(32), ir.IntType(1)))
        pointer = builder.alloca(oflow_type)

        check_oflow = func(left, right, name=name)
        builder.store(check_oflow, pointer)

        int0 = ir.Constant(ir.IntType(32), 0)
        int1 = ir.Constant(ir.IntType(32), 1)
        result_pointer = builder.gep(pointer, [int0, int1])
        result = builder.load(result_pointer)

        condition = builder.icmp_signed('==', result, ir.Constant(ir.IntType(1), 1), name='oflow_check')
        
        return_pointer = builder.alloca(ir.IntType(32))

        operation = func_proper(left, right, name=name)

        final_pointer = builder.gep(pointer, [int0, int0])
        final = builder.load(final_pointer)

        builder.store(final, return_pointer)
        with builder.if_then(condition) as then:
            self._exit_program(module, builder, scope)

        return builder.load(return_pointer)


class ReturnStatement(Node):

    def __init__(self, exp):
        self.exp = exp
        self._name = 'ret'

    def visit(self, builder, module, block, scope):
        if self.exp is None:
            builder.ret_void()
            return

        result = self.exp.visit(builder, module, block, scope)
        if getattr(result.type, 'pointee', None) is not None:
            result = builder.load(result)
        builder.ret(result)


class VDeclStatement(Node):

    def __init__(self, vdecl, exp):
        self.vdecl = vdecl
        self.exp = exp
        self._name = 'vardeclstmt'

    def visit(self, builder, module, block, scope):
        pointer = builder.alloca(self.vdecl.llvm_vtype)
        scope[self.vdecl.var] = pointer
        value = self.exp.visit(builder, module, block, scope)
        builder.store(value, pointer)


class PrintStatement(Node):

    _is_setup = False
    def __init__(self, exp):
        self.exp = exp
        self._name = 'print'

    @staticmethod
    def _setup(module, builder):
        if PrintStatement._is_setup:
            return

        # Declare argument list
        voidptr_ty = ir.IntType(8).as_pointer()
        
        fmt = "%i \n\0"
        c_fmt = ir.Constant(ir.ArrayType(ir.IntType(8), len(fmt)),
                            bytearray(fmt.encode("utf8")))
        
        global_fmt = ir.GlobalVariable(module, c_fmt.type, name="fstr")
        global_fmt.linkage = 'internal'
        global_fmt.global_constant = True
        global_fmt.initializer = c_fmt
        
        PrintStatement.fmt_arg = builder.bitcast(global_fmt, voidptr_ty)
        PrintStatement._is_setup = True

    def visit(self, builder, module, block, scope):
        func = scope.get('print')
        self._setup(module, builder)
        # Call Print Function
        value = self.exp.visit(builder, module, block, scope)
        if getattr(value.type, 'pointee', None) is not None:
            value = builder.load(value)
        args = [PrintStatement.fmt_arg, value]
        builder.call(func, args)


class PrintSLiteral(Node):

    _is_setup = False
    def __init__(self, string):
        self.string = string[1:-1]
        self._name = 'printslit'

    @classmethod
    def _setup(cls, module, builder):
        if PrintSLiteral._is_setup:
            return

        # Declare argument list
        voidptr_ty = ir.IntType(8).as_pointer()
        
        fmt = "%s \n\0"
        c_fmt = ir.Constant(ir.ArrayType(ir.IntType(8), len(fmt)),
                            bytearray(fmt.encode("utf8")))
        
        global_fmt = ir.GlobalVariable(module, c_fmt.type, name="slit")
        global_fmt.linkage = 'internal'
        global_fmt.global_constant = True
        global_fmt.initializer = c_fmt
        
        PrintSLiteral.fmt_arg = builder.bitcast(global_fmt, voidptr_ty)
        PrintSLiteral._is_setup = True

    def visit(self, builder, module, block, scope):
        func = scope.get('print')
        self._setup(module, builder)
        
        self.string += '\0'
        c_str_val = ir.Constant(ir.ArrayType(ir.IntType(8), len(self.string)),
                                bytearray(self.string.encode("utf8")))

        c_str = builder.alloca(c_str_val.type)
        builder.store(c_str_val, c_str)

        # Call Print Function
        args = [self.fmt_arg, c_str]
        builder.call(func, args)


class CastStatement(Node):

    def __init__(self, a, b):
        self.casttype = a
        self.expr = b
        self._name = 'caststmt'


class UOP(Node):

    def __init__(self, op, expr):
        self.op = op
        self.expr = expr
        self._name = 'uop'

    def get_vtype(self, local_scope):
        self.vtype = self.expr.get_vtype(local_scope)
        return self.vtype

    def visit(self, builder, module, block, scope):
        expr = self.expr.visit(builder, module, block, scope)
        if getattr(expr.type, 'pointee', None) is not None:
            expr = builder.load(expr)
        if self.op == '!':
            return builder.not_(expr)
        else:
            condition = builder.icmp_signed('==', expr, self.INT_MIN, name='oflow_check')
            with builder.if_then(condition) as conditional:
                self._exit_program(module, builder, scope)

            return builder.neg(expr)

class FuncCall(Node):

    def __init__(self, globid, params):
        self.globid = globid
        self.params = params
        self._name = 'funccall'

    def get_vtype_fake(self, local_scope):
        self.vtype = 'int'
        return self.vtype

    def visit(self, builder, module, block, scope):
        func = scope.get(self.globid)

        if func is None:
            logging.error('Attempting to use function before declaration: ' + self.globid)
            sys.exit(1)
        args = [p.visit(builder, module, block, scope) for p in self.params]

        for i, x in enumerate(args):
            pointer = getattr(func.args[i].type, 'pointee', None)
            arg_pointer = getattr(x.type, 'pointee', None)

            if pointer is None and arg_pointer is not None:
                args[i] = builder.load(x)

            if pointer is not None and arg_pointer is None:
                logging.error('Trying to pass raw value as a pointer')
                sys.exit(1)

        return builder.call(func, args)


class Literal(Node):

    def __init__(self, value):
        self.value = value
        self._name = 'lit'

    def get_vtype(self, local_scope):
        if self.value == 'true' or self.value == 'false':
            self.vtype = 'bool'
        elif '.' in self.value:
            self.vtype = 'float'
        else:
            self.vtype = 'int'
        return self.vtype

    def visit(self, builder, module, block, scope):
        if self.vtype == 'float':
            value = float(self.value)
        else:
            value = self.value
        return ir.Constant(self._get_ir_type(self.vtype), value)


class VarId(Node):

    def __init__(self, var):
        self.var = var
        self._name = 'varval'

    def get_vtype(self, local_scope):
        if self.var in local_scope:
            self.vtype = local_scope[self.var]
            return self.vtype
        else:
            logging.error('Could not find type for: ' + self.var)
            sys.exit(1)

    def visit(self, builder, module, block, scope):
        variable = scope[self.var]
        return variable


class Parser(object):

    tokens = Lexer.Lexer.tokens
    precedence = (
     ('right', 'EQUAL_SIGN'),
     ('left', 'DOUBLE_PIPE'),
     ('left', 'DOUBLE_AND'),
     ('left', 'DOUBLE_EQUAL'),
     ('left', 'LESS_THAN', 'GREATER_THAN'),
     ('left', 'PLUS', 'MINUS'),
     ('left', 'MULTIPLY', 'DIVIDE'), 
     ('right', 'EXCLAMATION', 'UMINUS') 
    )

    def p_error(self, p):
        if p:
            print("Syntax error at token", p.type, p.__dict__)
            # Just discard the token and tell the parser it's okay.
        else:
            print("Syntax error at EOF")

    def p_prog(self, p):
        'prog : extern_list func_list'
        p[0] = Program(p[1], p[2])

    def p_prog_empty_extern(self, p):
        'prog : func_list'
        p[0] = Program([], p[1])

    def p_extern_list(self, p):
        'extern_list :  extern_list extern'
        externs = p[1]
        externs.add_extern(p[2])
        p[0] = externs

    def p_extern(self, p):
        'extern_list : extern'
        p[0] = ExternList(p[1])

    def p_single_extern(self, p):
        '''
        extern : EXTERN type IDENTIFIER OPEN_PAREN tdecls CLOSE_PAREN SEMICOLON
               | EXTERN type IDENTIFIER OPEN_PAREN CLOSE_PAREN SEMICOLON
        '''
        if len(p) == 8:
            p[0] = Extern(p[2], p[3], p[5])
        else:
            p[0] = Extern(p[2], p[3], [])

    def p_tdecls(self, p):
        '''
        tdecls : type 
               | type COMMA tdecls'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = [p[1]] + p[3]

    def p_type(self, p):
        '''type : TYPE
        '''
        p[0] = ' '.join(p[1:])

    def p_ref_type(self, p):
        '''
        type : REF TYPE
             | NOALIAS REF TYPE
        '''
        type_string = p[len(p) - 1]
        if type_string == 'void':
            logging.error('In ref <type>, the type may not be void.')
            sys.exit(1)
        if 'ref' in type_string:
            logging.error('In ref <type>, the type may not also be a reference type.')
            sys.exit(1)
        p[0] = ' '.join(p[1:])

    def p_ref_ref_type(self, p):
        '''
        type : REF REF TYPE
        '''
        logging.error('In ref <type>, the type may not also be a reference type.')
        sys.exit(1)

    def p_single_func(self, p):
        '''func : DEF type IDENTIFIER OPEN_PAREN vdecls CLOSE_PAREN blk
                | DEF type IDENTIFIER OPEN_PAREN CLOSE_PAREN blk
        '''
        if len(p) == 8:
            p[0] = Function(p[2], p[3], p[7], p[5])
        else:
            p[0] = Function(p[2], p[3], p[6], VDeclList())

    def p_func_list(self, p):
        'func_list :  func_list func'
        func_list = p[1]
        func_list.add_func(p[2])
        p[0] = func_list

    def p_func(self, p):
        'func_list : func'
        p[0] = FuncList(p[1])

    def p_vdecl(self, p):
        'vdecl : type DOLLARSIGN IDENTIFIER'
        p[0] = VariableDecl(p[1], p[2] + p[3])

    def p_vdecls(self, p):
        '''
        vdecls : vdecl
               | vdecl COMMA vdecls
        '''
        if len(p) == 2:
            p[0] = VDeclList(p[1])
        else:
            vdecls = p[3]
            vdecls.add_vdecl(p[1])
            p[0] = vdecls

    def p_blk(self, p):
        'blk : OPEN_BRACE statement_list CLOSE_BRACE'
        p[0] = Block(p[2])    

    def p_blk_empty(self, p):
        'blk : OPEN_BRACE CLOSE_BRACE'
        p[0] = Block([])
    
    def p_statement_list(self, p):
        'statement_list : statement_list statement'
        stmts = p[1]
        stmts.add_stmt(p[2])
        p[0] = stmts

    def p_statement(self, p):
        'statement_list : statement'
        p[0] = StatementList(p[1])

    def p_single_statement(self, p):
        '''statement : blk
                     | exp SEMICOLON
        '''
        p[0] = p[1]

    def p_if_statement(self, p):
        '''
        statement : IF OPEN_PAREN exp CLOSE_PAREN statement
        '''
        p[0] = IfStatement(p[3], p[5])

    def p_if_else_statement(self, p):
        '''
        statement : IF OPEN_PAREN exp CLOSE_PAREN statement ELSE statement
        '''
        p[0] = IfElseStatement(p[3], p[5], p[7])

    def p_while_statement(self, p):
        '''
        statement : WHILE OPEN_PAREN exp CLOSE_PAREN statement
        '''
        p[0] = WhileStatement(p[3], p[5])

    def p_print_exp_statement(self, p):
        '''
        statement : PRINT exp SEMICOLON
        '''
        p[0] = PrintStatement(p[2])

    def p_print_slit_statement(self, p):
        '''
        statement : PRINT SLIT SEMICOLON
        '''
        p[0] = PrintSLiteral(p[2])

    def p_return_empty(self, p):
        '''
        statement : RETURN SEMICOLON
        '''
        p[0] = ReturnStatement(None)

    def p_return_full(self, p):
        '''
        statement : RETURN exp SEMICOLON
        '''
        p[0] = ReturnStatement(p[2])

    def p_vdeclstatement(self, p):
        '''
        statement : vdecl EQUAL_SIGN exp SEMICOLON
        '''
        p[0] = VDeclStatement(p[1], p[3])

    def p_exp(self, p):
        '''
        exp : OPEN_PAREN exp CLOSE_PAREN
            | binop
            | uop
            | lit
            | varid
        '''
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = p[2]

    def p_funccall(self, p):
        '''
        exp : IDENTIFIER OPEN_PAREN CLOSE_PAREN
            | IDENTIFIER OPEN_PAREN exps CLOSE_PAREN
        '''
        if len(p) == 4:
            p[0] = FuncCall(p[1], [])
        else:
            p[0] = FuncCall(p[1], p[3])

    def p_exps(self, p):
        '''exps : exp
                | exp COMMA exps
        '''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = [p[1]] + p[3]

    def p_binop(self, p):
        '''binop : arith_ops
                 | logic_ops
                 | varid EQUAL_SIGN exp
                 | OPEN_SQUARE type CLOSE_SQUARE exp
        '''
        if len(p) == 2:
            p[0] = p[1]
        elif len(p) == 4:
            p[0] = BinOp(p[2], p[1], p[3])
        else:
            p[0] = CastStatement(p[2], p[4])

    def p_arith_ops(self, p):
        '''
        arith_ops : exp MULTIPLY exp
                  | exp DIVIDE exp
                  | exp PLUS exp
                  | exp MINUS exp
        '''
        if p[2] == '*':
            sign = 'mul'
        elif p[2] == '/':
            sign = 'div'
        elif p[2] == '+':
            sign = 'add'
        elif p[2] == '-':
            sign = 'sub'

        p[0] = BinOp(sign, p[1], p[3])

    def p_logic_ops(self, p):
        '''
        logic_ops : exp DOUBLE_EQUAL exp
                  | exp LESS_THAN exp
                  | exp GREATER_THAN exp
                  | exp DOUBLE_AND exp
                  | exp DOUBLE_PIPE exp
        '''
        if p[2] == '==':
            sign = 'eq'
        elif p[2] == '<':
            sign = 'lt'
        elif p[2] == '>':
            sign = 'gt'
        elif p[2] == '&&':
            sign = 'and'
        elif p[2] == '||':
            sign = 'or'
        p[0] = BinOp(sign, p[1], p[3])

    def p_uop(self, p):
        '''
        uop : EXCLAMATION exp
            | MINUS exp %prec UMINUS
        '''
        if p[1] == '!':
            sign = 'not'
        elif p[1] == '-':
            sign = 'minus'
        p[0] = UOP(sign, p[2])

    def p_lit(self, p):
        'lit : LITERAL'
        p[0] = Literal(p[1])

    def p_varid(self, p):
        'varid : DOLLARSIGN IDENTIFIER'
        p[0] = VarId(p[1] + p[2])

    def validate_statement(self, statement, local_scope):
        if statement._name == 'vardeclstmt':
            variable = statement.vdecl
            if variable.var in local_scope:
                logging.error('Already defined variable ' + variable.var + ' in this scope!')
                sys.exit(1)
            local_scope[variable.var] = variable.vtype
            self.validate_statement(statement.exp, local_scope)
        else:
            if getattr(statement, 'exp', None) is not None:
                vtype = statement.exp.get_vtype(local_scope)
            elif getattr(statement, 'get_vtype', None) is not None:
                vtype = statement.get_vtype(local_scope)
            elif getattr(statement, 'params', None) is not None:
                for p in statement.params:
                    vtype = p.get_vtype(local_scope)
            else:
                pass

    def validate_block(self, block, scope):
        local_scope = {k:v for k,v in scope.items()}
        if type(block.contents) == list and len(block.contents) == 0:
            return

        for s in block.contents.stmts:
            if getattr(s, 'stmt', None) is not None:
                self.validate_statement(s.cond, local_scope)
                if s.stmt._name == 'blk':
                    self.validate_block(s.stmt, local_scope)
                else:
                    # This is just one statement, so we do not need a new local scope
                    self.validate_statement(s.stmt, local_scope)
                    if getattr(s, 'else_stmt', None) is not None:
                        self.validate_statement(s.else_stmt, local_scope)
            else:
                self.validate_statement(s, local_scope)
                
    def validate(self, tree):
        for function in tree.funcs.funcs:
            scope = {v.var: v.vtype for v in function.vdecls.vars}
            self.validate_block(function.blk, scope)

    def parse(self, data, lexer):
        start = datetime.datetime.utcnow()
        self.parser = ply.yacc.yacc(module=self)
        tree = self.parser.parse(data, lexer=lexer)
        self.validate(tree)
        end = datetime.datetime.utcnow()
        self.parse_time = end - start
        return tree
