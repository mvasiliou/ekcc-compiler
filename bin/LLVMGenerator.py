import llvmlite.ir as ir
import llvmlite.binding as binding
from ctypes import CFUNCTYPE, c_int
import logging
import datetime


class LLVMGenerator(object):

    def __init__(self):
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()

        self.module = ir.Module(name=__file__)
        self.module.triple = binding.get_default_triple()

        target = binding.Target.from_default_triple()
        target_machine = target.create_target_machine()

        # And an execution engine with an empty backing module
        backing_mod = binding.parse_assembly("")
        engine = binding.create_mcjit_compiler(backing_mod, target_machine)
        self.engine = engine

        self.llvm_time = None
        self.optimize_time = None
        self.run_time = None

    def generate(self, tree, emit, jit, use_optimizations, extra_args, input_file):
        start = datetime.datetime.utcnow()
        scope = self._setup(extra_args)
        tree.visit(self.module, scope, extra_args)

        self.finalize(use_optimizations)
        end = datetime.datetime.utcnow()
        self.llvm_time = end - start

        if emit:
            self._write()

        if jit:
            self._run()

    def finalize(self, use_optimizations):
        llvm_ir = str(self.module)
        self.module = binding.parse_assembly(llvm_ir)
        self.module.verify()

        if type(use_optimizations) == list:
            logging.info('Compiling with optimizations')
            start = datetime.datetime.utcnow()
            pmb = binding.create_pass_manager_builder()
            pm = binding.create_module_pass_manager()

            if 'dead_arg' in use_optimizations:
                logging.info('Adding dead arguments optimizations')
                pm.add_dead_arg_elimination_pass()
                
            if 'constant_merge' in use_optimizations:
                logging.info('Adding constant merge optimizations')
                pm.add_constant_merge_pass()

            if 'gvn' in use_optimizations:
                logging.info('Adding global value numbering')
                pm.add_gvn_pass()

            if 'licm' in use_optimizations:
                logging.info('Adding Loop Invariant Code Motion')
                pm.add_licm_pass()

            if 'inline' in use_optimizations:
                logging.info('Setting inline threshold to 1')
                pm.add_function_inlining_pass(1)

            if 'inline_50' in use_optimizations:
                logging.info('Setting inline threshold to 50')
                pm.add_function_inlining_pass(50)

            if 'inline_100' in use_optimizations:
                logging.info('Setting inline threshold to 100')
                pm.add_function_inlining_pass(100)

            if 'inline_200' in use_optimizations:
                logging.info('Setting inline threshold to 200')
                pm.add_function_inlining_pass(200)

            if 'inline_300' in use_optimizations:
                logging.info('Setting inline threshold to 300')
                pm.add_function_inlining_pass(300)

            if 'opt_3' in use_optimizations:
                logging.info('Using O3 optimizations')
                pmb.opt_level = 3
                pmb.populate(pm)
            elif 'opt_2' in use_optimizations:
                logging.info('Using O2 optimizations')
                pmb.opt_level = 2
                pmb.populate(pm)
            elif 'opt_1' in use_optimizations:
                logging.info('Using O1 optimizations')
                pmb.opt_level = 1
                pmb.populate(pm)

            if 'unroll_loops' in use_optimizations:
                pmb.disable_unroll_loops = False
                pmb.populate(pm)
            else:
                pmb.disable_unroll_loops = True
                pmb.populate(pm)

            if 'loop_vectorize' in use_optimizations:
                pmb.loop_vectorize = True
                pmb.populate(pm)

            if 'global_dce' in use_optimizations:
                pm.add_global_dce_pass()

            if 'global_optimizer' in use_optimizations:
                pm.add_global_optimizer_pass()

            if 'dce' in use_optimizations:
                pm.add_dead_code_elimination_pass()

            if 'ipsccp' in use_optimizations:
                pm.add_ipsccp_pass()

            if 'sccp' in use_optimizations:
                pm.add_sccp_pass()
            
            pm.run(self.module)

            end = datetime.datetime.utcnow()
            self.optimize_time = end - start

    def _setup(self, extra_args):
        scope = {}
        self._add_print_function(scope)
        self._add_exit_function(scope)
        return scope
        
    def _add_print_function(self, scope):
        voidptr_ty = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        scope['print'] = ir.Function(self.module, printf_ty, name="printf")

    def _add_exit_function(self, scope):
        intptr_ty = ir.IntType(32)
        exit_ty = ir.FunctionType(ir.VoidType(), [intptr_ty], var_arg=True)
        scope['exit'] = ir.Function(self.module, exit_ty, name="exit")

    def _write(self):
        with open('output.ll', 'w') as output_file:
            output_file.write(str(self.module))

    def _run(self):
        start = datetime.datetime.utcnow()
        self.engine.add_module(self.module)
        self.engine.finalize_object()
        self.engine.run_static_constructors()

        cfptr = self.engine.get_function_address('run')
        cfunc = CFUNCTYPE(c_int)(cfptr)
        res = cfunc()
        end = datetime.datetime.utcnow()
        logging.info('Exit Code: ' + str(res))
        self.run_time = end - start


