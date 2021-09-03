###############################################################################
# ProteusLib Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/nawi-hub/proteuslib/"
#
###############################################################################

from pyomo.environ import check_optimal_termination
from idaes.core.util import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom


# NOTE: In a full flowsheet, you will want to leave the default values for
#       bound_push and mu_init as is!!! However, if your flowsheet or block
#       contains ONLY chemistry, then setting bound_push = 1e-10 and mu_init = 1e-6
#       works ver well. 
def solve_with_user_scaling(blk, solver=None, tee=False, fail_flag=True, bound_push=0.1, mu_init=1e-1):
    if solver is None:
        solver = get_solver(options={'nlp_scaling_method': 'user-scaling',
                                     'bound_push': bound_push,
                                     'mu_init': mu_init})
    results = solver.solve(blk, tee=tee)
    if fail_flag:
        check_solve(results)


def check_dof(blk, dof_expected=0):
    if degrees_of_freedom(blk) != dof_expected:
        raise RuntimeError("The degrees of freedom on {blk} were {dof} but {dof_e} "
                           "were expected, check the fixed variables on that block".format(
            blk=blk, dof=degrees_of_freedom(blk), dof_e=dof_expected))


def check_solve(results):
    if not check_optimal_termination(results):
        raise RuntimeError("The solver failed to converge to an optimal solution. "
                           "This suggests that the user provided infeasible inputs "
                           "or that the model is poorly scaled.")
