# -*- coding: utf-8 -*-
# Copyright (C) 2020 Nathan Sime
# Contact: nsime _at_ carnegiescience.edu
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pytest
from dolfin import (
    FiniteElement,
    Expression,
    FunctionSpace,
    UnitSquareMesh,
    Function,
    Constant,
    DirichletBC,
    grad,
    assemble,
    dx, dS, ds,
    inner, outer, TestFunction, CellDiameter, FacetNormal, derivative,
PETScMatrix, PETScVector, PETScKrylovSolver, PETScOptions
)
from leopart import AssemblerStaticCondensation


def compute_convergence(iterator, errorlist):
    assert len(iterator) == len(errorlist), "Iterator list and error list not of same length"
    alpha_list = []
    for i in range(len(iterator) - 1):
        conv_rate = np.log(errorlist[i + 1] / errorlist[i]) / np.log(iterator[i + 1] / iterator[i])
        alpha_list.append(conv_rate)
    return np.array(alpha_list, dtype=np.double)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_poisson(k):
    # Polynomial order and mesh resolution
    nx_list = [4, 8, 16]

    # Error list
    error_u_l2, error_u_h1 = [], []

    for nx in nx_list:
        mesh = UnitSquareMesh(nx, nx)

        # Define FunctionSpaces and functions
        V = FunctionSpace(mesh, "DG", k)
        Vbar = FunctionSpace(
            mesh, FiniteElement("CG", mesh.ufl_cell(), k)["facet"])

        u_soln = Expression("sin(pi*x[0])*sin(pi*x[1])",
                            degree=k + 1, domain=mesh)
        f = Expression("2*pi*pi*sin(pi*x[0])*sin(pi*x[1])", degree=k + 1)
        u, v = Function(V), TestFunction(V)
        ubar, vbar = Function(Vbar), TestFunction(Vbar)

        n = FacetNormal(mesh)
        h = CellDiameter(mesh)
        alpha = Constant(6 * k * k)
        penalty = alpha / h

        def facet_integral(integrand):
            return integrand('-') * dS + integrand('+') * dS + integrand * ds

        u_flux = ubar
        F_v_flux = grad(u) + penalty * outer(u_flux - u, n)

        residual_local = inner(grad(u), grad(v)) * dx
        residual_local += facet_integral(inner(outer(u_flux - u, n), grad(v)))
        residual_local -= facet_integral(inner(F_v_flux, outer(v, n)))
        residual_local -= f*v*dx

        residual_global = facet_integral(inner(F_v_flux, outer(vbar, n)))

        a_ll = derivative(residual_local, u)
        a_lg = derivative(residual_local, ubar)
        a_gl = derivative(residual_global, u)
        a_gg = derivative(residual_global, ubar)

        l_l = -residual_local
        l_g = -residual_global

        bcs = [DirichletBC(Vbar, u_soln, "on_boundary")]

        # Initialize static condensation assembler
        assembler = AssemblerStaticCondensation(
            a_ll, a_lg,
            a_gl, a_gg,
            l_l, l_g,
            bcs
        )

        A_g, b_g = PETScMatrix(), PETScVector()
        assembler.assemble_global_lhs(A_g)
        assembler.assemble_global_rhs(b_g)

        for bc in bcs:
            bc.apply(A_g, b_g)

        solver = PETScKrylovSolver()
        solver.set_operator(A_g)
        PETScOptions.set("ksp_type", "preonly")
        PETScOptions.set("pc_type", "lu")
        PETScOptions.set("pc_factor_mat_solver_type", "mumps")
        solver.set_from_options()

        solver.solve(ubar.vector(), b_g)
        assembler.backsubstitute(ubar._cpp_object, u._cpp_object)

        # Compute L2 and H1 norms
        e_u_l2 = assemble((u - u_soln)**2 * dx)**0.5
        e_u_h1 = assemble(grad(u - u_soln)**2 * dx)**0.5

        if mesh.mpi_comm().rank == 0:
            error_u_l2.append(e_u_l2)
            error_u_h1.append(e_u_h1)

    if mesh.mpi_comm().rank == 0:
        iterator_list = [1.0 / float(nx) for nx in nx_list]
        conv_u_l2 = compute_convergence(iterator_list, error_u_l2)
        conv_u_h1 = compute_convergence(iterator_list, error_u_h1)

        # Optimal rate of k + 1 - tolerance
        assert np.all(conv_u_l2 >= (k + 1.0 - 0.15))
        # Optimal rate of k - tolerance
        assert np.all(conv_u_h1 >= (k - 0.1))
