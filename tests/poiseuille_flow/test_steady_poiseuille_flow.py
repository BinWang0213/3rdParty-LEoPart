# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import (VectorElement, FiniteElement, Expression,
                    FunctionSpace, RectangleMesh,UnitSquareMesh, Point,
                    Function,MeshFunction,Measure,FacetNormal,
                    SubDomain, Constant, near, MixedElement, DirichletBC, DOLFIN_EPS,File,
                    Identity, sym, grad, div, assemble, dx, dot)
import numpy as np
from mpi4py import MPI as pyMPI
from leopart import StokesStaticCondensation, FormsStokes
import pytest

comm = pyMPI.COMM_WORLD


# Input Parameters
Length=1.5       # Length of pipe
Height=1.0       # Diamter of pipe
P_inlet = 10.0   # inlet pressure
P_outlet = 0.0   # outlet pressure
visc=1.0         # Fluid viscosity

def Gamma(x, on_boundary): return on_boundary

def Corner(x, on_boundary): return x[0] < DOLFIN_EPS and x[1] < DOLFIN_EPS

class Noslip(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], Length)


def exact_solution(domain):
    P7 = VectorElement("Lagrange", "triangle", degree=8, dim=2)
    P2 = FiniteElement("Lagrange", "triangle", 3)
    coeff = (P_inlet-P_outlet)/(2*Length*visc)
    u_exact = Expression(("C*x[1]*(H - x[1])", "0.0"), C=coeff,H=Height, element=P7, domain=domain)
    p_exact = Expression("dP-dP/L*x[0]", dP=(P_inlet-P_outlet), L=Length, element=P2, domain=domain)
    return u_exact, p_exact

def compute_convergence(iterator, errorlist):
    assert len(iterator) == len(errorlist), 'Iterator list and error list not of same length'
    alpha_list = []
    for i in range(len(iterator)-1):
        conv_rate = np.log(errorlist[i+1]/errorlist[i])/np.log(iterator[i+1]/iterator[i])
        alpha_list.append(conv_rate)
    return alpha_list


@pytest.mark.parametrize('k', [1, 2, 3])
def test_steady_stokes(k):
    # Polynomial order and mesh resolution
    nx_list = [4, 8, 16]

    if comm.Get_rank() == 0:
        print('{:=^72}'.format('Computing for polynomial order '+str(k)))

    nu = Constant(visc) 

    # Error listst
    error_u, error_p, error_div = [], [], []

    for nx in nx_list:
        if comm.Get_rank() == 0:
            print('# Resolution '+str(nx))

        mesh = RectangleMesh(Point(0,0), Point(Length, Height), int(nx*Length), nx, "right")

        zero_vec = np.zeros(mesh.geometry().dim())

        #Mark boundary
        mark = {"Internal":0,"wall": 98,"inlet": 99,"outlet": 100 }
        boundaries = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
        boundaries.set_all(mark["Internal"])
        wall=Noslip()
        wall.mark(boundaries, mark["wall"])
        left = Left()
        left.mark(boundaries, mark["inlet"])
        right = Right()
        right.mark(boundaries, mark["outlet"])
        ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

        # Get forcing from exact solutions
        u_exact, p_exact = exact_solution(mesh)
        f = Constant(zero_vec)

        # Set up the normal pressure bc
        n = FacetNormal(mesh)
        h_d_inlet=-Constant(P_inlet) * n
        h_d_outlet=-Constant(P_outlet) * n
        h_d = [h_d_inlet,h_d_outlet]

        # Define FunctionSpaces and functions
        V = VectorElement("DG", mesh.ufl_cell(), k)
        Q = FiniteElement("DG", mesh.ufl_cell(), k-1)
        Vbar = VectorElement("DGT", mesh.ufl_cell(), k)
        Qbar = FiniteElement("DGT", mesh.ufl_cell(), k)

        mixedL = FunctionSpace(mesh, MixedElement([V, Q]))
        mixedG = FunctionSpace(mesh, MixedElement([Vbar, Qbar]))

        Uh = Function(mixedL)
        Uhbar = Function(mixedG)

        # Set forms
        alpha = Constant(6*k*k)
        forms_stokes = FormsStokes(mesh, mixedL, mixedG, alpha,h_d=h_d,ds=ds).forms_steady(nu, f)

        # No-slip boundary conditions, set pressure in one of the corners
        bc0 = DirichletBC(mixedG.sub(0), Constant(zero_vec), boundaries, mark["wall"])
        # Normal flow constrain, only for vertical inlet and outlet
        bc_in = DirichletBC(mixedG.sub(0).sub(1), Constant(0.0), boundaries, mark["inlet"])
        bc_out = DirichletBC(mixedG.sub(0).sub(1), Constant(0.0), boundaries, mark["outlet"])
        bcs=[bc0] + [bc_in,bc_out]

        # Initialize static condensation class
        ssc = StokesStaticCondensation(mesh,
                                       forms_stokes['A_S'], forms_stokes['G_S'],
                                       forms_stokes['B_S'],
                                       forms_stokes['Q_S'], forms_stokes['S_S'], bcs)

        # Assemble global system and incorporates bcs
        ssc.assemble_global_system(True)
        # Solve using mumps
        ssc.solve_problem(Uhbar, Uh, "mumps", "default")

        # Compute velocity/pressure/local div error
        uh, ph = Uh.split()
        e_u = np.sqrt(np.abs(assemble(dot(uh-u_exact, uh-u_exact)*dx)))
        e_p = np.sqrt(np.abs(assemble((ph-p_exact) * (ph-p_exact)*dx)))
        e_d = np.sqrt(np.abs(assemble(div(uh)*div(uh)*dx)))

        #Visulize the results
        ufile_pvd = File("velocity.pvd")
        ufile_pvd << uh
        pfile_pvd = File("pressure.pvd")
        pfile_pvd << ph
        # Check boundary conditions
        file = File("boundaries_mesh.pvd")
        file << boundaries

        if comm.rank == 0:
            error_u.append(e_u)
            error_p.append(e_p)
            error_div.append(e_d)
            print('Error in velocity '+str(error_u[-1]))
            print('Error in pressure '+str(error_p[-1]))
            print('Local mass error '+str(error_div[-1]))

    if comm.rank == 0:
        iterator_list = [1./float(nx) for nx in nx_list]
        conv_u = compute_convergence(iterator_list, error_u)
        conv_p = compute_convergence(iterator_list, error_p)

        assert any(conv > k+0.75 for conv in conv_u)
        assert any(conv > (k-1)+0.75 for conv in conv_p)

test_steady_stokes(1)