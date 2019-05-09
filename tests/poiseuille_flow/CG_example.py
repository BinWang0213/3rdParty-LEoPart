#from fenics import *
#from mshr import *
from dolfin import (VectorElement, FiniteElement, Expression,TrialFunctions,TestFunctions,
                    FunctionSpace, RectangleMesh,UnitSquareMesh, Point,
                    Function,MeshFunction,Measure,FacetNormal,
                    SubDomain, Constant, near, MixedElement, DirichletBC, DOLFIN_EPS,File,
                    Identity, sym, grad, div, assemble, inner, dx, dot,solve)
import numpy as np
from mpi4py import MPI as pyMPI

comm = pyMPI.COMM_WORLD

#https://fenicsproject.org/pub/tutorial/html/._ftut1009.html
#http://www.nbi.dk/~mathies/cm/fenics_instructions.pdf

# Input Parameters
Length=1.5            # Length of pipe
Height=1.0            # Diamter of pipe
P_inlet = 10.0   # inlet pressure
P_outlet = 0.0   # outlet pressure
visc=1.0         # Fluid viscosity


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

# 1. Mesh from fenics 
nx=4
mesh = RectangleMesh(Point(0,0), Point(Length, Height), int(nx*Length), nx, "right")

zero_vec = np.zeros(mesh.geometry().dim())

# Define boundaries
mark = {"Internal":0,"wall": 98,"inlet": 99,"outlet": 100 }

boundaries = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
boundaries.set_all(mark["Internal"])
wall=Noslip()
wall.mark(boundaries, mark["wall"])
left = Left()
left.mark(boundaries, mark["inlet"])
right = Right()
right.mark(boundaries, mark["outlet"])

ds = Measure('ds',domain=mesh,subdomain_data=boundaries)
n = FacetNormal(mesh)

# Define variational problem
k = 2
if(k>=2): # Taylor-Hood Function Space 3rd or higher order accurancy
    V = VectorElement("CG", mesh.ufl_cell(), k)
    Q = FiniteElement("CG", mesh.ufl_cell(), k-1)
    W = FunctionSpace(mesh, V*Q)
else:
    #Optional MINI element, 2nd order accurancy for velocity
    # - MINI
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    B = FiniteElement("Bubble",   mesh.ufl_cell(), 3)
    V = VectorElement(P1 + B)
    Q = P1
    W = FunctionSpace(mesh, V * Q)

# Define variational problem for stokes
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Weak formulation - Bilinear form
nu=Constant(visc)
a = (nu*inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx

# Body force - Linear form
f = Constant((0, 0))
Lf = inner(f, v)*dx

# Neumann boundary condition - Linear form
L_neumann = dot(-Constant(P_inlet)*n,v) * ds(mark["inlet"]) + \
            dot(-Constant(P_outlet)*n,v) * ds(mark["outlet"])


L= Lf + L_neumann

# Dirichlet boundary condition
noslip = Constant(zero_vec)
bc_wall = DirichletBC(W.sub(0), noslip, boundaries, mark["wall"])

bcs = [bc_wall]

# Compute solution
w = Function(W)
solve(a == L, w, bcs)

# Split the mixed solution using deepcopy
# (needed for further computation on coefficient vector)
(u, p) = w.split(True)

# # Split the mixed solution using a shallow copy
(u, p) = w.split()


# Save solution in VTK format
ufile_pvd = File("velocity.pvd")
ufile_pvd << u
pfile_pvd = File("pressure.pvd")
pfile_pvd << p

# Save sub domains to VTK files
file = File("boundaries_mesh.pvd")
file << boundaries

# Convergence analysis
u_exact, p_exact = exact_solution(mesh)
e_u = np.sqrt(np.abs(assemble(dot(u-u_exact, u-u_exact)*dx)))
e_p = np.sqrt(np.abs(assemble((p-p_exact) * (p-p_exact)*dx)))
e_d = np.sqrt(np.abs(assemble(div(u)*div(u)*dx)))

if comm.rank == 0:
    print('Error in velocity '+str(e_u))
    print('Error in pressure '+str(e_p))
    print('Local mass error '+str(e_d))