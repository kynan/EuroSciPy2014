# Firedrake: a High-level, Portable Finite Element Computation Framework

## http://firedrakeproject.org

## **Florian Rathgeber**<sup>1</sup>, Lawrence Mitchell<sup>1</sup>, David Ham<sup>1,2</sup>, Michael Lange<sup>3</sup>, Andrew McRae<sup>2</sup>, Fabio Luporini<sup>1</sup>, Gheorghe-teodor Bercea<sup>1</sup>, Paul Kelly<sup>1</sup>

<sup>1</sup> Department of Computing, Imperial College London
<sup>2</sup> Department of Mathematics, Imperial College London
<sup>3</sup> Department of Earth Science & Engineering, Imperial College London]

Slides from [my talk](https://www.euroscipy.org/2014/schedule/presentation/15/)
at the [EuroSciPy](https://www.euroscipy.org/2014/), Cambridge, UK,
August 29-30 2014: http://kynan.github.io/EuroSciPy2014

### Abstract

In an ideal world, scientific applications are computationally
efficient, maintainable, composable and allow scientists to work very
productively. In this talk we demonstrate that these goals are
achievable for a specific application domain by choosing suitable
domain-specific abstractions implemented in Python that encapsulate
domain knowledge with a high degree of expressiveness.
 
We present a [Firedrake], a high-level Python framework for the portable
solution of partial differential equations on unstructured meshes with
the finite element method widely used in science and engineering.
[Firedrake] is built on top of [PyOP2], a domain-specific language
embedded in Python for parallel mesh-based computations.  Finite element
local assembly operations execute the same computational kernel for
every element of the mesh and is therefore efficiently parallelisable.
 
[Firedrake] allows scientists to describe variational forms and
discretisations for finite element problems symbolically in a notation
very close to the maths using the Unified Form Language [UFL] from the
[FEniCS project]. Variational forms are translated into computational
kernels by the FEniCS Form Compiler [FFC].  Numerical linear algebra is
delegated to PETSc, leveraged via its petsc4py interface.
 
[PyOP2] abstracts away the performance-portable parallel execution of
these kernels on a range of hardware architectures, targeting multi-core
CPUs with OpenMP and GPUs and accelerators with PyCUDA and PyOpenCL and
distributed parallel computations with mpi4py. Backend-specific code
tailored to each specific computation is generated, just-in-time
compiled and efficiently scheduled for parallel execution at runtime.
 
Due to the composability of the [Firedrake] and [PyOP2] abstractions,
optimised implementations for different hardware architectures can be
automatically generated without any changes to a single high-level
source. Performance matches or exceeds what is realistically attainable
by hand-written code. Both projects are open source and developed at
Imperial College London.

[PyOP2]: http://op2.github.io/PyOP2
[Firedrake]: http://firedrakeproject.org
[FEniCS project]: http://fenicsproject.org
[UFL]: https://bitbucket.org/fenics-project/ufl/
[FFC]: https://bitbucket.org/fenics-project/ffc/

### Resources

  * **PyOP2** https://github.com/OP2/PyOP2
    * *[PyOP2: A High-Level Framework for Performance-Portable Simulations on Unstructured Meshes](http://dx.doi.org/10.1109/SC.Companion.2012.134)*
      Florian Rathgeber, Graham R. Markall, Lawrence Mitchell, Nicholas Loriant, David A. Ham, Carlo Bertolli, Paul H.J. Kelly,
      WOLFHPC 2012
    * *[Performance-Portable Finite Element Assembly Using PyOP2 and FEniCS](http://link.springer.com/chapter/10.1007/978-3-642-38750-0_21)*
       Graham R. Markall, Florian Rathgeber, Lawrence Mitchell, Nicolas Loriant, Carlo Bertolli, David A. Ham, Paul H. J. Kelly ,
       ISC 2013
  * **Firedrake** https://github.com/firedrakeproject/firedrake
    * *COFFEE: an Optimizing Compiler for Finite Element Local Assembly*
      Fabio Luporini, Ana Lucia Varbanescu, Florian Rathgeber, Gheorghe-Teodor Bercea, J. Ramanujam, David A. Ham, Paul H. J. Kelly,
      submitted
  * **UFL** https://bitbucket.org/mapdes/ufl
  * **FFC** https://bitbucket.org/mapdes/ffc
