class: center, middle, inverse

# Backup

---

## Portable, Extensible Toolkit for Scientific Computation (PETSc)

.left70[
![PETSc
architecture](https://upload.wikimedia.org/wikipedia/commons/4/4a/Petsc-components.svg)

.reference["Petsc-components" by Barry Smith and Jed Brown - PETSc source
repository and users manual.
Via Wikimedia Commons -
https://commons.wikimedia.org/wiki/File:Petsc-components.svg#mediaviewer/File:Petsc-components.svg]
]

.right30[
### PETSc
Data structures and algorithms for solving PDEs (in parallel)

### petsc4py
Python bindings for PETSc (via Cython)

### mpi4py
Python bindings for Message Passing Interface (MPI)
]

---

## Solve implementation
```python
def solve(problem, solution,
          bcs=None, J=None,
          solver_parameters=None)
```

1. If problem is linear, transform into residual form
2. If no Jacobian provided, compute Jacobian by automatic differentiation
3. Set up PETSc SNES solver (parameters user configurable)
4. Assign residual and Jacobian forms for SNES callbacks
5. Solve nonlinear problem. For each nonlinear iteration:
    * assemble Jacobian matrix
    * assemble residual vector
    * solve linear system using PETSc KSP

---

## PyOP2 Device Data State

.scale[![PyOP2 device data state](images/pyop2_device_data_state.svg)]

---

## Distributed Parallel Computations with MPI

.scale[![Decomposed mesh](images/pyop2_mpi_mesh.svg)]

???

* MPI: mesh needs to be decomposed using your favourite graph partitioner
* Computations on boundaries require up-to-date *halo* data
* Partial overlap: matching entities in matching colours in the diagram
* Enforce constraint on local mesh numbering for efficient comp-comm overlap
  i.e. constraint on the ordering of maps
* Contract: constraint has to be satisfied by PyOP2 user
* Local mesh entities partioned into four consecutive sections
  * **Core:** Entities owned by this processor which can be processed without
    accessing halo data.
  * **Owned:** Entities owned by this processor which access halo data when
    processed.
  * **Exec halo:** Off-processor entities redundantly executed over
    because they touch owned entities.
  * **Non-exec halo:** Off-processor entities which are not processed, but
    read when computing the exec halo.
* Entities that do not touch the boundary (core entities, by construction) can
  be computed while halo data exchange is in flight
* Halo exchange is automatic and happens only if needed i.e. halo is "dirty"

---
name: bcs

## Applying boundary conditions

* Always preserve symmetry of the operator
* Avoid costly search of CSR structure to zero rows/columns
* Zeroing during assembly, but requires boundary DOFs:
  * negative row/column indices for boundary DOFs during addto
  * instructs PETSc to drop entry, leaving 0 in assembled matrix

???

* How can we call assembly before knowing final BCs?
* BCs may change between point of assembly and solve
* assembly returns unassembled matrix with assembly "thunk" (recipe), called with BCs when solving
* Assembly is cached
  * pre-assembly not required in most circumstances
  * Matrices record BCs they have been assembled with, no need for reassembly
  * assembly cache has FIFO eviction strategy

---
template: bcs

## Preassembly

```python
A = assemble(a)
b = assemble(L)
solve(A, p, b, bcs=bcs)
```

---
template: bcs

## Preassembly

```python
*A = assemble(a)  # A unassembled, A.thunk(bcs) not yet called
b = assemble(L)
solve(A, p, b, bcs=bcs)
```

---
template: bcs

## Preassembly

```python
A = assemble(a)  # A unassembled, A.thunk(bcs) not yet called
b = assemble(L)
*solve(A, p, b, bcs=bcs)  # A.thunk(bcs) called, A assembled
```

---
template: bcs

## Preassembly

```python
A = assemble(a)  # A unassembled, A.thunk(bcs) not yet called
b = assemble(L)
solve(A, p, b, bcs=bcs)  # A.thunk(bcs) called, A assembled
# ...
*solve(A, p, b, bcs=bcs)  # bcs consistent, no need to reassemble
```

---
template: bcs

## Preassembly

```python
A = assemble(a)  # A unassembled, A.thunk(bcs) not yet called
b = assemble(L)
solve(A, p, b, bcs=bcs)  # A.thunk(bcs) called, A assembled
# ...
solve(A, p, b, bcs=bcs)  # bcs consistent, no need to reassemble
# ...
*solve(A, p, b, bcs=bcs2)  # bcs differ, reassemble, call A.thunk(bcs2)
```
