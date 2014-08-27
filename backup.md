class: center, middle, inverse

# Backup

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
