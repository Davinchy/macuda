# abi-* fixtures: one kernel, two CUDA ELF ABIs (B, 2026-09-21)

`abi-kfat.cu` was turned into PTX by nvcc 13.0 (`-ptx -arch=compute_80`). The `.version` line was then lowered from 9.0
to 8.7 so that ptxas 12.8 accepts it. That was the only edit: the kernel is one store, and it uses nothing from PTX 9.0.
The same PTX was then compiled on the 3090 box by two ptxas:
- ptxas 12.8 (V12.8.93, the bundle in /mnt/4tb/l-shim/cu128-site/triton/backends/nvidia/bin)
- ptxas 13.0 (V13.0.88, /usr/local/cuda-13.0)

| file | ptxas | OSABI / ABI | e_flags | sha256 |
|---|---|---|---|---|
| abi7-sm80.cubin | 12.8, -arch=sm_80 | 0x33 / 7 | 0x00500550 (sm in the LOW byte) | 68993f23f7bf86025fe9ecf9c2c04dd671226d6b56ddc918404efe57988112df |
| abi8-sm80.cubin | 13.0, -arch=sm_80 | 0x41 / 8 | 0x06005004 (sm in the SECOND byte) | f859d2eecf6302f039fb49a31d81775dea11ff057dc970cffdefa3a1e6d0ebb7 |
| abi7-sm90.cubin | 12.8, -arch=sm_90 | 0x33 / 7 | 0x0050055a | ad28d477f7f533578851685934831643e078f44e9f732e6adba2f52fb6e0e5ca |
| abi8-sm120-ptxas12.cubin | 12.8, -arch=sm_120 | 0x41 / 8 | 0x06007802 | 9447c171375debfa06ee19ce2b6ef259d4c03fe5a2d667a8619866a581b1cd1e |
| abi-kfat.ptx | - | - | - | 4132400e1a4d4655e100cfe1e6cd2949a24d5469ec8da27f4d5e0512c133943e |

ptxas 13.0 for sm_90 and sm_120 also gives ABI 8. That was measured, but those files are not kept: the table shows
which toolkit makes which ABI.
