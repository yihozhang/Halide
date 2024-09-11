tests(GROUPS instrsel-benchmarks
        SOURCES
        einsum_flat_1x1.cpp
        matmul_flat_1x1.cpp
        matmul_flat_1x4.cpp
        matmul_flat_1x4_preload_rhs.cpp
        matmul_vnni_1x4_preload_rhs.cpp
        matmul_vnni_1x4.cpp
        matmul_vnni_1x1.cpp
        matmul_vnni_2x2.cpp
)