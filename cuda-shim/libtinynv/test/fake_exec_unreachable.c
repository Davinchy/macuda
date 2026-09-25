/* The rest of exec.c's outside world for test_upload_order: every function it can call that the ordering paths under
 * test must NOT reach. Each one stops the test with its own name, so a path that wanders into a card-only dependency
 * fails loudly instead of being faked into agreement. Deliberately no headers: these are bare symbols, and none of
 * them is ever called with arguments on a path the test exercises. Regenerate from `nm -u build/src/exec.o` if exec.c
 * gains a dependency (the link then fails, naming it). */
#include <stdio.h>
#include <stdlib.h>
static void unreachable(const char *f) { printf("  FAIL: the tested path reached %s, a card-only dependency\n", f); exit(1); }
void tinynv_cmd_local_memory(void);
void tinynv_cmd_local_memory(void) { unreachable("tinynv_cmd_local_memory"); }
void tinynv_cmd_set_object(void);
void tinynv_cmd_set_object(void) { unreachable("tinynv_cmd_set_object"); }
void tinynv_cmd_shader_window(void);
void tinynv_cmd_shader_window(void) { unreachable("tinynv_cmd_shader_window"); }
void tinynv_cubin_image(void);
void tinynv_cubin_image(void) { unreachable("tinynv_cubin_image"); }
void tinynv_image_free(void);
void tinynv_image_free(void) { unreachable("tinynv_image_free"); }
void tinynv_image_relocate(void);
void tinynv_image_relocate(void) { unreachable("tinynv_image_relocate"); }
void tinynv_local_memory_size(void);
void tinynv_local_memory_size(void) { unreachable("tinynv_local_memory_size"); }
void tinynv_mm_alloc_buffer(void);
void tinynv_mm_alloc_buffer(void) { unreachable("tinynv_mm_alloc_buffer"); }
void tinynv_mm_alloc_va(void);
void tinynv_mm_alloc_va(void) { unreachable("tinynv_mm_alloc_va"); }
void tinynv_mm_free_va(void);
void tinynv_mm_free_va(void) { unreachable("tinynv_mm_free_va"); }
void tinynv_mm_host_owns_iova(void);
void tinynv_mm_host_owns_iova(void) { unreachable("tinynv_mm_host_owns_iova"); }
void tinynv_mm_map_range_fresh(void);
void tinynv_mm_map_range_fresh(void) { unreachable("tinynv_mm_map_range_fresh"); }
void tinynv_mm_range_is_mapped(void);
void tinynv_mm_range_is_mapped(void) { unreachable("tinynv_mm_range_is_mapped"); }
void tinynv_mm_reserve_cpu_region(void);
void tinynv_mm_reserve_cpu_region(void) { unreachable("tinynv_mm_reserve_cpu_region"); }
void tinynv_mm_stage(void);
void tinynv_mm_stage(void) { unreachable("tinynv_mm_stage"); }
void tinynv_mm_unmap_range(void);
void tinynv_mm_unmap_range(void) { unreachable("tinynv_mm_unmap_range"); }
void tinynv_mm_va_to_pa(void);
void tinynv_mm_va_to_pa(void) { unreachable("tinynv_mm_va_to_pa"); }
void tinynv_mm_wedge(void);
void tinynv_mm_wedge(void) { unreachable("tinynv_mm_wedge"); }
void tinynv_mmio_refuse(void);
void tinynv_mmio_refuse(void) { unreachable("tinynv_mmio_refuse"); }
void tinynv_qmd_cbuf0(void);
void tinynv_qmd_cbuf0(void) { unreachable("tinynv_qmd_cbuf0"); }
void tinynv_qmd_chain(void);
void tinynv_qmd_chain(void) { unreachable("tinynv_qmd_chain"); }
void tinynv_qmd_launch(void);
void tinynv_qmd_launch(void) { unreachable("tinynv_qmd_launch"); }
void tinynv_qmd_program(void);
void tinynv_qmd_program(void) { unreachable("tinynv_qmd_program"); }
void tinynv_qmd_release(void);
void tinynv_qmd_release(void) { unreachable("tinynv_qmd_release"); }
void tinynv_queue_state(void);
void tinynv_queue_state(void) { unreachable("tinynv_queue_state"); }
void tinynv_ring_entry(void);
void tinynv_ring_entry(void) { unreachable("tinynv_ring_entry"); }
void tinynv_ring_entry_read(void);
void tinynv_ring_entry_read(void) { unreachable("tinynv_ring_entry_read"); }
void tinynv_sass_version(void);
void tinynv_sass_version(void) { unreachable("tinynv_sass_version"); }
void tinynv_vmap_free(void);
void tinynv_vmap_free(void) { unreachable("tinynv_vmap_free"); }
void tinynv_submit_ring_complete(void);
void tinynv_submit_ring_complete(void) { unreachable("tinynv_submit_ring_complete"); }
void tinynv_submit_ring_send(void);
void tinynv_submit_ring_send(void) { unreachable("tinynv_submit_ring_send"); }
