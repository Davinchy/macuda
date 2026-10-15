// Stub libtinynv: lets libtinycudart build/link/run on the Mac before B's real driver exists. Logs the launch the driver would submit.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tinynv.h"
struct tinynv_device { int ord; };
struct tinynv_module { const void* cubin; size_t len; };
struct tinynv_kernel { struct tinynv_module* m; char name[128]; };
struct tinynv_stream { int id; };
struct tinynv_event  { int id; };
static struct tinynv_device g_d = {0};
tinynv_status_t tinynv_init(void){ return TINYNV_OK; }
int tinynv_device_count(void){ return 1; }
tinynv_status_t tinynv_device_get(tinynv_device_t* o,int ord){ (void)ord; *o=&g_d; return TINYNV_OK; }
tinynv_status_t tinynv_device_props(tinynv_device_t d,tinynv_device_props_t* p){ (void)d; memset(p,0,sizeof(*p)); strcpy(p->name,"tinynv-stub"); p->cc_major=8;p->cc_minor=6;p->sm_count=82;p->warp_size=32;p->total_mem=24ull<<30;p->max_threads_per_block=1024;p->max_shared_per_block=48*1024; return TINYNV_OK; }
tinynv_status_t tinynv_module_load(tinynv_device_t d,const void* c,size_t n,tinynv_module_t* o){ (void)d; struct tinynv_module* m=calloc(1,sizeof(*m)); m->cubin=c;m->len=n; *o=m; return TINYNV_OK; }
tinynv_status_t tinynv_module_unload(tinynv_module_t m){ free(m); return TINYNV_OK; }
tinynv_status_t tinynv_get_kernel(tinynv_module_t m,const char* name,tinynv_kernel_t* o){ struct tinynv_kernel* k=calloc(1,sizeof(*k)); k->m=m; strncpy(k->name,name,127); *o=k; return TINYNV_OK; }
tinynv_status_t tinynv_kernel_info(tinynv_kernel_t k,tinynv_kernel_info_t* i){ (void)k; memset(i,0,sizeof(*i)); i->param_base=-1; /* STUB ONLY: real libtinynv (B's cubin.c) reads this per-cubin from PARAM_CBANK; it is arch-dependent (sm_86=0x160, sm_90=0x210, sm_120=0x380) — do NOT hardcode */ return TINYNV_OK; }
tinynv_status_t tinynv_malloc(tinynv_device_t d,size_t n,tinynv_devptr_t* o){ (void)d; void* p=malloc(n); *o=(tinynv_devptr_t)p; return p?TINYNV_OK:TINYNV_ERR_OOM; }
tinynv_status_t tinynv_free(tinynv_device_t d,tinynv_devptr_t p){ (void)d; free((void*)p); return TINYNV_OK; }
tinynv_status_t tinynv_memcpy_htod(tinynv_stream_t s,tinynv_devptr_t dst,const void* src,size_t n){ (void)s; memcpy((void*)dst,src,n); return TINYNV_OK; }
tinynv_status_t tinynv_memcpy_dtoh(tinynv_stream_t s,void* dst,tinynv_devptr_t src,size_t n){ (void)s; memcpy(dst,(void*)src,n); return TINYNV_OK; }
tinynv_status_t tinynv_memcpy_dtod(tinynv_stream_t s,tinynv_devptr_t dst,tinynv_devptr_t src,size_t n){ (void)s; memcpy((void*)dst,(void*)src,n); return TINYNV_OK; }
tinynv_status_t tinynv_memset(tinynv_stream_t s,tinynv_devptr_t dst,int v,size_t n){ (void)s; memset((void*)dst,v,n); return TINYNV_OK; }
tinynv_status_t tinynv_host_alloc(tinynv_device_t d,size_t n,void** o){ (void)d; *o=malloc(n); return *o?TINYNV_OK:TINYNV_ERR_OOM; }
tinynv_status_t tinynv_host_free(tinynv_device_t d,void* p){ (void)d; free(p); return TINYNV_OK; }
tinynv_status_t tinynv_stream_create(tinynv_device_t d,tinynv_stream_t* o){ (void)d; *o=calloc(1,sizeof(struct tinynv_stream)); return TINYNV_OK; }
tinynv_status_t tinynv_stream_destroy(tinynv_stream_t s){ free(s); return TINYNV_OK; }
tinynv_status_t tinynv_stream_sync(tinynv_stream_t s){ (void)s; return TINYNV_OK; }
tinynv_status_t tinynv_event_create(tinynv_device_t d,tinynv_event_t* o){ (void)d; *o=calloc(1,sizeof(struct tinynv_event)); return TINYNV_OK; }
tinynv_status_t tinynv_event_record(tinynv_event_t e,tinynv_stream_t s){ (void)e;(void)s; return TINYNV_OK; }
tinynv_status_t tinynv_event_sync(tinynv_event_t e){ (void)e; return TINYNV_OK; }
tinynv_status_t tinynv_stream_wait_event(tinynv_stream_t s,tinynv_event_t e){ (void)s;(void)e; return TINYNV_OK; }
tinynv_status_t tinynv_launch(tinynv_stream_t s,tinynv_kernel_t k,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned sm,const void* params,size_t plen){
  (void)s;
  printf("[tinynv-stub] LAUNCH '%s' grid=(%u,%u,%u) block=(%u,%u,%u) dyn_smem=%u params=%zuB\n", k?k->name:"?",gx,gy,gz,bx,by,bz,sm,plen);
  const unsigned char* p=(const unsigned char*)params;
  printf("[tinynv-stub]   params:"); for(size_t i=0;i<plen && i<40;i++) printf(" %02x",p[i]); printf("  <- a,b,c ptrs + n\n");
  printf("[tinynv-stub]   (real libtinynv would: pick sm_120 cubin, build QMD, place params at c[0x0][0x160], set cta_raster=grid, ring doorbell)\n");
  return TINYNV_OK;
}
const char* tinynv_status_str(tinynv_status_t s){ return s==TINYNV_OK?"ok":"err"; }
