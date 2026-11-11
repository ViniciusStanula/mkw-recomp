// Draws, binds and dispatches of the Vulkan op-stream replay: the strong
// vk_exec_op_draw (vk_exec_ops.c) behind vk_exec.c's weak hook. The bind ops fill the VkExec slots; a draw opens the pass, resolves the
// bound program to its prepared shaders (a vertex shader's integer inputs come from the bound attribute formats), asks
// vk_pipeline.c for the pipeline, pushes (or allocates) set 0 as VkPipeInfo.bindings says and records the dynamic
// state, vertex and index buffers and the draw. A dispatch ends the pass and does the same for the compute shader.
// UPDATE_UB is a buffer copy from transient memory between full barriers.
//
// This file resolves programs to shaders, finds feedback loops and prepares each draw; the other parts are listed in
// vk_draw_internal.h. Nothing here locks: it runs on the submitting thread under vk_sync_submit's lock.
//
// Provenance: the op semantics (bind ops, program and slot conventions, draw and dispatch flow, vertex streams with a
// zeroed buffer for unbound ones) follow MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m:7409-7460,
// 9365-9488, 9489-9921, 10013-10079 @ b75be8cf (MIT, Copyright (c) 2026 jhaizhou); the uniform-buffer spans, texture and
// sampler descriptor lookups and the dispatch flow follow LAS Recomp runtime/nvn/vulkan/exec.c:1851-1908, 2300-2516
// @ da35b83a (MIT, Copyright (c) 2026 Peiikai). The support buffer layout is Ryujinx's SupportBuffer.cs (MIT).
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_hmap.h"
#include "tex.h"
#include "vk.h"
#include "vk_draw_internal.h"
#include "vk_exec.h"
#include "vk_reuse.h"
#include "platform/platform.h"

// MKW_SKIP_PROGRAMS=<hex>[,<hex>...] (debug, off when unset): a draw whose vertex or fragment program key starts with one
// of these hex prefixes (8 hex digits or more, as the breadcrumbs print them) is not drawn. Why: issue #22's device
// loss on NVIDIA is bounded by the breadcrumbs only to a window of ~75 draws; skipping halves of that window finds the
// one program that faults.
enum { SKIP_MAX = 128 };
static char skip_hex[SKIP_MAX][65];
static int skip_n, skip_state;   // 0 not read, 1 reading, 2 ready
static uint8_t skip_bytes[SKIP_MAX][32];
static uint8_t skip_nib[SKIP_MAX];   // hex digits in each prefix

static int hexval(char c) { return c <= '9' ? c - '0' : c - 'a' + 10; }

static void skip_read(void) {
    int z = 0;
    if (__atomic_compare_exchange_n(&skip_state, &z, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        const char *e = cfg_get("MKW_SKIP_PROGRAMS");
        while (e && *e && skip_n < SKIP_MAX) {
            while (*e == ',' || *e == ' ' || *e == ';') e++;
            size_t n = 0;
            while (n < 64 && ((e[n] >= '0' && e[n] <= '9') || (e[n] >= 'a' && e[n] <= 'f') || (e[n] >= 'A' && e[n] <= 'F'))) n++;
            if (n >= 8) {
                for (size_t i = 0; i < n; i++) skip_hex[skip_n][i] = (char)(e[i] >= 'A' && e[i] <= 'F' ? e[i] + 32 : e[i]);
                skip_hex[skip_n][n] = 0;
                skip_nib[skip_n] = (uint8_t)n;
                for (size_t i = 0; i < n; i++)
                    skip_bytes[skip_n][i / 2] |= (uint8_t)(hexval(skip_hex[skip_n][i]) << (i & 1 ? 0 : 4));
                rt_log("vk_exec: MKW_SKIP_PROGRAMS: draws with a program starting %s are skipped", skip_hex[skip_n]);
                skip_n++;
            }
            if (!n) break;
            e += n;
        }
        __atomic_store_n(&skip_state, 2, __ATOMIC_RELEASE);
    }
    while (__atomic_load_n(&skip_state, __ATOMIC_ACQUIRE) != 2) {}
}

// Against the key's bytes: the prefixes are turned into bytes once (skip_bytes), not the key into text on every draw.
static int skip_key(const uint8_t *k) {
    for (int i = 0; i < skip_n; i++) {
        int full = skip_nib[i] / 2;
        if (memcmp(k, skip_bytes[i], (size_t)full)) continue;
        if ((skip_nib[i] & 1) && (k[full] >> 4) != (skip_bytes[i][full] >> 4)) continue;
        return 1;
    }
    return 0;
}

static int skip_listed(const GfxShader *vs, const GfxShader *fs) {
    if (__atomic_load_n(&skip_state, __ATOMIC_ACQUIRE) != 2) skip_read();
    if (!skip_n) return 0;
    static uint64_t skipped;
    if ((vs && skip_key(gfx_shader_key(vs))) || (fs && skip_key(gfx_shader_key(fs)))) {
        if (!(__atomic_fetch_add(&skipped, 1, __ATOMIC_RELAXED) & ((1u << 20) - 1)))
            rt_log("vk_exec: MKW_SKIP_PROGRAMS skipped a draw (more follow without a line until 1M)");
        return 1;
    }
    return 0;
}

#define ONESHOT_WAIT_MS 30000u   // prepare_draw: how long a one-shot draw waits for its pipeline before it is skipped

void warn_once(int key, const char *fmt, ...) {
    static volatile int32_t seen[W_COUNT];
    if (__atomic_exchange_n(&seen[key], 1, __ATOMIC_RELAXED)) return;
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    rt_log("vk: %s (logged once)", msg);
}

// ---- the recording thread's state
// What belongs to the command buffer a thread records (bound pipeline and sets, memos of the last draw, the inline
// updates seen) lives per thread: MKW_REPLAY_PAR records one submission's chunks on several threads (vk_exec_par.c).
// The large blocks are allocated on a thread's first recording (vk_exec_draw_thread), not as _Thread_local storage,
// which every thread of the process would carry: B, DS, RS, UPD, SM, PM and BRA name this thread's, each through a
// pointer set there.

_Thread_local DrawB *dtl_b;
_Thread_local int pd_on;

void vk_exec_draw_new_cb(void) {
    vk_exec_draw_thread();
    memset(&B, 0, sizeof B);
}

// What is bound is not known any more (MKW_REUSE emitted packets of its own); the memos of this command buffer's
// transient buffers (c0, support) still hold.
void vk_exec_draw_forget(void) {
    vk_exec_draw_thread();
    B.pipe[0] = B.pipe[1] = VK_NULL_HANDLE;
    B.st_valid = B.vb_valid = B.res_valid = 0;
    B.ds_layout = VK_NULL_HANDLE;
    B.ds_bound[0] = B.ds_bound[1] = VK_NULL_HANDLE;
    B.ds_pushed[0] = B.ds_pushed[1] = 0;
}

// ---- program to shaders

// (program, stage, answers) -> shader, 4-way set-associative, most recently used first. A miss costs a SHA-256 of the
// whole shader code (gfx_shader.c req_init), slow on CPUs without SHA instructions: a field frame looks up ~3000 and
// the 2048-slot direct-mapped table this replaced missed 3% of them on conflicts.
#define SHADER_SETS 2048
#define SHADER_WAYS 4
typedef struct { uint64_t serial; int stage; int used; GfxShaderAnswers a; const GfxShader *s; } ShaderEnt;
static _Thread_local ShaderEnt (*shader_tab)[SHADER_WAYS];   // per recording thread (vk_exec_draw_thread)

static int find_stage_scan(const prog_t *p, int nvn_stage, const uint8_t **ctl, const uint8_t **code, uint32_t *len) {
    for (uint32_t i = 0; i < p->nstage && i < 6; i++) {   // prog_t keeps 6 shaders
        const uint8_t *c = p->sd[i].control;
        if (!c || !p->sd[i].data) continue;
        // NVN2 (MKW): the stage comes from the code's program header, the length from control+0x44 (nvn2.c); NVN1
        // kept both in its 0x880-byte control section (+0x714, +0x704).
        const uint8_t *data = (const uint8_t *)(uintptr_t)p->sd[i].data;
        if (nvn2_program_stage(c, data) != nvn_stage) continue;
        *len = nvn2_program_length(c, data);
        *ctl = c, *code = data;
        return 1;
    }
    return 0;
}

// Which shader of a program serves each NVN stage, and its code length, per program serial (new at every
// nvnProgramSetShaders): the scan reads two words of every stage's control block, cold guest cache lines, and runs
// several times a draw. Direct-mapped; a collision only scans again.
#define STAGEM_N 512
typedef struct { uint64_t serial; int8_t idx[8]; uint32_t len[8]; } StageM;   // NVN stages 0-7 (6 task, 7 mesh)
static _Thread_local StageM *stagem;   // STAGEM_N per recording thread (vk_exec_draw_thread)

int find_stage(const prog_t *p, int nvn_stage, const uint8_t **ctl, const uint8_t **code, uint32_t *len) {
    if (!p->serial || (unsigned)nvn_stage >= 8) return find_stage_scan(p, nvn_stage, ctl, code, len);
    typeof(stagem[0]) *m = &stagem[p->serial * 0x9E3779B97F4A7C15ull >> 55];
    if (m->serial != p->serial) {
        m->serial = p->serial;
        memset(m->idx, -1, sizeof m->idx);
        for (uint32_t i = 0; i < p->nstage && i < 6; i++) {   // prog_t keeps 6 shaders
            const uint8_t *c = p->sd[i].control;
            if (!c || !p->sd[i].data) continue;
            // NVN2 as find_stage_scan: stage from the code's program header, length as extract.py cuts it. This read
            // NVN1's control +0x714 / +0x704, past the 0xd8-byte NVN2 control: stages from garbage (run55: draws
            // skipped as "no vertex stage", pipelines with a "stage mismatch").
            const uint8_t *data = (const uint8_t *)(uintptr_t)p->sd[i].data;
            int st = nvn2_program_stage(c, data);
            if (st >= 0 && st < 8 && m->idx[st] < 0) m->idx[st] = (int8_t)i, m->len[st] = nvn2_program_length(c, data);
        }
    }
    int i = m->idx[nvn_stage];
    if (i < 0) return 0;
    *ctl = p->sd[i].control, *code = (const uint8_t *)(uintptr_t)p->sd[i].data, *len = m->len[nvn_stage];
    return 1;
}

// Which TEX/IMG bindings of a compute program have an integer-format view bound: the translator reads them as float
// and takes the bits (floatBitsToUint), as SM86 does; in Vulkan a float image over an integer view is undefined
// (format-07753): AMD and NVIDIA return the bits, Intel Arc drew the race black (issue #64). Array and buffer
// bindings are left as they are. MKW_INT_IMAGES=0 keeps the float images (to compare). s: the program's base shader.
static void cs_answers(VkExec *ex, const GfxShader *s, GfxShaderAnswers *a) {
    static int on = -1;
    if (on < 0) on = !cfg_get("MKW_INT_IMAGES") || strcmp(cfg_get("MKW_INT_IMAGES"), "0") != 0;
    if (!on) return;
    for (int img = 0; s && img < 2; img++) {
        const GfxShaderRes *res;
        int n = gfx_shader_resources(s, img ? GFX_RES_IMG : GFX_RES_TEX, &res);
        for (int i = 0; i < n; i++) {
            const GfxShaderRes *r = &res[i];
            if (r->count > 1 || r->binding >= 64 || r->slot < 0 || (r->type & 0xff) == 2) continue;
            uint64_t h = img ? (r->slot < 32 ? ex->img[NVN_STAGE_CS][r->slot] : 0)
                             : (r->slot < 128 ? ex->tex[NVN_STAGE_CS][r->slot] : 0);
            int t = vk_exec_res_numeric(h, r, img);
            if (t == GFX_TYPE_UINT) a->img_uint |= 1ull << r->binding;
            else if (t == GFX_TYPE_SINT) a->img_sint |= 1ull << r->binding;
        }
    }
}

// The shader of stage st of the bound program for the answers a, from this thread's table: gfx_shader_get keys the
// program by hashing its whole code, too slow for every draw and dispatch.
static const GfxShader *cached_shader(VkExec *ex, int st, const uint8_t *ctl, const uint8_t *code, const GfxShaderAnswers *a0) {
    GfxShaderAnswers a = *a0;
    uint64_t serial = ex->prog->serial;
    ShaderEnt *set = NULL;
    if (serial) {
        uint64_t h = (serial * 0x9E3779B97F4A7C15ull ^ (uint64_t)st) ^ nvn_hm_hash(&a, sizeof a);   // a is zero-padded
        set = shader_tab[(h ^ h >> 29) & (SHADER_SETS - 1)];
        for (int w = 0; w < SHADER_WAYS && set[w].used; w++) {
            ShaderEnt *e = &set[w];
            if (e->serial != serial || e->stage != st || memcmp(&e->a, &a, sizeof a)) continue;
            if (w) {   // to the front
                ShaderEnt hit = *e;
                memmove(&set[1], &set[0], w * sizeof *set);
                set[0] = hit;
            }
            return set[0].s;
        }
    }
    const GfxShader *s = gfx_shader_get(ctl, code, &a);
    if (!s) warn_once(W_SHADER, "a stage %d shader of program %llu has no prepared SPIR-V: draws with it are skipped", st,
                      (unsigned long long)serial);
    else if (set) {   // in front, the least recently used way drops out
        memmove(&set[1], &set[0], (SHADER_WAYS - 1) * sizeof *set);
        set[0] = (ShaderEnt){.serial = serial, .stage = st, .used = 1, .a = a, .s = s};
    }
    return s;
}

// *present: the program has this stage. Returns NULL when it has not or it has no prepared SPIR-V.
const GfxShader *shader_for(VkExec *ex, int st, int *present) {
    const uint8_t *ctl, *code;
    uint32_t len;
    *present = 0;
    if (!ex->prog || !find_stage(ex->prog, st, &ctl, &code, &len)) return NULL;
    *present = 1;
    // the vertex stage's attribute types pick its integer-input variant (gfx_shader.c), the fragment stage's target
    // formats its integer-output variant, the compute stage's bound image formats its integer-image variant, read
    // from the base shader's resources; other stages have none
    GfxShaderAnswers a = {0};
    if (st == NVN_STAGE_VS) vk_pipeline_answers(&ex->pipe, GFX_STAGE_VS, &a);
    else if (st == NVN_STAGE_FS) vk_pipeline_answers(&ex->pipe, GFX_STAGE_FS, &a);
    else if (st == NVN_STAGE_CS) {
        const GfxShader *base = cached_shader(ex, st, ctl, code, &a);
        cs_answers(ex, base, &a);
        if (!(a.img_uint | a.img_sint)) return base;
    }
    return cached_shader(ex, st, ctl, code, &a);
}

// The values the GPU reads for s's fixed-address constant words idx[0..n) (all words when idx is NULL) in this draw
// (vk_spec.c), as ubo_info binds them: bank 1 from the program's own constants, banks 3-20 from the submission's
// uniform snapshot. Other banks, words past the bound range and bindings served from live memory are unknown.
typedef struct { VkExec *ex; const GfxShader *s; int st; } SpecCtx;

static void spec_read(void *ctx, const uint32_t *idx, int n, uint32_t *val, uint8_t *known) {
    const SpecCtx *c = ctx;
    VkExec *ex = c->ex;
    const uint32_t *w;
    gfx_shader_spec(c->s, &w);
    if (!__atomic_load_n(&LIM.ubo_max, __ATOMIC_ACQUIRE)) limits_init();
    const uint8_t *base[21];
    uint64_t len[21];
    uint32_t looked = 0;
    for (int j = 0; j < n; j++) {
        uint32_t word = w[idx ? idx[j] : (uint32_t)j], bank = word >> 16, byte = word & 0xffff;
        known[j] = 0;
        if (bank >= 21 || bank == 0 || bank == 2) continue;
        if (!(looked >> bank & 1)) {
            looked |= 1u << bank, base[bank] = NULL, len[bank] = 0;
            if (bank == 1 && ex->prog) {
                const uint8_t *ctl, *code;
                uint32_t clen, c1_size, c1, total;
                if (find_stage(ex->prog, c->st, &ctl, &code, &clen)) {
                    memcpy(&c1_size, ctl + 0x3c, 4), memcpy(&c1, ctl + 0x40, 4), memcpy(&total, ctl + 0x44, 4);
                    if (c1_size && c1 < total && c1_size <= total - c1) base[1] = code + c1, len[1] = c1_size;
                }
            } else if (bank >= 3) {
                const p_bind_buf *b = &ex->ubo[c->st][bank - 3];
                uint64_t sz = b->size < LIM.ubo_max ? b->size : LIM.ubo_max;
                if (sz && (base[bank] = vk_usnap_peek(b->addr, sz, LIM.ubo_align))) len[bank] = sz;
            }
        }
        if (base[bank] && byte + 4 <= len[bank]) memcpy(&val[j], base[bank] + byte, 4), known[j] = 1;
    }
}

void spec_census(VkExec *ex, const GfxShader *s, int st) {
    static uint32_t val[4096];
    static uint8_t known[4096];
    const uint32_t *w;
    int n = s ? gfx_shader_spec(s, &w) : 0;
    if (!s || gfx_shader_mesh_kind(s) != GFX_MESH_NONE || n > 4096) return;
    SpecCtx c = {ex, s, st};
    spec_read(&c, NULL, n, val, known);
    vk_spec_observe(s, val, known);
}

// MKW_SPEC: s or its specialized variant (vk_spec.c).
static const GfxShader *spec_pick(VkExec *ex, const GfxShader *s, int st) {
    if (!s || gfx_shader_mesh_kind(s) != GFX_MESH_NONE) return s;
    SpecCtx c = {ex, s, st};
    return vk_spec_pick(s, __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), spec_read, &c);
}

// ---- feedback loops (maxwell.md 1.9, vk.h VkCtx.feedback)
// TotK's G-buffer and terrain shaders sample targets of the pass that renders them, at their own pixel, after earlier
// draws of the pass wrote them (albedo, normals, R32F linear depth). Maxwell reads see those writes; in Vulkan the read
// is undefined without a barrier, and on the RX 9070 it showed as dark screen-aligned blocks on the island ground
// and hid Tulin's sage spirit. Before a draw whose fragment shader samples an attachment of
// the open pass that a draw or clear of the pass wrote since the pass began or the last barrier:
// - VK_FEEDBACK_LAYOUT: the image is learned (VkTexImage.fb_learned; it has the feedback usage since its creation), the
//   pass restarts once with it in ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT, and from then on its passes attach it in that
//   layout, its descriptors carry that layout, the draws that sample it use a pipeline with the feedback flag
//   (VkPipeState.feedback) and a by-region barrier precedes such a draw.
// - VK_FEEDBACK_RESTART: the pass ends and restarts behind the full barrier (vk_exec_pass_begin).

// The image and the levels [l0, l1) and layers [a0, a1) a texture's view covers.
typedef struct { VkImage image; uint32_t l0, l1, a0, a1; } SubRange;

static SubRange sub_of(const VkTexImage *img, const view_t *v, int hv) {
    SubRange r = {img->image, 0, img->levels, 0, img->layers};
    if (hv && v->set_levels) r.l0 = v->base_level, r.l1 = v->base_level + (v->nlevels ? v->nlevels : img->levels);
    if (hv && v->set_layers) r.a0 = v->min_layer, r.a1 = v->min_layer + (v->nlayers ? v->nlayers : img->layers);
    return r;
}

// Attachment slots of the open pass (bit 8 depth) whose subresource r covers.
static uint32_t attached_as(const VkExec *ex, const SubRange *r) {
    uint32_t m = 0;
    if (!r->image) return 0;
    for (uint32_t i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if ((i < 8 && i >= ex->pass_ncolor) || t->image != r->image) continue;
        if (t->level < r->l0 || t->level >= r->l1) continue;
        if (!t->is3d && (t->base_layer + t->layers <= r->a0 || t->base_layer >= r->a1)) continue;
        m |= 1u << i;
    }
    return m;
}

// The subresource a texture handle samples, per handle, kept while its slot resolves the same (slot_still): a draw's
// texture bindings change the whole-scan memo below, but the handles they bind were mostly looked up before, in this
// pass or an earlier one. Direct-mapped.
#define FM_N 8192
typedef struct { uint64_t h, tok; uint32_t gen; SubRange r; } FeedMemo;
static _Thread_local FeedMemo *FM;   // FM_N per recording thread (vk_exec_draw_thread): its hits are read without a lock

// One lookup per texture lock: held over a draw's whole scan, the game's vk_texture_register waited for it (wlp2c,
// Windows: 54 ms per s of race, the scan holding it 71).
static SubRange sampled_sub(uint64_t h) {
    view_t v;
    int hv;
    vk_texture_lock();
    const VkTexImage *img = vk_texture_by_id_locked((int)(h & 0xfffff), &v, &hv);
    SubRange r = img ? sub_of(img, &v, hv) : (SubRange){0};
    vk_texture_unlock();
    return r;
}

// Attachments of the open pass the fragment shader fs samples; kept while the shader, the texture bindings and the
// pass stay the same.
static uint32_t feedback_scan(VkExec *ex, const GfxShader *fs) {
    if (ex->fb_fs == fs && ex->fb_serial == ex->tex_serial && ex->fb_pass == ex->pass_serial) return ex->fb_sampled;
    const GfxShaderRes *res;
    int n = gfx_shader_resources(fs, GFX_RES_TEX, &res);
    uint32_t m = 0, gen = vk_texture_slots_gen();   // before any lookup
    int memo = td_mode();
    for (int r = 0; r < n; r++)
        for (int k = 0; k < (res[r].count ? res[r].count : 1); k++) {
            int slot = res[r].slot + k;
            uint64_t h = slot >= 0 && slot < 128 ? ex->tex[NVN_STAGE_FS][slot] : 0;
            if (!h) continue;
            typeof(FM[0]) *f = &FM[(h * 0x9E3779B97F4A7C15ull) >> 51];
            if (memo && f->h == h && slot_still(&f->gen, f->tok, h, gen)) {
                if (memo == 2) {
                    SubRange c = sampled_sub(h);
                    if (memcmp(&c, &f->r, sizeof c))
                        rt_fatal("vk_exec: feedback memo of handle %#llx differs from a lookup", (unsigned long long)h);
                }
                m |= attached_as(ex, &f->r);
                continue;
            }
            uint64_t tok = token_of(h);   // before the lookup
            SubRange sr = sampled_sub(h);
            m |= attached_as(ex, &sr);
            if (tok) *f = (typeof(*f)){.h = h, .tok = tok, .gen = gen, .r = sr};
            else f->h = 0;
        }
    ex->fb_fs = fs, ex->fb_serial = ex->tex_serial, ex->fb_pass = ex->pass_serial, ex->fb_sampled = m;
    return m;
}

// Before the pipeline lookup of a draw with fragment shader fs: learns and restarts as the mode says. Returns the
// sampled attachments in the feedback layout (the pipeline's feedback flag and feedback_barrier).
static uint32_t feedback_prepare(VkExec *ex, const GfxShader *fs) {
    int mode = vk_ctx()->feedback;
    if (mode == VK_FEEDBACK_OFF || !fs || !ex->in_pass) return 0;
    uint32_t m = feedback_scan(ex, fs);
    uint32_t d = m & ex->fb_dirty;
    if (d) vk_exec_frame.fb_draws++;
    if (d & ~(mode == VK_FEEDBACK_LAYOUT ? ex->fbl_mask : 0)) {
        if (mode == VK_FEEDBACK_LAYOUT)
            for (uint32_t i = 0; i < 9; i++) {
                const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
                if ((d >> i & 1) && t->img && t->img->fb_usage) ((VkTexImage *)t->img)->fb_learned = 1;
            }
        vk_exec_pass_end(ex);
        vk_exec_pass_begin(ex);   // learned targets now in the feedback layout; the full barrier cleared fb_dirty
        vk_exec_frame.fb_restarts++;
    }
    return mode == VK_FEEDBACK_LAYOUT ? m & ex->fbl_mask : 0;
}

// MKW_PIPE_SKIP: which draws are skipped, rather than waited for, while their shaders or pipeline build.
// strict (0): only draws whose pass targets were all rendered in the previous presented frame; a pass with a target
//   that was not counts as one-shot and waits.
// any (1): also draws in a pass with at least one target rendered in the previous frame (a per-frame depth
//   buffer or g-buffer with a colour target the frame before did not use: half-resolution effect buffers in fights).
//   Passes with none (volumes, LUTs, shadow layers seen for the first time) still wait.
// fullscreen (2, default): as any, and full-screen draws into such passes are skipped instead of waiting from their
//   budget. A full-screen pass whose stage libraries are cold in the driver cache took 86 ms on average (up to 194 ms)
//   in a fight on an RX 9070: three frames in a row each spent the 50 ms budget, and some of the draws were skipped
//   after all. Skipping costs the pass for the frames until the pipeline is ready.
static int pipe_skip_mode(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *e = cfg_value("MKW_PIPE_SKIP");
        mode = !strcmp(e, "strict") || !strcmp(e, "0") ? 0 : !strcmp(e, "fullscreen") || !strcmp(e, "2") ? 2 : 1;
        static const char *const what[3] = {"only draws into per-frame passes (MKW_PIPE_SKIP=strict)",
                                            "draws into passes with a target the previous frame rendered",
                                            "those and full-screen draws (MKW_PIPE_SKIP=fullscreen)"};
        rt_log("vk_exec: draws skipped while their pipeline builds: %s", what[mode]);
    }
    return mode;
}

// The previous draw's shaders, kept while its program and bound pipeline state stay the same: shader_for reads nothing
// else (the vertex attribute formats are in the state), so it would return the same shaders. MKW_SHADER_MEMO=0 turns
// it off; =check looks the shaders up on every hit too and stops on a difference.
typedef struct {
    const prog_t *prog;
    uint64_t serial;
    VkPipeState pipe;
} MemoKey;
typedef struct { int on, valid, has_fs; MemoKey k; const GfxShader *vs, *fs; } DrawSM;
static _Thread_local DrawSM *dtl_sm;
#define SM (*dtl_sm)

static void memo_key(const VkExec *ex, MemoKey *k) {
    k->prog = ex->prog, k->serial = ex->prog->serial;
    memcpy(&k->pipe, &ex->pipe, sizeof k->pipe);
}

static int memo_match(const VkExec *ex, const MemoKey *k) {
    return k->prog == ex->prog && k->serial == ex->prog->serial && !memcmp(&k->pipe, &ex->pipe, sizeof k->pipe);
}

static int shader_memo_on(void) {
    if (SM.on < 0) {
        const char *e = cfg_value("MKW_SHADER_MEMO");
        SM.on = !strcmp(e, "check") ? 2 : strcmp(e, "0") != 0;
    }
    return SM.on;
}

typedef struct { int on, valid; VkPipeState s; const VkPipeInfo *info; } DrawPM;
static _Thread_local DrawPM *dtl_pm;
#define PM (*dtl_pm)

static int pipe_memo_on(void) {
    if (PM.on < 0) PM.on = cfg_bool("MKW_PIPE_MEMO");
    return PM.on;
}

// Pass, shaders, pipeline, fixed state and resources for a draw. 0: the draw is skipped.
// mesh: a DrawMeshTasks draw, whose stages are the task (NVN 6, optional) and mesh (7) programs instead of the vertex
// shader; VkPipeState.vs holds the first of them and .mesh the mesh program behind a task program.

static _Thread_local uint64_t pd_t;
_Thread_local uint64_t pd_tsc[PD_N], pd_n;
#define PD_MARK(i) do { if (pd_on) { uint64_t t_ = __rdtsc(); pd_tsc[i] += t_ - pd_t; pd_t = t_; } } while (0)

int prepare_draw_ex(VkExec *ex, int prim, int32_t count, int restart, int mesh) {
    pd_on = (rt_trace & RT_TRACE_DRAW) != 0;
    if (pd_on) pd_t = __rdtsc(), pd_n++;
    if (ex->render_disabled) {
        if (vk_cond_trace > 0) {
            int has_vs;
            const GfxShader *vs = shader_for(ex, NVN_STAGE_VS, &has_vs);
            const uint8_t *k = vs ? gfx_shader_input_id(vs) : NULL;
            vk_cond_trace--;
            rt_log("cond: f%llu skipped draw vs %02x%02x%02x%02x", (unsigned long long)vk_exec_presented, k ? k[0] : 0,
                   k ? k[1] : 0, k ? k[2] : 0, k ? k[3] : 0);
        }
        return vk_exec_frame.skip[VK_SKIP_OFF]++, 0;
    }
    cb_res(ex);
    dummies_init();
    if (ex->pass_noatt) {   // no attachment: the pass covers this draw's viewport or scissor, reopened when it needs more
        uint32_t w, h;
        vk_exec_noatt_dims(ex, &w, &h);
        if (ex->in_pass && (w > ex->pass_w || h > ex->pass_h)) vk_exec_pass_end(ex);
        if (!ex->in_pass) ex->rt_resolved = 0;
    }
    if (!ex->in_pass) ex->dirty |= VK_EXEC_DIRTY_VIEWPORT | VK_EXEC_DIRTY_SCISSOR;
    vk_exec_pass_begin(ex);
    VkRect2D sc = vk_exec_scissor(ex);
    if (!sc.extent.width || !sc.extent.height) return vk_exec_frame.skip[VK_SKIP_SCISSOR]++, 0;
    if (!ex->prog) return vk_exec_frame.skip[VK_SKIP_PROGRAM]++, warn_once(W_PROGRAM, "a draw without a bound program is skipped"), 0;
    PD_MARK(PD_HEAD);
    int has_vs, has_fs, memo = mesh ? 0 : shader_memo_on();
    const GfxShader *vs, *fs, *msh = NULL;
    if (mesh) {   // no memo: the memo key holds the vertex and fragment stages only
        vs = mesh_stages(ex, &msh, &has_vs);
        fs = shader_for(ex, NVN_STAGE_FS, &has_fs);
    } else if (memo && SM.valid && memo_match(ex, &SM.k)) {
        vs = SM.vs, fs = SM.fs, has_vs = 1, has_fs = SM.has_fs;
        if (memo == 2) {
            int hv, hf;
            const GfxShader *v2 = shader_for(ex, NVN_STAGE_VS, &hv), *f2 = shader_for(ex, NVN_STAGE_FS, &hf);
            if (v2 != vs || f2 != fs || hf != has_fs)
                rt_fatal("vk_exec: shader memo of program %llu differs from the lookup", (unsigned long long)ex->prog->serial);
        }
    } else {
        SM.valid = 0;
        vs = shader_for(ex, NVN_STAGE_VS, &has_vs);
        fs = shader_for(ex, NVN_STAGE_FS, &has_fs);
        if (memo && vs && (!has_fs || fs))
            memo_key(ex, &SM.k), SM.vs = vs, SM.fs = fs, SM.has_fs = has_fs, SM.valid = 1;
    }
    if (!has_vs) return vk_exec_frame.skip[VK_SKIP_NO_VS]++, warn_once(W_NO_VS, "a draw with a program that has no vertex stage is skipped"), 0;
    // A draw into a target the previous frame did not render (or into none) waits for a pending pipeline: one-shot
    // draws (volumes and LUTs generated once, later only sampled) would leave their target empty for the session. A
    // draw into per-frame targets is skipped until it is ready and comes back next frame.
    int oneshot = -1, skip = pipe_skip_mode();   // -1: not looked up
    if (!vs || (has_fs && !fs)) return vk_exec_frame.skip[VK_SKIP_SHADER_MISSING]++, 0;
    PD_MARK(PD_SHADER);
    uint32_t fbl = feedback_prepare(ex, fs);
    if (ru_cap) ru_note_fbl(fbl);
    PD_MARK(PD_FEEDBACK);
    VkPipeState s = ex->pipe;
    s.vs = vs, s.fs = fs, s.mesh = msh;
    if (mesh == 2) {   // task/mesh as compute: the draw of the vertices its passes wrote (op_draw_mesh_cs)
        if (!(s.vs = gfx_shader_mesh_cs_vs())) return 0;
        s.mesh = NULL;
    }
    int spec = 0;   // s holds a specialized variant (MKW_SPEC): the generic shaders while its pipeline is built
    if (vk_spec_mode() && !msh) {
        s.vs = spec_pick(ex, vs, NVN_STAGE_VS), s.fs = spec_pick(ex, fs, NVN_STAGE_FS);
        spec = s.vs != vs || s.fs != fs;
    }
    s.feedback = (uint8_t)((fbl & 0xff ? 1 : 0) | (fbl & 0x100 ? 2 : 0));
    s.alpha_ref = s.fs && s.color.alpha_func && s.color.alpha_func != NVN_ALPHA_FUNC_ALWAYS ? ex->alpha_ref : 0;
    s.prim = (uint8_t)prim;
    s.depth_minus_w = (uint8_t)(nvn_depth_minus_w() != 0);
    s.y_flip = 1;
    s.prim_restart = (uint8_t)(restart != 0);
    const VkPipeInfo *info = NULL;
    int pending;
    // A draw of at most 6 vertices over half the render area or more (a full-screen triangle or quad, a tile) waits for
    // its pipeline: skipping it can leave a black tile. The waits share 50 ms per presented frame
    // (MKW_PIPE_SKIP=fullscreen skips them instead). A one-shot draw waits without that budget. Everything else is
    // skipped while its pipeline is built.
    static _Thread_local uint64_t wait_frame;
    static _Thread_local uint64_t wait_left_ns;
    uint32_t wait_ms = 0;
    uint64_t t0 = 0;
    if (oneshot > 0) {
        wait_ms = ONESHOT_WAIT_MS;
    } else if (skip < 2 && count <= 6 && (uint64_t)sc.extent.width * sc.extent.height * 2 >= (uint64_t)ex->pass_w * ex->pass_h) {
        uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (f != wait_frame) wait_frame = f, wait_left_ns = 50000000;
        wait_ms = (uint32_t)(wait_left_ns / 1000000);
        if (wait_ms) t0 = kern_now_ns();
    }
    // The previous draw's pipeline while its state repeats (PM, MKW_PIPE_MEMO=0 turns it off): the cache key is made
    // from the state alone, and a pipeline once found only changes from its fast link to its optimised build, read
    // from the info the cache entry keeps.
    VkPipeline p;
    const char *why = NULL;
    uint64_t w0 = 0;
    if (pipe_memo_on() && PM.valid && !memcmp(&PM.s, &s, sizeof s)) {
        info = PM.info, pending = 0;
        p = __atomic_load_n(&info->pipeline, __ATOMIC_ACQUIRE);
    } else {
        w0 = t0 ? t0 : wait_ms ? kern_now_ns() : 0;   // the clock is read only for draws that may wait
        p = spec ? vk_pipeline_graphics_async_low(&s, &info, &pending) : vk_pipeline_graphics_async(&s, &info, &pending, wait_ms);
        if (pending && spec) {   // the variant's pipeline is being built: this draw goes with the generic shaders
            s.vs = vs, s.fs = fs;
            p = vk_pipeline_graphics_async(&s, &info, &pending, wait_ms);
        }
        if (t0) {
            uint64_t spent = kern_now_ns() - t0;
            wait_left_ns = spent < wait_left_ns ? wait_left_ns - spent : 0;
        }
        why = oneshot > 0 ? "one-shot, shader" : wait_ms ? "full-screen" : NULL;
        if (pending && oneshot < 0 && !vk_exec_targets_recent(ex, skip >= 1)) {
            perf_count(PERF_CNT_ONESHOT_WAIT);
            why = "one-shot";
            if (!w0) w0 = kern_now_ns();
            p = vk_pipeline_graphics_async(&s, &info, &pending, ONESHOT_WAIT_MS);
        }
        PM.valid = !pending && p && info;
        if (PM.valid) PM.s = s, PM.info = info;
    }
    uint64_t waited = w0 ? kern_now_ns() - w0 : 0;
    static int wait_logs;
    if (why && waited >= 2000000 && wait_logs < 200) {
        char tg[512];
        wait_logs++;
        vk_exec_targets_describe(ex, tg, sizeof tg);
        rt_log("vk_exec: frame %llu: a draw waited %.1f ms for its pipeline (%s%s): %u vertices, pass %ux%u, %s",
               (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1, waited / 1e6, why,
               pending ? ", skipped after all" : "", count, ex->pass_w, ex->pass_h, tg);
    }
    if (pending) return vk_exec_frame.skip[VK_SKIP_PIPE_WAIT]++, 0;   // a background worker is building it: the draw is skipped until it is ready
    if (!p || !info) return vk_exec_frame.skip[VK_SKIP_PIPE_FAILED]++, warn_once(W_PIPELINE, "a graphics pipeline could not be created: draws with it are skipped"), 0;
    if (skip_listed(vs, fs)) return 0;
    PD_MARK(PD_PIPE);
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    recent_note(vs, fs);
    vk_crumb_draw(ex->cb, gfx_shader_key(vs), fs ? gfx_shader_key(fs) : NULL);
    if (!B.st_valid || memcmp(&B.st, &s, sizeof s)) {
        vk_pipeline_set_dynamic(ex->cb, &s);
        ex->recorded++;
        B.st = s, B.st_valid = 1;
    }
    record_fixed(ex);
    repeat_stats(ex, info);
    PD_MARK(PD_STATE);
    bind_resources(ex, info);
    PD_MARK(PD_RES);
    if (fbl & ex->fb_dirty) feedback_barrier(ex);
    feedback_written(ex, &s);
    bind_vertex_buffers(ex, msh ? 0xffff : vk_exec_streams(ex, info));
    if (vk_dump_armed) vk_dump_draw(ex, &s, p, info);
    vk_exec_oq_draw(ex);
    if (vk_pt_enabled() && vk_pt_draw_mode()) vk_pt_draw(ex->cb, s.vs, s.fs);
    if (vk_spec_census_on()) spec_census(ex, vs, NVN_STAGE_VS), spec_census(ex, fs, NVN_STAGE_FS);
    PD_MARK(PD_VB);
    return 1;
}

struct DrawTls {
    DrawB b;
    DrawSM sm;
    DrawPM pm;
    ShaderEnt shader_tab[SHADER_SETS][SHADER_WAYS];
    StageM stagem[STAGEM_N];
    FeedMemo fm[FM_N];
};

// This thread's block, made at its first recording (never freed: the replay threads live for the run).
void vk_exec_draw_thread(void) {
    if (dtl_b) return;
    struct DrawTls *t = calloc(1, sizeof *t);
    if (!t) rt_fatal("vk: out of memory (replay thread state)");
    t->sm.on = t->pm.on = -1;
    dtl_b = &t->b, dtl_sm = &t->sm, dtl_pm = &t->pm;
    shader_tab = t->shader_tab, stagem = t->stagem, FM = t->fm;
    vk_exec_bind_thread();
}
