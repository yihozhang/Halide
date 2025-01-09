declare void @llvm.nvvm.barrier0()
declare  i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare  i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare  i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare  i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
declare  i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare  i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare  i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare  i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
declare  i32 @llvm.nvvm.read.ptx.sreg.tid.z()
declare  i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
declare  i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
declare  i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
declare  i32 @llvm.nvvm.read.ptx.sreg.tid.w()
declare  i32 @llvm.nvvm.read.ptx.sreg.ctaid.w()
declare  i32 @llvm.nvvm.read.ptx.sreg.ntid.w()
declare  i32 @llvm.nvvm.read.ptx.sreg.nctaid.w()
declare  i32 @llvm.nvvm.read.ptx.sreg.warpsize()

; Remove these two once the minimum required llvm version is 9.0
declare float @llvm.nvvm.atomic.load.add.f32.p0f32(float*, float)
declare double @llvm.nvvm.atomic.load.add.f64.p0f64(double *, double)

; Legacy - to replace
;declare void @llvm.ptx.red.global.add.s32(i32*, i32)
;declare void @llvm.ptx.red.global.add.f32(float*, float)
;declare void @llvm.ptx.red.shared.add.s32(i32 addrspace(4)*, i32)

define weak_odr float @nan_f32() nounwind uwtable readnone alwaysinline {
       ret float 0x7FF8000000000000;
}

define weak_odr float @neg_inf_f32() nounwind uwtable readnone alwaysinline {
       ret float 0xFFF0000000000000;
}

define weak_odr float @inf_f32() nounwind uwtable readnone alwaysinline {
       ret float 0x7FF0000000000000;
}

declare float @__nv_sqrtf(float) nounwind readnone
declare double @__nv_sqrt(double) nounwind readnone

define weak_odr float @sqrt_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @__nv_sqrtf(float %x) nounwind readnone
       ret float %y
}

define weak_odr double @sqrt_f64(double %x) nounwind uwtable readnone alwaysinline {
       %y = tail call double @__nv_sqrt(double %x) nounwind readnone
       ret double %y
}

declare float @__nv_frcp_rn(float) nounwind readnone

define weak_odr float @fast_inverse_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @__nv_frcp_rn(float %x) nounwind readnone
       ret float %y
}

declare float @llvm.nvvm.rsqrt.approx.ftz.f(float) nounwind readnone

define weak_odr float @fast_inverse_sqrt_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @llvm.nvvm.rsqrt.approx.ftz.f(float %x) nounwind readnone
       ret float %y
}

declare float @__nv_sinf(float) nounwind readnone
declare double @__nv_sin(double) nounwind readnone

define weak_odr float @sin_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @__nv_sinf(float %x) nounwind readnone
       ret float %y
}

define weak_odr double @sin_f64(double %x) nounwind uwtable readnone alwaysinline {
       %y = tail call double @__nv_sin(double %x) nounwind readnone
       ret double %y
}

declare float @__nv_cosf(float) nounwind readnone
declare double @__nv_cos(double) nounwind readnone

define weak_odr float @cos_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @__nv_cosf(float %x) nounwind readnone
       ret float %y
}

define weak_odr double @cos_f64(double %x) nounwind uwtable readnone alwaysinline {
       %y = tail call double @__nv_cos(double %x) nounwind readnone
       ret double %y
}

declare float @__nv_expf(float) nounwind readnone
declare double @__nv_exp(double) nounwind readnone

define weak_odr float @exp_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @__nv_expf(float %x) nounwind readnone
       ret float %y
}

define weak_odr double @exp_f64(double %x) nounwind uwtable readnone alwaysinline {
       %y = tail call double @__nv_exp(double %x) nounwind readnone
       ret double %y
}

declare float @__nv_logf(float) nounwind readnone
declare double @__nv_log(double) nounwind readnone

define weak_odr float @log_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @__nv_logf(float %x) nounwind readnone
       ret float %y
}

define weak_odr double @log_f64(double %x) nounwind uwtable readnone alwaysinline {
       %y = tail call double @__nv_log(double %x) nounwind readnone
       ret double %y
}

declare float @__nv_fabsf(float) nounwind readnone
declare double @__nv_fabs(double) nounwind readnone

define weak_odr float @abs_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @__nv_fabsf(float %x) nounwind readnone
       ret float %y
}

define weak_odr double @abs_f64(double %x) nounwind uwtable readnone alwaysinline {
       %y = tail call double @__nv_fabs(double %x) nounwind readnone
       ret double %y
}

declare float @__nv_floorf(float) nounwind readnone
declare double @__nv_floor(double) nounwind readnone

define weak_odr float @floor_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @__nv_floorf(float %x) nounwind readnone
       ret float %y
}

define weak_odr double @floor_f64(double %x) nounwind uwtable readnone alwaysinline {
       %y = tail call double @__nv_floor(double %x) nounwind readnone
       ret double %y
}

declare float @__nv_ceilf(float) nounwind readnone
declare double @__nv_ceil(double) nounwind readnone

define weak_odr float @ceil_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @__nv_ceilf(float %x) nounwind readnone
       ret float %y
}

define weak_odr double @ceil_f64(double %x) nounwind uwtable readnone alwaysinline {
       %y = tail call double @__nv_ceil(double %x) nounwind readnone
       ret double %y
}

declare float @__nv_truncf(float) nounwind readnone
declare double @__nv_trunc(double) nounwind readnone

define weak_odr float @trunc_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @__nv_truncf(float %x) nounwind readnone
       ret float %y
}

define weak_odr double @trunc_f64(double %x) nounwind uwtable readnone alwaysinline {
       %y = tail call double @__nv_trunc(double %x) nounwind readnone
       ret double %y
}

declare float @__nv_powf(float, float) nounwind readnone
declare double @__nv_pow(double, double) nounwind readnone

define weak_odr float @pow_f32(float %x, float %y) nounwind uwtable readnone alwaysinline {
       %z = tail call float @__nv_powf(float %x, float %y) nounwind readnone
       ret float %z
}

define weak_odr double @pow_f64(double %x, double %y) nounwind uwtable readnone alwaysinline {
       %z = tail call double @__nv_pow(double %x, double %y) nounwind readnone
       ret double %z
}

declare float @__nv_asinf(float) nounwind readnone
declare double @__nv_asin(double) nounwind readnone

define weak_odr float @asin_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @__nv_asinf(float %x) nounwind readnone
       ret float %y
}

define weak_odr double @asin_f64(double %x) nounwind uwtable readnone alwaysinline {
       %y = tail call double @__nv_asin(double %x) nounwind readnone
       ret double %y
}

declare float @__nv_acosf(float) nounwind readnone
declare double @__nv_acos(double) nounwind readnone

define weak_odr float @acos_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @__nv_acosf(float %x) nounwind readnone
       ret float %y
}

define weak_odr double @acos_f64(double %x) nounwind uwtable readnone alwaysinline {
       %y = tail call double @__nv_acos(double %x) nounwind readnone
       ret double %y
}

declare float @__nv_tanf(float) nounwind readnone
declare double @__nv_tan(double) nounwind readnone

define weak_odr float @tan_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @__nv_tanf(float %x) nounwind readnone
       ret float %y
}

define weak_odr double @tan_f64(double %x) nounwind uwtable readnone alwaysinline {
       %y = tail call double @__nv_tan(double %x) nounwind readnone
       ret double %y
}

declare float @__nv_atanf(float) nounwind readnone
declare double @__nv_atan(double) nounwind readnone

define weak_odr float @atan_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @__nv_atanf(float %x) nounwind readnone
       ret float %y
}

define weak_odr double @atan_f64(double %x) nounwind uwtable readnone alwaysinline {
       %y = tail call double @__nv_atan(double %x) nounwind readnone
       ret double %y
}

declare float @__nv_atan2f(float, float) nounwind readnone
declare double @__nv_atan2(double, double) nounwind readnone

define weak_odr float @atan2_f32(float %y, float %x) nounwind uwtable readnone alwaysinline {
       %z = tail call float @__nv_atan2f(float %y, float %x) nounwind readnone
       ret float %z
}

define weak_odr double @atan2_f64(double %y, double %x) nounwind uwtable readnone alwaysinline {
       %z = tail call double @__nv_atan2(double %y, double %x) nounwind readnone
       ret double %z
}

declare float @__nv_sinhf(float) nounwind readnone
declare double @__nv_sinh(double) nounwind readnone

define weak_odr float @sinh_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @__nv_sinhf(float %x) nounwind readnone
       ret float %y
}

define weak_odr double @sinh_f64(double %x) nounwind uwtable readnone alwaysinline {
       %y = tail call double @__nv_sinh(double %x) nounwind readnone
       ret double %y
}

declare float @__nv_asinhf(float) nounwind readnone
declare double @__nv_asinh(double) nounwind readnone

define weak_odr float @asinh_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @__nv_asinhf(float %x) nounwind readnone
       ret float %y
}

define weak_odr double @asinh_f64(double %x) nounwind uwtable readnone alwaysinline {
       %y = tail call double @__nv_asinh(double %x) nounwind readnone
       ret double %y
}

declare float @__nv_coshf(float) nounwind readnone
declare double @__nv_cosh(double) nounwind readnone

define weak_odr float @cosh_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @__nv_coshf(float %x) nounwind readnone
       ret float %y
}

define weak_odr double @cosh_f64(double %x) nounwind uwtable readnone alwaysinline {
       %y = tail call double @__nv_cosh(double %x) nounwind readnone
       ret double %y
}

declare float @__nv_acoshf(float) nounwind readnone
declare double @__nv_acosh(double) nounwind readnone

define weak_odr float @acosh_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @__nv_acoshf(float %x) nounwind readnone
       ret float %y
}

define weak_odr double @acosh_f64(double %x) nounwind uwtable readnone alwaysinline {
       %y = tail call double @__nv_acosh(double %x) nounwind readnone
       ret double %y
}

declare float @__nv_tanhf(float) nounwind readnone
declare double @__nv_tanh(double) nounwind readnone

define weak_odr float @tanh_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @__nv_tanhf(float %x) nounwind readnone
       ret float %y
}

define weak_odr double @tanh_f64(double %x) nounwind uwtable readnone alwaysinline {
       %y = tail call double @__nv_tanh(double %x) nounwind readnone
       ret double %y
}

declare float @__nv_atanhf(float) nounwind readnone
declare double @__nv_atanh(double) nounwind readnone

define weak_odr float @atanh_f32(float %x) nounwind uwtable readnone alwaysinline {
       %y = tail call float @__nv_atanhf(float %x) nounwind readnone
       ret float %y
}

define weak_odr double @atanh_f64(double %x) nounwind uwtable readnone alwaysinline {
       %y = tail call double @__nv_atanh(double %x) nounwind readnone
       ret double %y
}

define weak_odr i32 @halide_ptx_trap() nounwind uwtable alwaysinline {
       tail call void asm sideeffect "
       trap;
       ", ""() nounwind
       ret i32 0
}

; llvm doesn't expose dot product instructions as intrinsics
define weak_odr i32 @dp4a_s32_s32(<4 x i8> %a, <4 x i8> %b, i32 %i) nounwind readnone alwaysinline {
       %a_32 = bitcast <4 x i8> %a to i32
       %b_32 = bitcast <4 x i8> %b to i32
       %d = tail call i32 asm "dp4a.s32.s32    $0, $1, $2, $3;", "=r,r,r,r"(i32 %a_32, i32 %b_32, i32 %i) nounwind readnone
       ret i32 %d
}

define weak_odr i32 @dp4a_s32_u32(<4 x i8> %a, <4 x i8> %b, i32 %i) nounwind readnone alwaysinline {
       %a_32 = bitcast <4 x i8> %a to i32
       %b_32 = bitcast <4 x i8> %b to i32
       %d = tail call i32 asm "dp4a.s32.u32    $0, $1, $2, $3;", "=r,r,r,r"(i32 %a_32, i32 %b_32, i32 %i) nounwind readnone
       ret i32 %d
}

define weak_odr i32 @dp4a_u32_s32(<4 x i8> %a, <4 x i8> %b, i32 %i) nounwind readnone alwaysinline {
       %a_32 = bitcast <4 x i8> %a to i32
       %b_32 = bitcast <4 x i8> %b to i32
       %d = tail call i32 asm "dp4a.u32.s32    $0, $1, $2, $3;", "=r,r,r,r"(i32 %a_32, i32 %b_32, i32 %i) nounwind readnone
       ret i32 %d
}

define weak_odr i32 @dp4a_u32_u32(<4 x i8> %a, <4 x i8> %b, i32 %i) nounwind readnone alwaysinline {
       %a_32 = bitcast <4 x i8> %a to i32
       %b_32 = bitcast <4 x i8> %b to i32
       %d = tail call i32 asm "dp4a.u32.u32    $0, $1, $2, $3;", "=r,r,r,r"(i32 %a_32, i32 %b_32, i32 %i) nounwind readnone
       ret i32 %d
}


define weak_odr i32 @dp2a_s32_s32(<4 x i16> %a, <4 x i8> %b, i32 %i) nounwind readnone alwaysinline {
       %a_32 = bitcast <4 x i16> %a to <2 x i32>
       %a_lo = extractelement <2 x i32> %a_32, i32 0
       %a_hi = extractelement <2 x i32> %a_32, i32 1
       %b_32 = bitcast <4 x i8> %b to i32
       %d = tail call i32 asm "dp2a.lo.s32.s32    $0, $1, $3, $4; dp2a.hi.s32.s32    $0, $2, $3, $0;", "=r,r,r,r,r"(i32 %a_lo, i32 %a_hi, i32 %b_32, i32 %i) nounwind readnone
       ret i32 %d
}

define weak_odr i32 @dp2a_s32_u32(<4 x i16> %a, <4 x i8> %b, i32 %i) nounwind readnone alwaysinline {
       %a_32 = bitcast <4 x i16> %a to <2 x i32>
       %a_lo = extractelement <2 x i32> %a_32, i32 0
       %a_hi = extractelement <2 x i32> %a_32, i32 1
       %b_32 = bitcast <4 x i8> %b to i32
       %d = tail call i32 asm "dp2a.lo.s32.u32    $0, $1, $3, $4; dp2a.hi.s32.u32    $0, $2, $3, $0;", "=r,r,r,r,r"(i32 %a_lo, i32 %a_hi, i32 %b_32, i32 %i) nounwind readnone
       ret i32 %d
}

define weak_odr i32 @dp2a_u32_s32(<4 x i16> %a, <4 x i8> %b, i32 %i) nounwind readnone alwaysinline {
       %a_32 = bitcast <4 x i16> %a to <2 x i32>
       %a_lo = extractelement <2 x i32> %a_32, i32 0
       %a_hi = extractelement <2 x i32> %a_32, i32 1
       %b_32 = bitcast <4 x i8> %b to i32
       %d = tail call i32 asm "dp2a.lo.u32.s32    $0, $1, $3, $4; dp2a.hi.u32.s32    $0, $2, $3, $0;", "=r,r,r,r,r"(i32 %a_lo, i32 %a_hi, i32 %b_32, i32 %i) nounwind readnone
       ret i32 %d
}

define weak_odr i32 @dp2a_u32_u32(<4 x i16> %a, <4 x i8> %b, i32 %i) nounwind readnone alwaysinline {
       %a_32 = bitcast <4 x i16> %a to <2 x i32>
       %a_lo = extractelement <2 x i32> %a_32, i32 0
       %a_hi = extractelement <2 x i32> %a_32, i32 1
       %b_32 = bitcast <4 x i8> %b to i32
       %d = tail call i32 asm "dp2a.lo.u32.u32    $0, $1, $3, $4; dp2a.hi.u32.u32    $0, $2, $3, $0;", "=r,r,r,r,r"(i32 %a_lo, i32 %a_hi, i32 %b_32, i32 %i) nounwind readnone
       ret i32 %d
}

declare { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16(i8* nocapture readonly, i32) 
declare { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16(i8* nocapture readonly, i32)  
declare { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.f32(i8* nocapture readonly, i32) 
; declare { i32, i32, i32, i32} @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.f16.p0i32(i32* nocapture readonly , i32) 
declare { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, float, float, float, float, float, float, float, float) nounwind readnone
declare void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32(i8* nocapture writeonly, float, float, float, float, float, float, float, float, i32) 


; declare { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0i32(i32* nocapture readonly, i32) 
; declare { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16.p0i32(i32* nocapture readonly, i32)  
; declare { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.f32.p0f32(float* nocapture readonly, i32) 
; declare { <2 x half>, <2 x half>, <2 x half>, <2 x half>} @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.f16.p0i32(i32* nocapture readonly , i32) 
; declare { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, float, float, float, float, float, float, float, float) nounwind readnone

define weak_odr <8 x i32> @adapted.llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16(i8* %ptr, i32 %offset, i32 %stride) nounwind readnone alwaysinline {
  %start = getelementptr i8, i8* %ptr, i32 %offset
  %result = tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16(i8* %start, i32 %stride)
  %v0 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %result, 0
  %v1 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %result, 1
  %v2 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %result, 2
  %v3 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %result, 3
  %v4 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %result, 4
  %v5 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %result, 5
  %v6 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %result, 6
  %v7 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %result, 7

  %v0i32 = bitcast <2 x half> %v0 to i32
  %v1i32 = bitcast <2 x half> %v1 to i32
  %v2i32 = bitcast <2 x half> %v2 to i32
  %v3i32 = bitcast <2 x half> %v3 to i32
  %v4i32 = bitcast <2 x half> %v4 to i32
  %v5i32 = bitcast <2 x half> %v5 to i32
  %v6i32 = bitcast <2 x half> %v6 to i32
  %v7i32 = bitcast <2 x half> %v7 to i32

  %vec0 = insertelement <8 x i32> poison, i32 %v0i32, i32 0
  %vec1 = insertelement <8 x i32> %vec0, i32 %v1i32, i32 1
  %vec2 = insertelement <8 x i32> %vec1, i32 %v2i32, i32 2
  %vec3 = insertelement <8 x i32> %vec2, i32 %v3i32, i32 3
  %vec4 = insertelement <8 x i32> %vec3, i32 %v4i32, i32 4
  %vec5 = insertelement <8 x i32> %vec4, i32 %v5i32, i32 5
  %vec6 = insertelement <8 x i32> %vec5, i32 %v6i32, i32 6
  %vec7 = insertelement <8 x i32> %vec6, i32 %v7i32, i32 7
  ret <8 x i32> %vec7
}

define weak_odr <8 x i32> @adapted.llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16(i8* %ptr, i32 %offset, i32 %stride) nounwind readnone alwaysinline {
  %start = getelementptr i8, i8* %ptr, i32 %offset
  %result = tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16(i8* %start, i32 %stride)
  %v0 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %result, 0
  %v1 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %result, 1
  %v2 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %result, 2
  %v3 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %result, 3
  %v4 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %result, 4
  %v5 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %result, 5
  %v6 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %result, 6
  %v7 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %result, 7

  %v0i32 = bitcast <2 x half> %v0 to i32
  %v1i32 = bitcast <2 x half> %v1 to i32
  %v2i32 = bitcast <2 x half> %v2 to i32
  %v3i32 = bitcast <2 x half> %v3 to i32
  %v4i32 = bitcast <2 x half> %v4 to i32
  %v5i32 = bitcast <2 x half> %v5 to i32
  %v6i32 = bitcast <2 x half> %v6 to i32
  %v7i32 = bitcast <2 x half> %v7 to i32

  %vec0 = insertelement <8 x i32> poison, i32 %v0i32, i32 0
  %vec1 = insertelement <8 x i32> %vec0, i32 %v1i32, i32 1
  %vec2 = insertelement <8 x i32> %vec1, i32 %v2i32, i32 2
  %vec3 = insertelement <8 x i32> %vec2, i32 %v3i32, i32 3
  %vec4 = insertelement <8 x i32> %vec3, i32 %v4i32, i32 4
  %vec5 = insertelement <8 x i32> %vec4, i32 %v5i32, i32 5
  %vec6 = insertelement <8 x i32> %vec5, i32 %v6i32, i32 6
  %vec7 = insertelement <8 x i32> %vec6, i32 %v7i32, i32 7
  ret <8 x i32> %vec7
}

define weak_odr <8 x float> @adapted.llvm.nvvm.wmma.m16n16k16.load.c.row.stride.f32(i8* %ptr, i32 %offset, i32 %stride) nounwind readnone alwaysinline {
  %start = getelementptr i8, i8* %ptr, i32 %offset
  %result = tail call {float, float, float, float, float, float, float, float} @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.f32(i8* %start, i32 %stride)

  %v0 = extractvalue {float, float, float, float, float, float, float, float} %result, 0
  %v1 = extractvalue {float, float, float, float, float, float, float, float} %result, 1
  %v2 = extractvalue {float, float, float, float, float, float, float, float} %result, 2
  %v3 = extractvalue {float, float, float, float, float, float, float, float} %result, 3
  %v4 = extractvalue {float, float, float, float, float, float, float, float} %result, 4
  %v5 = extractvalue {float, float, float, float, float, float, float, float} %result, 5
  %v6 = extractvalue {float, float, float, float, float, float, float, float} %result, 6
  %v7 = extractvalue {float, float, float, float, float, float, float, float} %result, 7

  %vec0 = insertelement <8 x float> poison, float %v0, i32 0
  %vec1 = insertelement <8 x float> %vec0, float %v1, i32 1
  %vec2 = insertelement <8 x float> %vec1, float %v2, i32 2
  %vec3 = insertelement <8 x float> %vec2, float %v3, i32 3
  %vec4 = insertelement <8 x float> %vec3, float %v4, i32 4
  %vec5 = insertelement <8 x float> %vec4, float %v5, i32 5
  %vec6 = insertelement <8 x float> %vec5, float %v6, i32 6
  %vec7 = insertelement <8 x float> %vec6, float %v7, i32 7
  ret <8 x float> %vec7
}

define weak_odr <8 x float> @adapted.llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<8 x i32> %a, <8 x i32> %b, <8 x float> %c) nounwind readnone alwaysinline {
  %a0 = extractelement <8 x i32> %a, i32 0
  %a1 = extractelement <8 x i32> %a, i32 1
  %a2 = extractelement <8 x i32> %a, i32 2
  %a3 = extractelement <8 x i32> %a, i32 3
  %a4 = extractelement <8 x i32> %a, i32 4
  %a5 = extractelement <8 x i32> %a, i32 5
  %a6 = extractelement <8 x i32> %a, i32 6
  %a7 = extractelement <8 x i32> %a, i32 7

  %a0half = bitcast i32 %a0 to <2 x half>
  %a1half = bitcast i32 %a1 to <2 x half>
  %a2half = bitcast i32 %a2 to <2 x half>
  %a3half = bitcast i32 %a3 to <2 x half>
  %a4half = bitcast i32 %a4 to <2 x half>
  %a5half = bitcast i32 %a5 to <2 x half>
  %a6half = bitcast i32 %a6 to <2 x half>
  %a7half = bitcast i32 %a7 to <2 x half>

  %b0 = extractelement <8 x i32> %b, i32 0
  %b1 = extractelement <8 x i32> %b, i32 1
  %b2 = extractelement <8 x i32> %b, i32 2
  %b3 = extractelement <8 x i32> %b, i32 3
  %b4 = extractelement <8 x i32> %b, i32 4
  %b5 = extractelement <8 x i32> %b, i32 5
  %b6 = extractelement <8 x i32> %b, i32 6
  %b7 = extractelement <8 x i32> %b, i32 7

  %b0half = bitcast i32 %b0 to <2 x half>
  %b1half = bitcast i32 %b1 to <2 x half>
  %b2half = bitcast i32 %b2 to <2 x half>
  %b3half = bitcast i32 %b3 to <2 x half>
  %b4half = bitcast i32 %b4 to <2 x half>
  %b5half = bitcast i32 %b5 to <2 x half>
  %b6half = bitcast i32 %b6 to <2 x half>
  %b7half = bitcast i32 %b7 to <2 x half>

  %c0 = extractelement <8 x float> %c, i32 0
  %c1 = extractelement <8 x float> %c, i32 1
  %c2 = extractelement <8 x float> %c, i32 2
  %c3 = extractelement <8 x float> %c, i32 3
  %c4 = extractelement <8 x float> %c, i32 4
  %c5 = extractelement <8 x float> %c, i32 5
  %c6 = extractelement <8 x float> %c, i32 6
  %c7 = extractelement <8 x float> %c, i32 7

  %result = call {float, float, float, float, float, float, float, float} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<2 x half> %a0half, <2 x half> %a1half, <2 x half> %a2half, <2 x half> %a3half, <2 x half> %a4half, <2 x half> %a5half, <2 x half> %a6half, <2 x half> %a7half, <2 x half> %b0half, <2 x half> %b1half, <2 x half> %b2half, <2 x half> %b3half, <2 x half> %b4half, <2 x half> %b5half, <2 x half> %b6half, <2 x half> %b7half, float %c0, float %c1, float %c2, float %c3, float %c4, float %c5, float %c6, float %c7)

  %v0 = extractvalue {float, float, float, float, float, float, float, float} %result, 0
  %v1 = extractvalue {float, float, float, float, float, float, float, float} %result, 1
  %v2 = extractvalue {float, float, float, float, float, float, float, float} %result, 2
  %v3 = extractvalue {float, float, float, float, float, float, float, float} %result, 3
  %v4 = extractvalue {float, float, float, float, float, float, float, float} %result, 4
  %v5 = extractvalue {float, float, float, float, float, float, float, float} %result, 5
  %v6 = extractvalue {float, float, float, float, float, float, float, float} %result, 6
  %v7 = extractvalue {float, float, float, float, float, float, float, float} %result, 7
  %vec0 = insertelement <8 x float> poison, float %v0, i32 0
  %vec1 = insertelement <8 x float> %vec0, float %v1, i32 1
  %vec2 = insertelement <8 x float> %vec1, float %v2, i32 2
  %vec3 = insertelement <8 x float> %vec2, float %v3, i32 3
  %vec4 = insertelement <8 x float> %vec3, float %v4, i32 4
  %vec5 = insertelement <8 x float> %vec4, float %v5, i32 5
  %vec6 = insertelement <8 x float> %vec5, float %v6, i32 6
  %vec7 = insertelement <8 x float> %vec6, float %v7, i32 7
  ret <8 x float> %vec7
}

define weak_odr i32 @adapted.llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32(i8* %ptr, <8 x float> %out, i32 %offset, i32 %stride) nounwind readnone alwaysinline {
  %v0 = extractelement <8 x float> %out, i32 0
  %v1 = extractelement <8 x float> %out, i32 1
  %v2 = extractelement <8 x float> %out, i32 2
  %v3 = extractelement <8 x float> %out, i32 3
  %v4 = extractelement <8 x float> %out, i32 4
  %v5 = extractelement <8 x float> %out, i32 5
  %v6 = extractelement <8 x float> %out, i32 6
  %v7 = extractelement <8 x float> %out, i32 7
  
  %start = getelementptr i8, i8* %ptr, i32 %offset  
  call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32(i8* %start, float %v0, float %v1, float %v2, float %v3, float %v4, float %v5, float %v6, float %v7, i32 %stride)
  ret i32 0
}




; ; Function Attrs: nounwind readnone
; declare { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, float, float, float, float, float, float, float, float) nounwind readnone




; declare void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p0f32(float* nocapture writeonly, float, float, float, float, float, float, float, float, i32) 

; declare void @llvm.nvvm.wmma.m16n16k16.store.d.row.f16.p0i32(i32* nocapture writeonly, <2 x half>, <2 x half>, <2 x half>, <2 x half>) 

; ; Function Attrs: argmemonly nounwind writeonly
; declare void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.s32.p0i32(i32* nocapture writeonly, i32, i32, i32, i32, i32, i32, i32, i32, i32)


; ; Function Attrs: argmemonly nounwind readonly
; declare { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.s32.p0i32(i32* nocapture readonly, i32)

; ; Function Attrs: argmemonly nounwind readonly
; declare { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.u8.p0i32(i32* nocapture readonly, i32)

; ; Function Attrs: argmemonly nounwind readonly
; declare { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.u8.p0i32(i32* nocapture readonly, i32)
; declare { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.b.col.stride.u8.p0i32(i32* nocapture readonly, i32)
; ; Function Attrs: nounwind readnone
; declare { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.row.row.u8(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
; declare { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.row.col.u8(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)



