// Additional lowering operations, included in the lower module.
impl<'a> Lower<'a> {
    /// A 64-bit floating-point source, constructed from its bit pattern via `dx.op.makeDouble`.
    fn src_f64(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let bits = self.fp_bits(inst, k, 64)?;
        let (lo, hi) = self.split64(bits);
        self.s.flags |= flags::DOUBLES;
        let f64t = self.s.m.double();
        Ok(self
            .s
            .op(self.cur, "makeDouble.f64", Attr::ReadNone, f64t, 101, &[lo, hi]))
    }

    /// Writes a 64-bit floating-point result into a VGPR pair via `dx.op.splitDouble`.
    fn write_f64(&mut self, inst: &Inst, val: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("an f64 destination");
        };
        self.s.flags |= flags::DOUBLES;
        let i32t = self.s.t.i32;
        let pair = self.s.m.named_struct("dx.types.splitdouble", vec![i32t, i32t]);
        let split = self
            .s
            .op(self.cur, "splitDouble.f64", Attr::ReadNone, pair, 102, &[val]);
        let lo = self.s.m.extract(self.cur, split, 0);
        let hi = self.s.m.extract(self.cur, split, 1);
        self.write(inst, Reg::V(reg), lo, guard);
        self.write(inst, Reg::V(reg + 1), hi, guard);
        Ok(())
    }

    /// A 64-bit source (a register pair; inline constants sign-extend).
    fn src64(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 || v.sdwa.is_some() || v.dpp.is_some() {
            return Err("64-bit operand modifiers");
        }
        self.raw_src64(inst, k, false)
    }

    fn raw_src64(&mut self, inst: &Inst, k: usize, float: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else { unreachable!() };
        if v.sdwa.is_some() || v.dpp.is_some() {
            return Err("64-bit SDWA or DPP operands");
        }
        let c = self.cur;
        let i64t = self.s.m.int(64);
        self.s.flags |= flags::INT64;
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let pair = |l: &mut Self, r: Reg, r1: Reg| -> Result<V, Why> {
            let lo = l.i32_of(inst.read(r).ok_or("an unread operand")?)?;
            let hi = l.i32_of(inst.read(r1).ok_or("an unread operand")?)?;
            Ok(l.join64(lo, hi))
        };
        match op {
            Operand::Vector { reg, .. } => pair(self, Reg::V(reg), Reg::V(reg + 1)),
            Operand::Scalar { code, .. } if code != NULL => {
                pair(self, Reg::S(code), Reg::S(code + 1))
            }
            Operand::Const(k @ 240..=248) if float => {
                let f = if k == 248 {
                    0.5 * std::f64::consts::FRAC_1_PI
                } else {
                    let b = crate::resource::inline(k).ok_or("a special operand")?;
                    f64::from(f32::from_bits(b))
                };
                Ok(self.s.m.int_const(i64t, f.to_bits() as i64))
            }
            Operand::Literal(x) if float => Ok(self.s.m.int_const(i64t, i64::from(x) << 32)),
            Operand::Literal(x) if inst.name() == "v_mad_i64_i32" => {
                Ok(self.s.m.int_const(i64t, i64::from(x as i32)))
            }
            Operand::Const(k @ 128..=208) => {
                let x = crate::resource::inline(k).ok_or("a special operand")?;
                Ok(self.s.m.int_const(i64t, i64::from(x as i32)))
            }
            _ => {
                // Literals and the null register: zero-extended.
                let x = self.operand(inst, &op)?;
                Ok(self.s.m.cast(c, Cast::ZExt, x, i64t))
            }
        }
    }

    /// IEEE source bits, with sign modifiers applied as bit operations so
    /// that a signalling NaN, a denormal or a negative zero is preserved.
    fn fp_bits(&mut self, inst: &Inst, k: usize, bits: u32) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else { unreachable!() };
        let mut x = match bits {
            16 => self.half_src(inst, k)?,
            32 => match (self.src0, k) {
                (Some(x), 0) => x,
                _ => {
                    let op = v.src.get(k).ok_or("a missing operand")?;
                    let x = self.operand(inst, op)?;
                    match Self::sdwa_src(inst, k) {
                        Some((sel, sext)) => self.select_bits(x, sel, sext),
                        None => x,
                    }
                }
            },
            _ => self.raw_src64(inst, k, true)?,
        };
        let ty = self.s.m.int(bits.max(32));
        let sign = 1u64 << (bits - 1);
        if v.abs >> k & 1 != 0 {
            let mask = self.s.m.int_const(ty, (sign - 1) as i64);
            x = self.s.m.bin(self.cur, Bin::And, x, mask);
        }
        if v.neg >> k & 1 != 0 {
            let sign = self.s.m.int_const(ty, sign as i64);
            x = self.s.m.bin(self.cur, Bin::Xor, x, sign);
        }
        Ok(x)
    }

    /// Decompose the IEEE encoding directly. AMD's RDNA 2 ISA specifies
    /// zero exponent for infinities and NaNs and passes their mantissa
    /// through. LLVM's AMDGPU frexp fold also normalizes subnormal inputs.
    fn frexp(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else { unreachable!() };
        if v.clamp || v.omod != 0 {
            return Err("floating output modifiers on frexp");
        }
        let name = inst.name();
        let (bits, fraction, bias) = if name.ends_with("f64") { (64, 52, 1023) }
            else if name.ends_with("f32") { (32, 23, 127) } else { (16, 10, 15) };
        let c = self.cur;
        let ty = self.s.m.int(bits.max(32));
        let x = self.fp_bits(inst, 0, bits)?;
        let k = |l: &mut Self, n| l.s.m.int_const(ty, n);
        let (z, frac, shift, emax, abs) = (k(self, 0), k(self, (1 << fraction) - 1),
            k(self, fraction), k(self, 2 * bias + 1), k(self, ((1u64 << (bits - 1)) - 1) as i64));
        let mant = self.s.m.bin(c, Bin::And, x, frac);
        let exp = self.s.m.bin(c, Bin::LShr, x, shift);
        let exp = self.s.m.bin(c, Bin::And, exp, emax);
        let subnormal = self.s.m.cmp(c, pred::EQ, exp, z);
        let abs = self.s.m.bin(c, Bin::And, x, abs);
        let nonzero = self.s.m.cmp(c, pred::NE, abs, z);
        let finite = self.s.m.cmp(c, pred::NE, exp, emax);
        let compute = self.s.m.bin(c, Bin::And, nonzero, finite);
        // Locate the highest set fraction bit with DXIL's i32 first-bit-high.
        let fb = |l: &mut Self, a| l.s.op(c, "unaryBits.i32", Attr::ReadNone, l.s.t.i32, 33, &[a]);
        let leading = if bits == 64 {
            let (lo, hi) = self.split64(mant);
            let (a, b) = (fb(self, lo), fb(self, hi));
            let (size, zero) = (self.s.m.i32(32), self.s.m.i32(0));
            let a = self.s.m.bin(c, Bin::Add, a, size);
            let low = self.s.m.cmp(c, pred::EQ, hi, zero);
            self.s.m.select(c, low, a, b)
        } else {
            fb(self, mant)
        };
        // A zero mantissa can produce -1. Mask the normalization shift
        // before emitting even the unselected path, so it is always defined.
        let offset = self.s.m.i32(bits.max(32) - 1 - fraction as u32);
        let left = self.s.m.bin(c, Bin::Sub, leading, offset);
        let limit = self.s.m.i32(bits.max(32) - 1);
        let left = self.s.m.bin(c, Bin::And, left, limit);
        let left = if bits == 64 { self.extend64(left, false) } else { left };
        let Some(Operand::Vector { reg, .. }) = v.dst else { return Err("a frexp result") };
        if name.starts_with("v_frexp_exp_") {
            let one = k(self, 1);
            let e_sub = self.s.m.bin(c, Bin::Sub, one, left);
            let e = self.s.m.select(c, subnormal, e_sub, exp);
            let bias = k(self, bias - 1);
            let e = self.s.m.bin(c, Bin::Sub, e, bias);
            let e = self.s.m.select(c, compute, e, z);
            if bits == 16 {
                self.write_half(inst, e, guard)
            } else {
                let e = if bits == 64 { self.s.m.cast(c, Cast::Trunc, e, self.s.t.i32) } else { e };
                self.write_i(inst, e, guard)
            }
        } else {
            let norm = self.s.m.bin(c, Bin::Shl, mant, left);
            let mant = self.s.m.select(c, subnormal, norm, mant);
            let mant = self.s.m.bin(c, Bin::And, mant, frac);
            let sign = k(self, (1u64 << (bits - 1)) as i64);
            let sign = self.s.m.bin(c, Bin::And, x, sign);
            let half = k(self, (bias - 1) << fraction);
            let value = self.s.m.bin(c, Bin::Or, sign, half);
            let value = self.s.m.bin(c, Bin::Or, value, mant);
            let value = self.s.m.select(c, compute, value, x);
            match bits {
                16 => self.write_half(inst, value, guard),
                32 => self.write_i(inst, value, guard),
                _ => {
                    let (lo, hi) = self.split64(value);
                    self.write(inst, Reg::V(reg), lo, guard);
                    self.write(inst, Reg::V(reg + 1), hi, guard);
                    Ok(())
                }
            }
        }
    }

    fn join64(&mut self, lo: V, hi: V) -> V {
        let c = self.cur;
        let ty = self.s.m.int(64);
        let lo = self.s.m.cast(c, Cast::ZExt, lo, ty);
        let hi = self.s.m.cast(c, Cast::ZExt, hi, ty);
        let k32 = self.s.m.int_const(ty, 32);
        let hi = self.s.m.bin(c, Bin::Shl, hi, k32);
        self.s.m.bin(c, Bin::Or, lo, hi)
    }

    fn extend64(&mut self, x: V, signed: bool) -> V {
        self.s.flags |= flags::INT64;
        let ty = self.s.m.int(64);
        self.s.m.cast(self.cur, if signed { Cast::SExt } else { Cast::ZExt }, x, ty)
    }

    /// The low and high dwords of an i64.
    fn split64(&mut self, x: V) -> (V, V) {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let k32 = self.s.m.int_const(i64t, 32);
        let lo = self.s.m.cast(c, Cast::Trunc, x, self.s.t.i32);
        let hi = self.s.m.bin(c, Bin::LShr, x, k32);
        let hi = self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32);
        (lo, hi)
    }

    /// Source `k` of a 16-bit operation in the low 16 bits (the rest 0):
    /// op_sel (or an SDWA word select) picks the high half; inline float
    /// constants are f16 values here.
    fn half_src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let mut high = v.opsel >> k & 1 != 0;
        if let Some(sd) = v.sdwa {
            match sd.src_sel.get(k).copied().unwrap_or(6) {
                4 => {}
                5 => high = true,
                6 => {}
                _ => return Err("a byte select on a 16-bit operation"),
            }
        }
        let x = match (op, self.src0) {
            (_, Some(x)) if k == 0 => x,
            (Operand::Const(c @ 240..=248), _) => self.s.m.i32(inline_f16(c)),
            _ => self.operand(inst, &op)?,
        };
        let c = self.cur;
        let x = if high {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// An f16 source as an f32 (exact), with its abs/neg modifiers.
    fn half_fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let h = self.half_src(inst, k)?;
        let f32t = self.s.t.f32;
        let mut f = self
            .s
            .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    /// Writes a 16-bit result (low bits of `x`) into the destination's low
    /// half, or its high half with op_sel bit 3 (or an SDWA WORD_1 select);
    /// the other half is kept (GFX10).
    fn write_half(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a 16-bit result outside a VGPR");
        };
        let mut high = v.opsel >> 3 & 1 != 0;
        // SDWA: the word written, the other one kept (preserve) or zero
        // (pad; a sign extension above the high word adds nothing).
        let mut keep = true;
        if let Some(sd) = v.sdwa {
            match (sd.dst_sel, sd.dst_unused) {
                (4, 2) => {}
                (5, 2) => high = true,
                (4 | 6, _) => keep = false,
                (5, _) => (high, keep) = (true, false),
                _ => return Err("an SDWA destination of a 16-bit operation"),
            }
        }
        let c = self.cur;
        let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
            Some(Ok(o)) if keep => o,
            _ => self.s.m.i32(0),
        };
        let (k16, low) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
        let x = self.s.m.bin(c, Bin::And, x, low);
        let r = if high {
            let o = self.s.m.bin(c, Bin::And, old, low);
            let x = self.s.m.bin(c, Bin::Shl, x, k16);
            self.s.m.bin(c, Bin::Or, o, x)
        } else {
            let keep = self.s.m.i32(0xFFFF_0000);
            let o = self.s.m.bin(c, Bin::And, old, keep);
            self.s.m.bin(c, Bin::Or, o, x)
        };
        self.write_placed(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    /// 16-bit operations, computed in f32 or i32: f16 sums, differences and
    /// products are exact there and round once to f16 (f32 has the 2 x 11 +
    /// 2 bits that make double rounding harmless); `None` if `name` is none.
    fn valu16(&mut self, inst: &Inst, name: &str, guard: Option<V>) -> Option<Result<(), Why>> {
        const FLOAT2: &[&str] = &[
            "v_add_f16",
            "v_sub_f16",
            "v_subrev_f16",
            "v_mul_f16",
            "v_max_f16",
            "v_min_f16",
        ];
        const FLOAT3: &[&str] = &[
            "v_fma_f16",
            "v_mad_f16",
            "v_fmac_f16",
            "v_mac_f16",
            "v_max3_f16",
            "v_min3_f16",
            "v_med3_f16",
        ];
        const FLOAT1: &[&str] = &[
            "v_rcp_f16",
            "v_sqrt_f16",
            "v_rsq_f16",
            "v_log_f16",
            "v_exp_f16",
            "v_fract_f16",
            "v_trunc_f16",
            "v_floor_f16",
            "v_ceil_f16",
            "v_rndne_f16",
            "v_sin_f16",
            "v_cos_f16",
            "v_cvt_f16_u16",
            "v_cvt_f16_i16",
        ];
        const INT: &[&str] = &[
            "v_add_nc_u16",
            "v_mad_i16",
            "v_mad_u16",
            "v_min3_i16",
            "v_min3_u16",
            "v_max3_i16",
            "v_max3_u16",
            "v_med3_i16",
            "v_med3_u16",
            "v_sub_nc_u16",
            "v_add_nc_i16",
            "v_sub_nc_i16",
            "v_subrev_nc_u16",
            "v_mul_lo_u16",
            "v_lshlrev_b16",
            "v_lshrrev_b16",
            "v_ashrrev_i16",
            "v_max_u16",
            "v_max_i16",
            "v_min_u16",
            "v_min_i16",
            "v_cvt_u16_f16",
            "v_cvt_i16_f16",
            "v_cvt_norm_i16_f16",
            "v_cvt_norm_u16_f16",
        ];
        let all = [FLOAT1, FLOAT2, FLOAT3, INT];
        if !all.iter().any(|l| l.contains(&name))
            && !matches!(name, "v_cvt_f16_f32" | "v_cvt_f32_f16")
        {
            return None;
        }
        Some(self.valu16_op(inst, name, guard, FLOAT1, FLOAT2, FLOAT3))
    }

    #[allow(clippy::too_many_arguments)]
    fn valu16_op(
        &mut self,
        inst: &Inst,
        name: &str,
        guard: Option<V>,
        float1: &[&str],
        float2: &[&str],
        float3: &[&str],
    ) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let i32t = self.s.t.i32;
        let f32t = self.s.t.f32;
        if name == "v_cvt_f32_f16" {
            let f = self.half_fsrc(inst, 0)?;
            return self.write_f(inst, f, guard);
        }
        let float_result = |l: &mut Self, mut f: V| -> Result<(), Why> {
            if v.omod != 0 {
                return Err("an output modifier on a 16-bit operation");
            }
            if v.clamp {
                f = l.unary(7, f);
            }
            let h =
                l.s.op(l.cur, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
            l.write_half(inst, h, guard)
        };
        if name == "v_cvt_f16_f32" {
            let f = self.fsrc(inst, 0)?;
            return float_result(self, f);
        }
        if float2.contains(&name) {
            let (a, b) = (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?);
            let r = match name {
                "v_add_f16" => self.s.m.bin(c, Bin::Add, a, b),
                "v_sub_f16" => self.s.m.bin(c, Bin::Sub, a, b),
                "v_subrev_f16" => self.s.m.bin(c, Bin::Sub, b, a),
                "v_mul_f16" => self.s.m.bin(c, Bin::Mul, a, b),
                "v_max_f16" => self.max_min(true, a, b),
                _ => self.max_min(false, a, b),
            };
            return float_result(self, r);
        }
        if float3.contains(&name) {
            let (a, b) = (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?);
            let acc = if matches!(name, "v_fmac_f16" | "v_mac_f16") {
                // The accumulator is the destination's half that is written.
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("an accumulator");
                };
                let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("an accumulator")?)?;
                let x = if v.opsel >> 3 & 1 != 0 {
                    let k16 = self.s.m.i32(16);
                    self.s.m.bin(c, Bin::LShr, x, k16)
                } else {
                    x
                };
                let m = self.s.m.i32(0xFFFF);
                let x = self.s.m.bin(c, Bin::And, x, m);
                self.s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
            } else {
                self.half_fsrc(inst, 2)?
            };
            let r = match name {
                // shortcut: fused f16 multiply-adds as FMad in f32 (the
                // product is exact; the sum rounds twice).
                "v_fma_f16" | "v_fmac_f16" | "v_mad_f16" | "v_mac_f16" => self.fmad(a, b, acc),
                "v_max3_f16" => {
                    let t = self.max_min(true, a, b);
                    self.max_min(true, t, acc)
                }
                "v_min3_f16" => {
                    let t = self.max_min(false, a, b);
                    self.max_min(false, t, acc)
                }
                _ => self.med3(a, b, acc),
            };
            return float_result(self, r);
        }
        if float1.contains(&name) {
            let r = match name {
                "v_cvt_f16_u16" | "v_cvt_f16_i16" => {
                    let mut x = self.half_src(inst, 0)?;
                    let op = if name == "v_cvt_f16_i16" {
                        let k16 = self.s.m.i32(16);
                        let t = self.s.m.bin(c, Bin::Shl, x, k16);
                        x = self.s.m.bin(c, Bin::AShr, t, k16);
                        Cast::SiToFp
                    } else {
                        Cast::UiToFp
                    };
                    self.s.m.cast(c, op, x, f32t)
                }
                "v_rcp_f16" => {
                    let a = self.half_fsrc(inst, 0)?;
                    let one = self.s.m.f32(1.0f32.to_bits());
                    self.s.m.bin(c, Bin::SDiv, one, a)
                }
                _ => {
                    let a = self.half_fsrc(inst, 0)?;
                    let op = match name {
                        "v_sqrt_f16" => 24,
                        "v_rsq_f16" => 25,
                        "v_log_f16" => 23,
                        "v_exp_f16" => 21,
                        "v_fract_f16" => 22,
                        "v_trunc_f16" => 29,
                        "v_floor_f16" => 27,
                        "v_ceil_f16" => 28,
                        "v_rndne_f16" => 26,
                        "v_sin_f16" => 13,
                        _ => 12,
                    };
                    match op {
                        12 | 13 => self.trig_turns(a, op == 12),
                        // Any f16 in (-2^-11, 0) has fract >= 1 - 2^-12; clamp to 0x3bff (1 - 2^-11).
                        22 => self.fract(a, 1.0 - 1.0 / 4096.0, 1.0 - 1.0 / 2048.0),
                        _ => self.unary(op, a),
                    }
                }
            };
            return float_result(self, r);
        }
        if matches!(name, "v_cvt_norm_i16_f16" | "v_cvt_norm_u16_f16") {
            if v.clamp || v.omod != 0 {
                return Err("modifiers on a 16-bit integer operation");
            }
            let r = self.cvt_norm16(inst, name == "v_cvt_norm_i16_f16")?;
            return self.write_half(inst, r, guard);
        }
        // 16-bit integers: computed on the low halves, the low 16 bits kept.
        let saturates = matches!(name, "v_mad_i16" | "v_mad_u16");
        if v.clamp && !saturates || v.omod != 0 || v.abs != 0 || v.neg != 0 {
            return Err("modifiers on a 16-bit integer operation");
        }
        let r = match name {
            "v_cvt_u16_f16" | "v_cvt_i16_f16" => {
                // Toward zero, saturating at the 16-bit range (infinities
                // included); NaN gives 0. Clamped in f32 first, where every
                // f16 is exact, so the conversion never leaves i32's range.
                let a = self.half_fsrc(inst, 0)?;
                let (lo, hi) = if name == "v_cvt_u16_f16" {
                    (0.0f32, 65535.0f32)
                } else {
                    (-32768.0, 32767.0)
                };
                let (lo, hi) = (self.s.m.f32(lo.to_bits()), self.s.m.f32(hi.to_bits()));
                let t = self.binary(35, a, lo);
                let t = self.binary(36, t, hi);
                let r = self.s.m.cast(c, Cast::FpToSi, t, i32t);
                let nan = self.s.m.cmp(c, pred::UNO, a, a);
                let z = self.s.m.i32(0);
                self.s.m.select(c, nan, z, r)
            }
            _ => {
                let signed = name.ends_with("_i16");
                let read = |l: &mut Self, i| -> Result<V, Why> {
                    let x = l.half_src(inst, i)?;
                    Ok(if signed { l.sign_extend(x, 16) } else { x })
                };
                let (a, b) = (read(self, 0)?, read(self, 1)?);
                let k15 = self.s.m.i32(15);
                match name {
                    "v_add_nc_u16" | "v_add_nc_i16" => self.s.m.bin(c, Bin::Add, a, b),
                    "v_sub_nc_u16" | "v_sub_nc_i16" => self.s.m.bin(c, Bin::Sub, a, b),
                    "v_subrev_nc_u16" => self.s.m.bin(c, Bin::Sub, b, a),
                    "v_mul_lo_u16" => self.s.m.bin(c, Bin::Mul, a, b),
                    "v_mad_i16" | "v_mad_u16" => {
                        let d = read(self, 2)?;
                        let m = self.s.m.bin(c, Bin::Mul, a, b);
                        self.s.m.bin(c, Bin::Add, m, d)
                    }
                    "v_min3_i16" | "v_min3_u16" | "v_max3_i16" | "v_max3_u16"
                    | "v_med3_i16" | "v_med3_u16" => {
                        let d = read(self, 2)?;
                        self.int3(&name[2..6], signed, a, b, d)
                    }
                    "v_lshlrev_b16" | "v_lshrrev_b16" | "v_ashrrev_i16" => {
                        // src0 is the shift (4 bits), src1 the value.
                        let sh = self.s.m.bin(c, Bin::And, a, k15);
                        let op = match name {
                            "v_lshlrev_b16" => Bin::Shl,
                            "v_lshrrev_b16" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        self.s.m.bin(c, op, b, sh)
                    }
                    _ => {
                        let p = match name {
                            "v_max_u16" => pred::IUGT,
                            "v_min_u16" => pred::IULT,
                            "v_max_i16" => pred::SGT,
                            _ => pred::SLT,
                        };
                        let t = self.s.m.cmp(c, p, a, b);
                        self.s.m.select(c, t, a, b)
                    }
                }
            }
        };
        let r = if v.clamp {
            let signed = name.ends_with("_i16");
            let (lo, hi) = if signed { (-32768, 32767) } else { (0, 65535) };
            self.clamp_int(r, 32, lo, hi, signed)
        } else {
            r
        };
        self.write_half(inst, r, guard)
    }

    /// `v_cvt_norm_i16_f16` / `v_cvt_norm_u16_f16`: the f16 source (with its
    /// abs/neg modifiers) clamped to [-1, 1] or [0, 1], times 32767 or 65535,
    /// rounded to nearest even once from the exact product; NaN gives 0.
    /// Worked on the f16's bits in integers, because in f32 the product
    /// (up to 26 significant bits) would round before the rounding to an
    /// integer.
    fn cvt_norm16(&mut self, inst: &Inst, signed: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let mut h = self.half_src(inst, 0)?;
        if v.abs & 1 != 0 {
            let m = k(self, 0x7FFF);
            h = self.s.m.bin(c, Bin::And, h, m);
        }
        if v.neg & 1 != 0 {
            let m = k(self, 0x8000);
            h = self.s.m.bin(c, Bin::Xor, h, m);
        }
        let scale = if signed { 0x7FFF } else { 0xFFFF };
        let (k0, k1, k10) = (k(self, 0), k(self, 1), k(self, 10));
        let mag_mask = k(self, 0x7FFF);
        let mag = self.s.m.bin(c, Bin::And, h, mag_mask);
        let k15 = k(self, 15);
        let neg = self.s.m.bin(c, Bin::LShr, h, k15);
        let neg = self.s.m.cmp(c, pred::NE, neg, k0);
        // |x| < 1: x = m * 2^-s with m < 2^11 and 11 <= s <= 24.
        let exp = self.s.m.bin(c, Bin::LShr, mag, k10);
        let man_mask = k(self, 0x3FF);
        let man = self.s.m.bin(c, Bin::And, mag, man_mask);
        let implicit = k(self, 0x400);
        let normal = self.s.m.bin(c, Bin::Or, man, implicit);
        let sub = self.s.m.cmp(c, pred::EQ, exp, k0);
        let m = self.s.m.select(c, sub, man, normal);
        let k25 = k(self, 25);
        let s_norm = self.s.m.bin(c, Bin::Sub, k25, exp);
        let k24 = k(self, 24);
        let sh = self.s.m.select(c, sub, k24, s_norm);
        let ks = k(self, scale);
        let p = self.s.m.bin(c, Bin::Mul, m, ks);
        let q = self.s.m.bin(c, Bin::LShr, p, sh);
        let one_s = self.s.m.bin(c, Bin::Shl, k1, sh);
        let low_mask = self.s.m.bin(c, Bin::Sub, one_s, k1);
        let rem = self.s.m.bin(c, Bin::And, p, low_mask);
        let half = self.s.m.bin(c, Bin::LShr, one_s, k1);
        let above = self.s.m.cmp(c, pred::IUGT, rem, half);
        let tie = self.s.m.cmp(c, pred::EQ, rem, half);
        let odd = self.s.m.bin(c, Bin::And, q, k1);
        let odd = self.s.m.cmp(c, pred::NE, odd, k0);
        let tie_up = self.s.m.bin(c, Bin::And, tie, odd);
        let up = self.s.m.bin(c, Bin::Or, above, tie_up);
        let r = self.s.m.select(c, up, k1, k0);
        let r = self.s.m.bin(c, Bin::Add, q, r);
        // |x| >= 1 (and infinity) saturates; NaN gives 0.
        let one_h = k(self, 0x3C00);
        let big = self.s.m.cmp(c, pred::IUGE, mag, one_h);
        let r = self.s.m.select(c, big, ks, r);
        let inf = k(self, 0x7C00);
        let nan = self.s.m.cmp(c, pred::IUGT, mag, inf);
        let r = self.s.m.select(c, nan, k0, r);
        let r = if signed {
            let n = self.s.m.bin(c, Bin::Sub, k0, r);
            self.s.m.select(c, neg, n, r)
        } else {
            self.s.m.select(c, neg, k0, r)
        };
        let m16 = k(self, 0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, r, m16))
    }

    /// Half `high` of source `k` of a packed (VOP3P) operation, in the low
    /// 16 bits: op_sel picks the half the low result reads, op_sel_hi the
    /// half the high result reads. Packed integer instructions see the f32
    /// bits of an inline float constant.
    fn packed_src(&mut self, inst: &Inst, k: usize, high: bool, float: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let sel = if high { v.opsel_hi } else { v.opsel } >> k & 1 != 0;
        let x = if float {
            self.f16_operand(inst, &op)?
        } else {
            self.operand(inst, &op)?
        };
        let c = self.cur;
        let x = if sel {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// Packed 16-bit operations (`v_pk_*`): each half computed as the
    /// 16-bit operation does, f16 through f32 as in [`Self::valu16`].
    /// `v_pk_fmac_f16`: each half of the destination plus the product of the
    /// sources' halves.
    // shortcut: FMad in f32, as v_fma_f16 (a second rounding).
    fn packed_fmac(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a packed half FMA's destination");
        };
        if v.src.iter().any(|o| matches!(o, Operand::Const(_) | Operand::Literal(_)))
            || v.sdwa.is_some()
            || v.dpp.is_some()
        {
            return Err("a packed half FMA of constants");
        }
        let c = self.cur;
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let old = inst.read(Reg::V(reg)).ok_or("an unread destination")?;
        let words = [self.operand(inst, &v.src[0])?, self.operand(inst, &v.src[1])?, self.i32_of(old)?];
        let (k16, low) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
        let mut halves = Vec::new();
        for high in [false, true] {
            let xs: Vec<V> = words
                .iter()
                .map(|&w| {
                    let w = if high { self.s.m.bin(c, Bin::LShr, w, k16) } else { w };
                    let w = self.s.m.bin(c, Bin::And, w, low);
                    self.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[w])
                })
                .collect();
            let f = self.fmad(xs[0], xs[1], xs[2]);
            halves.push(self.s.op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]));
        }
        let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
        let lo = self.s.m.bin(c, Bin::And, halves[0], low);
        let x = self.s.m.bin(c, Bin::Or, lo, hi);
        self.write_i(inst, x, guard)
    }

    fn packed16(&mut self, inst: &Inst, op: &str, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let float = op.ends_with("_f16");
        let n = match op {
            "fma_f16" | "mad_u16" | "mad_i16" => 3,
            "add_f16" | "mul_f16" | "min_f16" | "max_f16" | "add_u16" | "sub_u16" | "add_i16"
            | "sub_i16" | "mul_lo_u16" | "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16"
            | "max_u16" | "min_u16" | "max_i16" | "min_i16" => 2,
            _ => return Err("a packed 16-bit operation"),
        };
        if v.omod != 0 || !float && (v.clamp || v.neg != 0 || v.neg_hi != 0) {
            return Err("modifiers on a packed integer operation");
        }
        let c = self.cur;
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let mut halves = Vec::new();
        for high in [false, true] {
            let mut xs = Vec::new();
            for k in 0..n {
                let h = self.packed_src(inst, k, high, float)?;
                if float {
                    let mut f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    let neg = if high { v.neg_hi } else { v.neg };
                    if neg >> k & 1 != 0 {
                        let z = self.s.m.f32(0x8000_0000);
                        f = self.s.m.bin(c, Bin::Sub, z, f);
                    }
                    xs.push(f);
                } else {
                    xs.push(h);
                }
            }
            let r = if float {
                let mut f = match op {
                    "add_f16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "mul_f16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "min_f16" => self.max_min(false, xs[0], xs[1]),
                    "max_f16" => self.max_min(true, xs[0], xs[1]),
                    // shortcut: FMad in f32, as v_fma_f16.
                    _ => self.fmad(xs[0], xs[1], xs[2]),
                };
                if v.clamp {
                    f = self.unary(7, f);
                }
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
            } else {
                let k16 = self.s.m.i32(16);
                let sext = |l: &mut Self, x: V| {
                    let t = l.s.m.bin(c, Bin::Shl, x, k16);
                    l.s.m.bin(c, Bin::AShr, t, k16)
                };
                if op.ends_with("_i16") {
                    for x in &mut xs {
                        *x = sext(self, *x);
                    }
                }
                let k15 = self.s.m.i32(15);
                match op {
                    "add_u16" | "add_i16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "sub_u16" | "sub_i16" => self.s.m.bin(c, Bin::Sub, xs[0], xs[1]),
                    "mul_lo_u16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "mad_u16" | "mad_i16" => {
                        let m = self.s.m.bin(c, Bin::Mul, xs[0], xs[1]);
                        self.s.m.bin(c, Bin::Add, m, xs[2])
                    }
                    "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16" => {
                        let sh = self.s.m.bin(c, Bin::And, xs[0], k15);
                        let o = match op {
                            "lshlrev_b16" => Bin::Shl,
                            "lshrrev_b16" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        let x = if op == "ashrrev_i16" {
                            sext(self, xs[1])
                        } else {
                            xs[1]
                        };
                        self.s.m.bin(c, o, x, sh)
                    }
                    _ => {
                        let p = match op {
                            "max_u16" => pred::IUGT,
                            "min_u16" => pred::IULT,
                            "max_i16" => pred::SGT,
                            _ => pred::SLT,
                        };
                        let t = self.s.m.cmp(c, p, xs[0], xs[1]);
                        self.s.m.select(c, t, xs[0], xs[1])
                    }
                }
            };
            let m = self.s.m.i32(0xFFFF);
            halves.push(self.s.m.bin(c, Bin::And, r, m));
        }
        let k16 = self.s.m.i32(16);
        let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
        let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a packed result outside a VGPR");
        };
        self.write(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    /// The sign or carry bit above the 64-bit multiply-add is returned in
    /// SDST. LLVM's AMDGPURegisterBankInfo describes the signed form as the
    /// low bit of sign(product) + sign(accumulator) + unsigned carry-out.
    fn wide_mad(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else { unreachable!() };
        if v.clamp {
            return Err("clamped 64-bit multiply-add carry semantics");
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else { return Err("a wide multiply-add result") };
        let signed = inst.name() == "v_mad_i64_i32";
        let (a, b) = (self.src(inst, 0)?, self.src(inst, 1)?);
        let (a, b) = (self.extend64(a, signed), self.extend64(b, signed));
        let acc = self.src64(inst, 2)?;
        let c = self.cur;
        let prod = self.s.m.bin(c, Bin::Mul, a, b);
        let sum = self.s.m.bin(c, Bin::Add, prod, acc);
        let mut carry = self.s.m.cmp(c, pred::IULT, sum, prod);
        if signed {
            let t = self.s.m.int(64);
            let z = self.s.m.int_const(t, 0);
            let pneg = self.s.m.cmp(c, pred::SLT, prod, z);
            let aneg = self.s.m.cmp(c, pred::SLT, acc, z);
            let signs = self.s.m.bin(c, Bin::Xor, pneg, aneg);
            carry = self.s.m.bin(c, Bin::Xor, carry, signs);
        }
        if let Some(Operand::Scalar { code, .. }) = v.carry
            && let Some(w) = inst.write(Reg::S(code))
        {
            self.lane_bits.insert(w, carry);
        }
        let (lo, hi) = self.split64(sum);
        self.write(inst, Reg::V(reg), lo, guard);
        self.write(inst, Reg::V(reg + 1), hi, guard);
        for &(_, w) in &inst.writes {
            self.vals
                .entry(w)
                .or_insert(Val::Missing("a lane mask read as data"));
        }
        Ok(())
    }

    fn int_field(&mut self, x: V, shift: u32, width: u32) -> V {
        if width == 32 {
            return x;
        }
        let sh = self.s.m.i32(shift);
        let mask = self.s.m.i32((1 << width) - 1);
        let x = self.s.m.bin(self.cur, Bin::LShr, x, sh);
        self.s.m.bin(self.cur, Bin::And, x, mask)
    }

    fn int3(&mut self, op: &str, signed: bool, a: V, b: V, d: V) -> V {
        let (lt, gt) = if signed { (pred::SLT, pred::SGT) } else { (pred::IULT, pred::IUGT) };
        match op {
            "min3" => {
                let t = self.int_pick(lt, a, b);
                self.int_pick(lt, t, d)
            }
            "max3" => {
                let t = self.int_pick(gt, a, b);
                self.int_pick(gt, t, d)
            }
            _ => {
                let lo = self.int_pick(lt, a, b);
                let hi = self.int_pick(gt, a, b);
                let t = self.int_pick(lt, hi, d);
                self.int_pick(gt, lo, t)
            }
        }
    }

    /// Integer dot products, including the optional output saturation in
    /// VOP3P (LLVM AMDGPUUsage's sdot/udot intrinsics).
    fn integer_dot(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else { unreachable!() };
        let name = inst.name();
        let implicit = name == "v_dot4c_i32_i8";
        let width = if name.starts_with("v_dot2") { 16 } else if name.starts_with("v_dot4") { 8 } else { 4 };
        let signed = name.contains("_i32");
        if v.neg != 0 || v.neg_hi != 0 || v.abs != 0 || v.omod != 0 || v.sdwa.is_some() {
            return Err("modifiers on an integer dot product");
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else { return Err("an integer dot product result") };
        let c = self.cur;
        let a = self.src(inst, 0)?;
        let b = self.src(inst, 1)?;
        let d = if implicit {
            self.i32_of(inst.read(Reg::V(reg)).ok_or("a dot product accumulator")?)?
        } else {
            self.src(inst, 2)?
        };
        let mut sum = if v.clamp { self.extend64(d, signed) } else { d };
        for i in 0..32 / width {
            // dot2 has independent low/high half selects for each input.
            let (sa, sb) = if width == 16 {
                let sel = if i == 0 { v.opsel } else { v.opsel_hi };
                (u32::from(sel & 1) * 16, u32::from(sel >> 1 & 1) * 16)
            } else {
                (i * width, i * width)
            };
            let (mut a, mut b) = (self.int_field(a, sa, width), self.int_field(b, sb, width));
            if signed {
                a = self.sign_extend(a, width);
                b = self.sign_extend(b, width);
            }
            if v.clamp {
                a = self.extend64(a, signed);
                b = self.extend64(b, signed);
            }
            let p = self.s.m.bin(c, Bin::Mul, a, b);
            sum = self.s.m.bin(c, Bin::Add, sum, p);
        }
        if v.clamp {
            let (min, max) = if signed { (i32::MIN as i64, i32::MAX as i64) } else { (0, u32::MAX as i64) };
            sum = self.clamp_int(sum, 64, min, max, signed);
            sum = self.s.m.cast(c, Cast::Trunc, sum, self.s.t.i32);
        }
        self.write(inst, Reg::V(reg), sum, guard);
        Ok(())
    }

    /// Sum of the unsigned absolute differences of the packed elements.
    fn sad(&mut self, a: V, b: V, width: u32, masked: bool) -> V {
        let c = self.cur;
        let zero = self.s.m.i32(0);
        let mut sum = zero;
        for offset in (0..32).step_by(width as usize) {
            let x = self.int_field(a, offset, width);
            let y = self.int_field(b, offset, width);
            let ge = self.s.m.cmp(c, pred::IUGE, x, y);
            let xy = self.s.m.bin(c, Bin::Sub, x, y);
            let yx = self.s.m.bin(c, Bin::Sub, y, x);
            let mut diff = self.s.m.select(c, ge, xy, yx);
            if masked {
                let skip = self.s.m.cmp(c, pred::EQ, y, zero);
                diff = self.s.m.select(c, skip, zero, diff);
            }
            sum = self.s.m.bin(c, Bin::Add, sum, diff);
        }
        sum
    }

    fn integer(&mut self, inst: &Inst, name: &'static str) -> Result<V, Why> {
        let c = self.cur;
        let s = |l: &mut Self, k| l.src(inst, k);
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let m31 = |l: &mut Self, x: V| {
            let k31 = l.s.m.i32(31);
            l.s.m.bin(c, Bin::And, x, k31)
        };
        Ok(match name {
            "v_and_b32" | "v_or_b32" | "v_xor_b32" | "v_add_nc_u32" | "v_add_nc_i32"
            | "v_sub_nc_u32" | "v_sub_nc_i32" | "v_subrev_nc_u32" | "v_mul_lo_u32"
            | "v_mul_lo_i32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let (op, a, b) = match name {
                    "v_and_b32" => (Bin::And, a, b),
                    "v_or_b32" => (Bin::Or, a, b),
                    "v_xor_b32" => (Bin::Xor, a, b),
                    "v_add_nc_u32" | "v_add_nc_i32" => (Bin::Add, a, b),
                    "v_sub_nc_u32" | "v_sub_nc_i32" => (Bin::Sub, a, b),
                    "v_subrev_nc_u32" => (Bin::Sub, b, a),
                    _ => (Bin::Mul, a, b),
                };
                self.s.m.bin(c, op, a, b)
            }
            "v_xnor_b32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let x = self.s.m.bin(c, Bin::Xor, a, b);
                let ones = k(self, u32::MAX);
                self.s.m.bin(c, Bin::Xor, x, ones)
            }
            "v_not_b32" => {
                let a = s(self, 0)?;
                let ones = k(self, u32::MAX);
                self.s.m.bin(c, Bin::Xor, a, ones)
            }
            "v_lshlrev_b32" | "v_lshrrev_b32" | "v_ashrrev_i32" => {
                let (sh, a) = (s(self, 0)?, s(self, 1)?);
                let sh = m31(self, sh);
                let op = match name {
                    "v_lshlrev_b32" => Bin::Shl,
                    "v_lshrrev_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                self.s.m.bin(c, op, a, sh)
            }
            "v_lshl_add_u32" | "v_lshl_or_b32" | "v_add_lshl_u32" | "v_and_or_b32"
            | "v_or3_b32" | "v_xad_u32" | "v_add3_u32" | "v_xor3_b32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                match name {
                    "v_lshl_add_u32" | "v_lshl_or_b32" => {
                        let b = m31(self, b);
                        let t = self.s.m.bin(c, Bin::Shl, a, b);
                        let op = if name == "v_lshl_add_u32" {
                            Bin::Add
                        } else {
                            Bin::Or
                        };
                        self.s.m.bin(c, op, t, d)
                    }
                    "v_add_lshl_u32" => {
                        let t = self.s.m.bin(c, Bin::Add, a, b);
                        let d = m31(self, d);
                        self.s.m.bin(c, Bin::Shl, t, d)
                    }
                    _ => {
                        let (o1, o2) = match name {
                            "v_and_or_b32" => (Bin::And, Bin::Or),
                            "v_or3_b32" => (Bin::Or, Bin::Or),
                            "v_xad_u32" => (Bin::Xor, Bin::Add),
                            "v_add3_u32" => (Bin::Add, Bin::Add),
                            _ => (Bin::Xor, Bin::Xor),
                        };
                        let t = self.s.m.bin(c, o1, a, b);
                        self.s.m.bin(c, o2, t, d)
                    }
                }
            }
            "v_mul_u32_u24" | "v_mul_i32_i24" | "v_mul_hi_i32_i24" | "v_mul_hi_u32_u24" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let (sh, op) = (
                    k(self, 8),
                    if name.ends_with("i24") {
                        Bin::AShr
                    } else {
                        Bin::LShr
                    },
                );
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                if name.starts_with("v_mul_hi") {
                    self.mul_hi(a, b, name.ends_with("i24"))
                } else {
                    self.s.m.bin(c, Bin::Mul, a, b)
                }
            }
            "v_sad_u8" | "v_sad_hi_u8" | "v_sad_u16" | "v_sad_u32" | "v_msad_u8" => {
                // RDNA 2 ISA, VOP3A: accumulation wraps in the dword.
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let width = if name.ends_with("u8") { 8 } else if name.ends_with("u16") { 16 } else { 32 };
                let sum = self.sad(a, b, width, name == "v_msad_u8");
                let sum = if name == "v_sad_hi_u8" {
                    let k16 = k(self, 16);
                    self.s.m.bin(c, Bin::Shl, sum, k16)
                } else {
                    sum
                };
                self.s.m.bin(c, Bin::Add, sum, d)
            }
            "v_lerp_u8" => {
                // RDNA 2 ISA, VOP3A: low bit of each byte of S2 rounds a tie up.
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let one = k(self, 1);
                let mut result = k(self, 0);
                for shift in [0, 8, 16, 24] {
                    let x = self.int_field(a, shift, 8);
                    let y = self.int_field(b, shift, 8);
                    let round = self.int_field(d, shift, 1);
                    let sum = self.s.m.bin(c, Bin::Add, x, y);
                    let sum = self.s.m.bin(c, Bin::Add, sum, round);
                    let avg = self.s.m.bin(c, Bin::LShr, sum, one);
                    let sh = k(self, shift);
                    let field = self.s.m.bin(c, Bin::Shl, avg, sh);
                    result = self.s.m.bin(c, Bin::Or, result, field);
                }
                result
            }
            "v_bfe_u32" | "v_bfe_i32" => {
                let (a, off, w) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let w = m31(self, w);
                self.bfe(a, off, w, name == "v_bfe_i32")
            }
            "v_bfi_b32" => {
                let (m, a, b) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let ones = k(self, u32::MAX);
                let nm = self.s.m.bin(c, Bin::Xor, m, ones);
                let x = self.s.m.bin(c, Bin::And, m, a);
                let y = self.s.m.bin(c, Bin::And, nm, b);
                self.s.m.bin(c, Bin::Or, x, y)
            }
            "v_max_i32" | "v_min_i32" | "v_max_u32" | "v_min_u32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let p = match name {
                    "v_max_i32" => pred::SGT,
                    "v_min_i32" => pred::SLT,
                    "v_max_u32" => pred::IUGT,
                    _ => pred::IULT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                self.s.m.select(c, t, a, b)
            }
            "v_mul_hi_u32" | "v_mul_hi_i32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                self.mul_hi(a, b, name == "v_mul_hi_i32")
            }
            "v_cvt_pk_u16_u32" | "v_cvt_pk_i16_i32" => {
                // Each source saturated to 16 bits: src0 into the low half,
                // src1 into the high one.
                let unsigned = name == "v_cvt_pk_u16_u32";
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let mut halves = [a, b];
                for x in &mut halves {
                    *x = if unsigned {
                        let max = k(self, 0xFFFF);
                        let over = self.s.m.cmp(c, pred::IUGT, *x, max);
                        self.s.m.select(c, over, max, *x)
                    } else {
                        let (lo, hi, m) = (k(self, 0xFFFF_8000), k(self, 0x7FFF), k(self, 0xFFFF));
                        let below = self.s.m.cmp(c, pred::SLT, *x, lo);
                        let t = self.s.m.select(c, below, lo, *x);
                        let above = self.s.m.cmp(c, pred::SGT, t, hi);
                        let t = self.s.m.select(c, above, hi, t);
                        self.s.m.bin(c, Bin::And, t, m)
                    };
                }
                let k16 = k(self, 16);
                let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
                self.s.m.bin(c, Bin::Or, halves[0], hi)
            }
            "v_dot2_i32_i16" | "v_dot2_u32_u16" | "v_dot4_i32_i8" | "v_dot4_u32_u8"
            | "v_dot8_u32_u4" => {
                // The products of the packed fields of src0 and src1, summed
                // into src2, wrapping (clamp is refused with the other
                // integer output modifiers).
                let Fields::Valu(v) = &inst.fields else {
                    unreachable!()
                };
                if v.opsel != 0 || v.neg_hi != 0 {
                    return Err("dot product operand selects");
                }
                let (bits, signed) = match name {
                    "v_dot2_i32_i16" => (16, true),
                    "v_dot2_u32_u16" => (16, false),
                    "v_dot4_i32_i8" => (8, true),
                    "v_dot4_u32_u8" => (8, false),
                    _ => (4, false),
                };
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let mut sum = s(self, 2)?;
                let op = if signed { Bin::AShr } else { Bin::LShr };
                for i in 0..32 / bits {
                    // Field i to the top, then down again (extending it).
                    let (up, down) = (k(self, 32 - bits * (i + 1)), k(self, 32 - bits));
                    let x = self.s.m.bin(c, Bin::Shl, a, up);
                    let x = self.s.m.bin(c, op, x, down);
                    let y = self.s.m.bin(c, Bin::Shl, b, up);
                    let y = self.s.m.bin(c, op, y, down);
                    let p = self.s.m.bin(c, Bin::Mul, x, y);
                    sum = self.s.m.bin(c, Bin::Add, sum, p);
                }
                sum
            }
            "v_alignbit_b32" | "v_alignbyte_b32" => {
                // The low dword of {a, b} >> shift.
                let (a, b, sh) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let sh = if name == "v_alignbyte_b32" {
                    let three = k(self, 3);
                    let t = self.s.m.bin(c, Bin::And, sh, three);
                    let k3 = k(self, 3);
                    self.s.m.bin(c, Bin::Shl, t, k3)
                } else {
                    m31(self, sh)
                };
                let k32 = k(self, 32);
                let lo = self.s.m.bin(c, Bin::LShr, b, sh);
                let left = self.s.m.bin(c, Bin::Sub, k32, sh);
                let left = m31(self, left);
                let hi = self.s.m.bin(c, Bin::Shl, a, left);
                let zero = k(self, 0);
                let none = self.s.m.cmp(c, pred::EQ, sh, zero);
                let hi = self.s.m.select(c, none, zero, hi);
                self.s.m.bin(c, Bin::Or, lo, hi)
            }
            "v_bfrev_b32" => {
                let a = s(self, 0)?;
                self.s
                    .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a])
            }
            "v_bcnt_u32_b32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let n = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 31, &[a]);
                self.s.m.bin(c, Bin::Add, n, b)
            }
            "v_mad_u32_u24" | "v_mad_i32_i24" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let (sh, op) = (
                    k(self, 8),
                    if name == "v_mad_i32_i24" {
                        Bin::AShr
                    } else {
                        Bin::LShr
                    },
                );
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                let m = self.s.m.bin(c, Bin::Mul, a, b);
                self.s.m.bin(c, Bin::Add, m, d)
            }
            "v_min3_i32" | "v_max3_i32" | "v_min3_u32" | "v_max3_u32" | "v_med3_i32"
            | "v_med3_u32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                self.int3(&name[2..6], name.ends_with("i32"), a, b, d)
            }
            "v_perm_b32" => {
                // Byte i of the result is byte S of {src0, src1} (src1 the
                // low dword) for selector byte S below 8; 8-11 repeat the
                // sign of bytes 1 and 3 of src1, then of src0; 12 is 0x00,
                // higher 0xFF.
                let Fields::Valu(v) = &inst.fields else {
                    unreachable!()
                };
                let sel = match v.src.get(2) {
                    Some(&Operand::Literal(x)) => x,
                    Some(&Operand::Const(k)) => crate::resource::inline(k).ok_or(name)?,
                    _ => {
                        // A computed selector: each byte picked at run time.
                        let (a, b, sv) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                        let mut r = k(self, 0);
                        for i in 0..4u32 {
                            let (sh, ff) = (k(self, 8 * i), k(self, 0xFF));
                            let sel = self.s.m.bin(c, Bin::LShr, sv, sh);
                            let sel = self.s.m.bin(c, Bin::And, sel, ff);
                            // Select a dword and a byte within it.
                            // Later selects handle codes 8 and above.
                            let four = k(self, 4);
                            let zero = k(self, 0);
                            let high = self.s.m.bin(c, Bin::And, sel, four);
                            let high = self.s.m.cmp(c, pred::NE, high, zero);
                            let word = self.s.m.select(c, high, a, b);
                            let three = k(self, 3);
                            let index = self.s.m.bin(c, Bin::And, sel, three);
                            let bits = self.s.m.bin(c, Bin::Shl, index, three);
                            let x = self.s.m.bin(c, Bin::LShr, word, bits);
                            let mut byte = self.s.m.bin(c, Bin::And, x, ff);
                            // 8-11: the sign of bytes 1, 3 of src1, then of src0.
                            for (n, src, bit) in [(8, b, 15), (9, b, 31), (10, a, 15), (11, a, 31)]
                            {
                                let left = k(self, 31 - bit);
                                let t = self.s.m.bin(c, Bin::Shl, src, left);
                                let k31 = k(self, 31);
                                let t = self.s.m.bin(c, Bin::AShr, t, k31);
                                let t = self.s.m.bin(c, Bin::And, t, ff);
                                let kn = k(self, n);
                                let is = self.s.m.cmp(c, pred::EQ, sel, kn);
                                byte = self.s.m.select(c, is, t, byte);
                            }
                            let (k12, z) = (k(self, 12), k(self, 0));
                            let is12 = self.s.m.cmp(c, pred::EQ, sel, k12);
                            byte = self.s.m.select(c, is12, z, byte);
                            let above = self.s.m.cmp(c, pred::IUGT, sel, k12);
                            byte = self.s.m.select(c, above, ff, byte);
                            let at = k(self, 8 * i);
                            let byte = self.s.m.bin(c, Bin::Shl, byte, at);
                            r = self.s.m.bin(c, Bin::Or, r, byte);
                        }
                        return Ok(r);
                    }
                };
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let mut r = k(self, 0);
                for i in 0..4 {
                    let pick = sel >> (8 * i) & 0xFF;
                    let byte = match pick {
                        0..=7 => {
                            let (src, n) = if pick < 4 { (b, pick) } else { (a, pick - 4) };
                            let sh = k(self, 8 * n);
                            let t = self.s.m.bin(c, Bin::LShr, src, sh);
                            let ff = k(self, 0xFF);
                            self.s.m.bin(c, Bin::And, t, ff)
                        }
                        8..=11 => {
                            let src = if pick < 10 { b } else { a };
                            let bit = if pick % 2 == 0 { 15 } else { 31 };
                            let left = k(self, 31 - bit);
                            let t = self.s.m.bin(c, Bin::Shl, src, left);
                            let k31 = k(self, 31);
                            let t = self.s.m.bin(c, Bin::AShr, t, k31);
                            let ff = k(self, 0xFF);
                            self.s.m.bin(c, Bin::And, t, ff)
                        }
                        12 => continue,
                        _ => k(self, 0xFF),
                    };
                    let at = k(self, 8 * i);
                    let byte = self.s.m.bin(c, Bin::Shl, byte, at);
                    r = self.s.m.bin(c, Bin::Or, r, byte);
                }
                r
            }
            "v_bfm_b32" => {
                // ((1 << src0[4:0]) - 1) << src1[4:0]
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let one = k(self, 1);
                let (a, b) = (m31(self, a), m31(self, b));
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                self.s.m.bin(c, Bin::Shl, m, b)
            }
            "v_cvt_pk_u8_f32" => {
                // src0 rounded to nearest even and saturated to a byte, into
                // byte src1[1:0] of src2.
                let f = self.fsrc(inst, 0)?;
                let (b, d) = (s(self, 1)?, s(self, 2)?);
                let f = self.unary(26, f);
                let (zero, max) = (self.s.m.f32(0), self.s.m.f32(255.0f32.to_bits()));
                let f = self.binary(35, f, zero);
                let f = self.binary(36, f, max);
                let x = self.s.m.cast(c, Cast::FpToUi, f, self.s.t.i32);
                let (three, k3) = (k(self, 3), k(self, 3));
                let sh = self.s.m.bin(c, Bin::And, b, three);
                let sh = self.s.m.bin(c, Bin::Shl, sh, k3);
                let x = self.s.m.bin(c, Bin::Shl, x, sh);
                let ff = k(self, 0xFF);
                let hole = self.s.m.bin(c, Bin::Shl, ff, sh);
                let ones = k(self, u32::MAX);
                let hole = self.s.m.bin(c, Bin::Xor, hole, ones);
                let d = self.s.m.bin(c, Bin::And, d, hole);
                self.s.m.bin(c, Bin::Or, d, x)
            }
            "v_ffbl_b32" => {
                // -1 for zero, as DXIL's first-bit-low.
                let a = s(self, 0)?;
                self.s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 32, &[a])
            }
            "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
                // Bits of the mask's low (high) half below this lane, plus src1:
                // with an all-ones mask, the lane index.
                let b = s(self, 1)?;
                // Mesh lanes are the console's by construction; pixel and
                // vertex lanes are the host's, as their masks are.
                if self.mask_data == MaskData::Host {
                    // Host lanes: the mask must be one (a ballot) or all lanes.
                    let Fields::Valu(v) = &inst.fields else {
                        unreachable!()
                    };
                    if name == "v_mbcnt_lo_u32_b32" && !self.x.native.wave64 {
                        if matches!(v.src.first(), Some(&Operand::Const(193))) {
                            self.s.flags |= flags::WAVE_OPS;
                            let lane = self.lane_index();
                            return Ok(self.s.m.bin(c, Bin::Add, lane, b));
                        }
                        if let Some((lo, hi)) = v
                            .src
                            .first()
                            .map(|op| self.host_mask_pair(inst, op))
                            .transpose()?
                            .flatten()
                        {
                            self.s.flags |= flags::WAVE_OPS;
                            let lane = self.lane_index();
                            let m_lo = self.below_mask(c, lane);
                            let (k32, z) = (k(self, 32), k(self, 0));
                            let t = self.s.m.bin(c, Bin::Sub, lane, k32);
                            let low = self.s.m.cmp(c, pred::IULT, lane, k32);
                            let below_hi = self.s.m.select(c, low, z, t);
                            let m_hi = self.below_mask(c, below_hi);
                            let x_lo = self.s.m.bin(c, Bin::And, lo, m_lo);
                            let x_hi = self.s.m.bin(c, Bin::And, hi, m_hi);
                            let n = self.bcnt1_64(c, x_lo, x_hi);
                            return Ok(self.s.m.bin(c, Bin::Add, n, b));
                        }
                    }
                    let host = match v.src.first() {
                        Some(&Operand::Const(193)) => true,
                        Some(&Operand::Scalar { code, .. }) => {
                            inst.read(Reg::S(code)).is_some_and(|m| {
                                self.x.st.masks.contains_key(&m) || self.x.st.highs.contains_key(&m)
                            })
                        }
                        _ => false,
                    };
                    if !host {
                        return Err("v_mbcnt of a mask that is no ballot (console lane layout)");
                    }
                    self.s.flags |= flags::WAVE_OPS;
                } else if self.s.mesh.is_none() {
                    self.cross_lane(false)?;
                }
                let a = s(self, 0)?;
                let lane = if self.console_cf {
                    self.console_lane().0
                } else {
                    self.lane_index()
                };
                let below = if name == "v_mbcnt_lo_u32_b32" {
                    lane
                } else {
                    let (k32, z) = (k(self, 32), k(self, 0));
                    let t = self.s.m.bin(c, Bin::Sub, lane, k32);
                    let low = self.s.m.cmp(c, pred::IULT, lane, k32);
                    self.s.m.select(c, low, z, t)
                };
                let m = self.below_mask(c, below);
                let x = self.s.m.bin(c, Bin::And, a, m);
                let n = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 31, &[x]);
                self.s.m.bin(c, Bin::Add, n, b)
            }
            _ => return Err(name),
        })
    }

    /// AMD RDNA 2 ISA, VOPC: classify the input bits without converting to a
    /// host float. Conversion can quiet an sNaN or flush a denormal.
    fn fp_class(&mut self, inst: &Inst, format: &str) -> Result<V, Why> {
        let (bits, fraction, exp_mask) = match format {
            "f16" => (16, 10, 0x1F),
            "f32" => (32, 23, 0xFF),
            _ => (64, 52, 0x7FF),
        };
        let c = self.cur;
        let ty = self.s.m.int(bits.max(32));
        let x = self.fp_bits(inst, 0, bits)?;
        let mask = if bits == 16 { self.half_src(inst, 1)? } else { self.src(inst, 1)? };
        let (zero, sh, emax, frac, sign, quiet) = (
            self.s.m.int_const(ty, 0), self.s.m.int_const(ty, fraction),
            self.s.m.int_const(ty, exp_mask), self.s.m.int_const(ty, (1 << fraction) - 1),
            self.s.m.int_const(ty, (1u64 << (bits - 1)) as i64),
            self.s.m.int_const(ty, 1 << (fraction - 1)),
        );
        let e = self.s.m.bin(c, Bin::LShr, x, sh);
        let e = self.s.m.bin(c, Bin::And, e, emax);
        let m = self.s.m.bin(c, Bin::And, x, frac);
        let neg = self.s.m.bin(c, Bin::And, x, sign);
        let q = self.s.m.bin(c, Bin::And, x, quiet);
        let subnormal = self.s.m.cmp(c, pred::EQ, e, zero);
        let nonfinite = self.s.m.cmp(c, pred::EQ, e, emax);
        let no_frac = self.s.m.cmp(c, pred::EQ, m, zero);
        let has_frac = self.s.m.cmp(c, pred::NE, m, zero);
        let neg = self.s.m.cmp(c, pred::NE, neg, zero);
        let quiet = self.s.m.cmp(c, pred::NE, q, zero);
        // Positive classes 6-9, negative classes 11 minus those.
        let zero = self.s.m.i32(0);
        let (one, six, seven, eight, nine, eleven) = (self.s.m.i32(1), self.s.m.i32(6),
            self.s.m.i32(7), self.s.m.i32(8), self.s.m.i32(9), self.s.m.i32(11));
        let denorm_or_zero = self.s.m.select(c, no_frac, six, seven);
        let class = self.s.m.select(c, subnormal, denorm_or_zero, eight);
        let class = self.s.m.select(c, nonfinite, nine, class);
        let mirrored = self.s.m.bin(c, Bin::Sub, eleven, class);
        let class = self.s.m.select(c, neg, mirrored, class);
        let nan = self.s.m.bin(c, Bin::And, nonfinite, has_frac);
        let nan_class = self.s.m.select(c, quiet, one, zero);
        let class = self.s.m.select(c, nan, nan_class, class);
        let bit = self.s.m.bin(c, Bin::LShr, mask, class);
        let bit = self.s.m.bin(c, Bin::And, bit, one);
        Ok(self.s.m.cmp(c, pred::NE, bit, zero))
    }

    /// Compare double encodings in IEEE order using integer operations.
    /// Flipping the sign for nonnegative values and complementing negative
    /// ones makes their unsigned order match their numeric order.
    fn compare_f64(&mut self, inst: &Inst, op: &str) -> Result<V, Why> {
        if matches!(op, "f" | "tru") {
            return Ok(self.s.m.bool(op == "tru"));
        }
        let c = self.cur;
        let (a, b) = (self.fp_bits(inst, 0, 64)?, self.fp_bits(inst, 1, 64)?);
        let ty = self.s.m.int(64);
        let (z, sign, abs, inf, ones) = (self.s.m.int_const(ty, 0), self.s.m.int_const(ty, i64::MIN),
            self.s.m.int_const(ty, i64::MAX), self.s.m.int_const(ty, 0x7FF0_0000_0000_0000),
            self.s.m.int_const(ty, -1));
        let abs_a = self.s.m.bin(c, Bin::And, a, abs);
        let abs_b = self.s.m.bin(c, Bin::And, b, abs);
        let nan_a = self.s.m.cmp(c, pred::IUGT, abs_a, inf);
        let nan_b = self.s.m.cmp(c, pred::IUGT, abs_b, inf);
        let unordered = self.s.m.bin(c, Bin::Or, nan_a, nan_b);
        let yes = self.s.m.bool(true);
        let ordered = self.s.m.bin(c, Bin::Xor, unordered, yes);
        if op == "u" || op == "o" {
            return Ok(if op == "u" { unordered } else { ordered });
        }
        let key = |l: &mut Self, x, magnitude| {
            let zero = l.s.m.cmp(c, pred::EQ, magnitude, z);
            let x = l.s.m.select(c, zero, z, x); // both zeros compare equal
            let neg = l.s.m.cmp(c, pred::SLT, x, z);
            let xor = l.s.m.select(c, neg, ones, sign);
            l.s.m.bin(c, Bin::Xor, x, xor)
        };
        let (a, b) = (key(self, a, abs_a), key(self, b, abs_b));
        let (p, negated) = match op {
            "lt" | "nlt" => (pred::IULT, op == "nlt"),
            "eq" | "neq" => (pred::EQ, op == "neq"),
            "le" | "nle" => (pred::IULE, op == "nle"),
            "gt" | "ngt" => (pred::IUGT, op == "ngt"),
            "lg" | "nlg" => (pred::NE, op == "nlg"),
            "ge" | "nge" => (pred::IUGE, op == "nge"),
            _ => return Err("a double compare predicate"),
        };
        let test = self.s.m.cmp(c, p, a, b);
        let test = self.s.m.bin(c, Bin::And, ordered, test);
        Ok(if negated { self.s.m.bin(c, Bin::Xor, test, yes) } else { test })
    }

    fn compare(&mut self, inst: &Inst, rest: &str) -> Result<(), Why> {
        let (op, ty) = rest.rsplit_once('_').ok_or("a compare")?;
        let c = self.cur;
        let r = match ty {
            "f16" | "f32" | "f64" if op == "class" => self.fp_class(inst, ty)?,
            "f64" => self.compare_f64(inst, op)?,
            "f32" | "f16" => {
                let (a, b) = if ty == "f16" {
                    (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?)
                } else {
                    (self.fsrc(inst, 0)?, self.fsrc(inst, 1)?)
                };
                // `fsrc` may emit two separate bitcasts for the same register,
                // which host CSE later merges into `fcmp ... %f, %f`.
                let same_src = a == b
                    || (self.src0.is_none()
                        && matches!(
                            &inst.fields,
                            Fields::Valu(v)
                                if v.src.len() >= 2
                                    && v.src[0] == v.src[1]
                                    && (v.abs & 1) == ((v.abs >> 1) & 1)
                                    && (v.neg & 1) == ((v.neg >> 1) & 1)
                                    && (v.opsel & 1) == ((v.opsel >> 1) & 1)
                                    && v.sdwa.is_none()
                        ));
                // Some D3D12 drivers (observed on Intel Iris Xe) assume no NaNs
                // for plain `fcmp`, folding `fcmp ord` to true, `fcmp uno` to
                // false (even across distinct registers), and self-comparisons
                // like `fcmp oeq/une x, x` to constants. Use DXIL `IsNaN`
                // (`dx.op.isSpecialFloat.f32`, opcode 8) for ordered/unordered
                // and same-value NaN-sensitive predicates.
                let is_nan = |l: &mut Self, x: V| {
                    l.s.op(c, "isSpecialFloat.f32", Attr::ReadNone, l.s.t.i1, 8, &[x])
                };
                let any_nan = |l: &mut Self, x: V, y: V| {
                    let nx = is_nan(l, x);
                    if same_src {
                        nx
                    } else {
                        let ny = is_nan(l, y);
                        l.s.m.bin(c, Bin::Or, nx, ny)
                    }
                };
                match op {
                    "u" => any_nan(self, a, b),
                    "o" => {
                        let u = any_nan(self, a, b);
                        let yes = self.s.m.bool(true);
                        self.s.m.bin(c, Bin::Xor, u, yes)
                    }
                    "eq" | "le" | "ge" if same_src => {
                        let n = is_nan(self, a);
                        let yes = self.s.m.bool(true);
                        self.s.m.bin(c, Bin::Xor, n, yes)
                    }
                    "neq" | "nge" | "nle" if same_src => is_nan(self, a),
                    // Ordered and not equal, false for a NaN, as `olt | ogt`.
                    // `fcmp one` means the same, but the AMD D3D12 driver
                    // runs it as unordered (true for a NaN) where NVIDIA
                    // and WARP do not; olt and ogt agree on all three.
                    "lg" => {
                        let lt = self.s.m.cmp(c, pred::OLT, a, b);
                        let gt = self.s.m.cmp(c, pred::OGT, a, b);
                        self.s.m.bin(c, Bin::Or, lt, gt)
                    }
                    _ => {
                        let p = match op {
                            "f" => pred::FALSE,
                            "lt" => pred::OLT,
                            "eq" => pred::OEQ,
                            "le" => pred::OLE,
                            "gt" => pred::OGT,
                            "ge" => pred::OGE,
                            "nge" => pred::ULT,
                            "nlg" => pred::UEQ,
                            "ngt" => pred::ULE,
                            "nle" => pred::UGT,
                            "neq" => pred::UNE,
                            "nlt" => pred::UGE,
                            "tru" => pred::TRUE,
                            _ => return Err(inst.name()),
                        };
                        self.s.m.cmp(c, p, a, b)
                    }
                }
            }
            "u64" | "i64" if matches!(op, "f" | "t") => self.s.m.bool(op == "t"),
            "u64" | "i64" => {
                let (a, b) = (self.src64(inst, 0)?, self.src64(inst, 1)?);
                let signed = ty == "i64";
                let p = match op {
                    "eq" => pred::EQ,
                    "ne" | "lg" => pred::NE,
                    "lt" if signed => pred::SLT,
                    "le" if signed => pred::SLE,
                    "gt" if signed => pred::SGT,
                    "ge" if signed => pred::SGE,
                    "lt" => pred::IULT,
                    "le" => pred::IULE,
                    "gt" => pred::IUGT,
                    "ge" => pred::IUGE,
                    _ => return Err("a 64-bit compare"),
                };
                self.s.m.cmp(c, p, a, b)
            }
            "u32" | "i32" | "u16" | "i16" => {
                let (a, b) = if ty.ends_with("16") {
                    let (mut a, mut b) = (self.half_src(inst, 0)?, self.half_src(inst, 1)?);
                    if ty == "i16" {
                        let k16 = self.s.m.i32(16);
                        for x in [&mut a, &mut b] {
                            let t = self.s.m.bin(c, Bin::Shl, *x, k16);
                            *x = self.s.m.bin(c, Bin::AShr, t, k16);
                        }
                    }
                    (a, b)
                } else {
                    (self.src(inst, 0)?, self.src(inst, 1)?)
                };
                let signed = ty == "i32" || ty == "i16";
                let p = match op {
                    "eq" => pred::EQ,
                    "ne" | "lg" => pred::NE,
                    "lt" if signed => pred::SLT,
                    "le" if signed => pred::SLE,
                    "gt" if signed => pred::SGT,
                    "ge" if signed => pred::SGE,
                    "lt" => pred::IULT,
                    "le" => pred::IULE,
                    "gt" => pred::IUGT,
                    "ge" => pred::IUGE,
                    "f" | "t" => {
                        self.lane(inst, Some(op == "t"));
                        return Ok(());
                    }
                    _ => return Err("an integer compare"),
                };
                self.s.m.cmp(c, p, a, b)
            }
            _ => return Err(inst.name()),
        };
        self.lane_bits_for(inst, r);
        Ok(())
    }

    fn lane(&mut self, inst: &Inst, value: Option<bool>) {
        let b = self.s.m.bool(value == Some(true));
        self.lane_bits_for(inst, b);
    }

    fn lane_bits_for(&mut self, inst: &Inst, r: V) {
        let Fields::Valu(v) = &inst.fields else {
            return;
        };
        let code = if inst.name().starts_with("v_cmpx") {
            Some(EXEC)
        } else {
            match v.dst {
                Some(Operand::Scalar { code, .. }) => Some(code),
                _ => None,
            }
        };
        if let Some(lo) = code.and_then(|c| inst.write(Reg::S(c))) {
            self.lane_bits.insert(lo, r);
        }
        for &(_, w) in &inst.writes {
            self.vals
                .entry(w)
                .or_insert(Val::Missing("a lane mask read as data"));
        }
    }

    fn interp(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Some(&interp) = self.interps.get(&site) else {
            return Err("the first half of an interpolation");
        };
        let name = inst.name();
        let (dst, src) = match &inst.fields {
            Fields::Vintrp { dst, src, .. } => (*dst, Some(*src)),
            Fields::Valu(v) => {
                if v.abs != 0 || v.neg != 0 {
                    return Err("interpolation operand modifiers");
                }
                (v.dst.ok_or("an interpolation")?, v.src.get(1).copied())
            }
            _ => unreachable!(),
        };
        let Operand::Vector { reg, .. } = dst else {
            return Err("an interpolation");
        };
        // A float operand of the instruction.
        let float = |l: &mut Self, op: Option<Operand>| -> Result<V, Why> {
            let x = l.operand(inst, &op.ok_or("an interpolation operand")?)?;
            Ok(l.as_float(x))
        };
        // What the result adds to the evaluated channel (a second half's
        // first half, when the halves evaluate on their own).
        let mut add = None;
        let (attribute, eval) = match *interp {
            Interp::Interpolated {
                attribute,
                barycentric,
                ..
            } => {
                let mode = interp_mode(barycentric).ok_or("pull-model interpolation")?;
                let eval = if self.per_vertex(attribute.row) {
                    // The element is read per vertex elsewhere, so it is
                    // constant: weigh its vertices with SV_Barycentrics.
                    let mut ij = [None; 2];
                    for (index, w) in ij.iter_mut().enumerate() {
                        let x = match self.barycentric(barycentric, index as u8) {
                            Val::I32(x) => x,
                            Val::Missing(w) => return Err(w),
                            Val::Bit(_) => unreachable!(),
                        };
                        *w = Some(self.as_float(x));
                    }
                    Eval::Weights {
                        p0: true,
                        i: ij[0],
                        j: ij[1],
                    }
                } else {
                    Eval::Mode(mode)
                };
                (attribute, eval)
            }
            // P0, the provoking vertex: a flat (nointerpolation) input.
            Interp::Vertex {
                attribute,
                vertex: 2,
            } => (attribute, Eval::Mode(1)),
            // P10 and P20: what the first and second halves weigh, P1 - P0
            // and P2 - P0, which is what the hardware holds for an
            // attribute the program also interpolates.
            Interp::Vertex { attribute, vertex } if self.interpolated(attribute.row) => {
                let one = Some(self.s.m.f32(1f32.to_bits()));
                let (i, j) = if vertex == 0 {
                    (one, None)
                } else {
                    (None, one)
                };
                (attribute, Eval::Weights { p0: false, i, j })
            }
            Interp::Vertex { .. } => return Err("a non-provoking vertex value"),
            // A half of an interpolation with computed i or j: P0 + i P10,
            // or the old dst (f16: the third operand) + j P20.
            Interp::Weighted { attribute } if name.starts_with("v_interp_p1") => {
                let i = Some(float(self, src)?);
                (
                    attribute,
                    Eval::Weights {
                        p0: true,
                        i,
                        j: None,
                    },
                )
            }
            Interp::Weighted { attribute } => {
                let old = match &inst.fields {
                    Fields::Valu(v) => v.src.get(2).copied(),
                    _ => Some(dst),
                };
                add = Some(float(self, old)?);
                let j = Some(float(self, src)?);
                (
                    attribute,
                    Eval::Weights {
                        p0: false,
                        i: None,
                        j,
                    },
                )
            }
        };
        let packing = self.half_inputs.get(&attribute.row).copied();
        let f16 = name.ends_with("_f16");
        // The evaluated channel plus `add`, as a float.
        let total = |l: &mut Self, row: u8, ch: u8| -> Result<V, Why> {
            let x = l.eval_channel(row, ch, eval)?;
            let x = l.as_float(x);
            Ok(match add {
                Some(a) => l.s.m.bin(l.cur, Bin::Add, a, x),
                None => x,
            })
        };
        let x = match packing {
            None if f16 || attribute.high => return Err("a 16-bit read of a 32-bit input"),
            None => {
                let x = total(self, attribute.row, attribute.channel)?;
                self.as_int(x)
            }
            Some((pack2, b)) => {
                // 16-bit values: one in the low half of each dword, or two
                // per dword (`packed_half`); the host's are 32-bit floats.
                let half = |l: &mut Self, h: u8| -> Result<Option<V>, Why> {
                    let src = if pack2 {
                        packed_half(&l.s.inputs, attribute.row, b, h)
                            .or_else(|| l.half_input_spills.get(&(attribute.row, h)).copied())
                    } else {
                        h.is_multiple_of(2).then_some((attribute.row, h / 2))
                    };
                    match src {
                        Some((row, ch)) => total(l, row, ch).map(Some),
                        None => Ok(None),
                    }
                };
                let to_f16 = |l: &mut Self, f: V| {
                    let i32t = l.s.t.i32;
                    l.s.op(l.cur, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
                };
                let h = 2 * attribute.channel + u8::from(attribute.high);
                let c = self.cur;
                if name == "v_interp_p1ll_f16" {
                    // An f32 result from 16-bit vertex values.
                    let f = half(self, h)?.ok_or("a packed input half no element holds")?;
                    self.as_int(f)
                } else if f16 {
                    let f = half(self, h)?.ok_or("a packed input half no element holds")?;
                    let lo = to_f16(self, f);
                    self.low_half(inst, reg, lo)
                } else if matches!(eval, Eval::Mode(1)) {
                    // The whole dword of a flat input: both halves.
                    let h = 2 * attribute.channel;
                    let z = self.s.m.i32(0);
                    let lo = match half(self, h)? {
                        Some(f) => to_f16(self, f),
                        None => z,
                    };
                    let hi = match half(self, h + 1)? {
                        Some(f) => to_f16(self, f),
                        None => z,
                    };
                    let k16 = self.s.m.i32(16);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                } else {
                    return Err("32-bit interpolation of a 16-bit input");
                }
            }
        };
        // A pixel interpolates in every lane; guards only hide inactive ones.
        self.write(inst, Reg::V(reg), x, None);
        Ok(())
    }

    /// Whether the program reads attribute row `row` per vertex (an
    /// interpolation with computed barycentrics, or P10 or P20), or the row
    /// must be constant for another of its elements: then its elements are
    /// declared constant and read with `attributeAtVertex`.
    fn per_vertex(&self, row: u8) -> bool {
        self.x.io.interps.iter().any(|(_, i)| match *i {
            Interp::Weighted { attribute } => attribute.row == row,
            Interp::Vertex { attribute, vertex } => {
                attribute.row == row && vertex != 2 && self.interpolated(row)
            }
            Interp::Interpolated { .. } => false,
        }) || self.constant_row(row)
    }

    /// Whether row `row` holds an element that must be constant (an
    /// integer, `SV_IsFrontFace`, a provoking-vertex read) next to an
    /// interpolated one. DXIL wants one interpolation mode per packed row,
    /// and the row keeps the console's register (the stages link by it), so
    /// its interpolated elements are weighed from their vertices instead.
    fn constant_row(&self, row: u8) -> bool {
        let interps = || {
            self.x
                .io
                .interps
                .iter()
                .filter(move |(_, i)| match *i {
                    Interp::Interpolated { attribute, .. }
                    | Interp::Vertex { attribute, .. }
                    | Interp::Weighted { attribute } => attribute.row == row,
                })
                .map(|(_, i)| i)
        };
        let interpolated = interps().any(|i| matches!(i, Interp::Interpolated { .. }));
        if !interpolated {
            return false;
        }
        let flat_read = interps().any(|i| matches!(i, Interp::Vertex { vertex: 2, .. }));
        // The signature rows the hardware row carries (a second one for
        // packed 16-bit pairs).
        let rows: Vec<u32> = match &self.x.native.io {
            StageIo::Ps { inputs, .. } => inputs
                .iter()
                .filter(|i| i.row == row && i.fp16)
                .map(|i| u32::from(i.row2))
                .chain([u32::from(row)])
                .collect(),
            _ => vec![u32::from(row)],
        };
        let integer = self
            .x
            .inputs
            .iter()
            .any(|e| e.row.is_some_and(|r| rows.contains(&r)) && e.comp != 9);
        flat_read || integer
    }

    /// Whether the program interpolates attribute row `row` (so the
    /// hardware holds P1 - P0 and P2 - P0 for it).
    fn interpolated(&self, row: u8) -> bool {
        self.x.io.interps.iter().any(|(_, i)| match *i {
            Interp::Interpolated { attribute, .. } | Interp::Weighted { attribute } => {
                attribute.row == row
            }
            Interp::Vertex { .. } => false,
        })
    }

    /// Channel `ch` of input row `row` evaluated as `eval` says, as raw bits.
    fn eval_channel(&mut self, row: u8, ch: u8, eval: Eval) -> Result<V, Why> {
        let (id, col) = self
            .find_input(row, ch)
            .ok_or("an input the signature lacks")?;
        let mode = match eval {
            Eval::Mode(mode) => mode,
            Eval::Weights { .. } => 1,
        };
        let e = &mut self.s.inputs[id];
        if e.interp != mode && e.usage != 0 {
            return Err("one input interpolated two ways");
        }
        e.interp = mode;
        let Eval::Weights { p0, i, j } = eval else {
            return match self.load_any_input(id, col) {
                Val::I32(x) => Ok(x),
                _ => Err("an input"),
            };
        };
        let v0 = self.vertex_input(id, col, 0);
        let c = self.cur;
        let mut acc = p0.then_some(v0);
        for (k, w) in [(1, i), (2, j)] {
            let Some(w) = w else { continue };
            let vk = self.vertex_input(id, col, k);
            let d = self.s.m.bin(c, Bin::Sub, vk, v0);
            acc = Some(match acc {
                Some(a) => self.fmad(w, d, a),
                None => self.s.m.bin(c, Bin::Mul, w, d),
            });
        }
        let acc = acc.ok_or("an interpolation")?;
        Ok(self.as_int(acc))
    }

    /// Vertex `k` (0 the provoking vertex, as `barycentric` orders them)
    /// of a constant input element's channel, as a float.
    fn vertex_input(&mut self, id: usize, col: u8, k: u8) -> V {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let int = e.comp != 9;
        let (ty, suffix) = if int {
            (self.s.t.i32, ".i32")
        } else {
            (self.s.t.f32, ".f32")
        };
        self.s.flags |= flags::BARYCENTRICS;
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.i8(k),
        ];
        let v = self.s.op(
            entry,
            &format!("attributeAtVertex{suffix}"),
            Attr::ReadNone,
            ty,
            137,
            &args,
        );
        if int {
            self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.f32)
        } else {
            v
        }
    }

    /// `x` (an f16 in the low bits) as the low half of VGPR `reg`: 16-bit
    /// results keep the register's high half (GFX10).
    fn low_half(&mut self, inst: &Inst, reg: u8, x: V) -> V {
        let c = self.cur;
        let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let (high, low) = (self.s.m.i32(0xFFFF_0000), self.s.m.i32(0xFFFF));
        let old = self.s.m.bin(c, Bin::And, old, high);
        let x = self.s.m.bin(c, Bin::And, x, low);
        self.s.m.bin(c, Bin::Or, old, x)
    }

    fn image(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Image {
            data,
            addr,
            dmask,
            dim,
            unorm,
            a16,
            d16,
            tfe,
            lwe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        if *tfe || *lwe {
            return Err("an image access with tfe or lwe");
        }
        let name = inst.name();
        if name.starts_with("image_bvh") {
            // shortcut: rays hit nothing (no node after this one): nothing is
            // in shadow, and indirect light is the sky's.
            let guard = self.guard(b, i)?;
            let Operand::Vector { reg, .. } = *data else {
                return Err("a ray intersection's result");
            };
            if name == "image_bvh64_intersect_ray" && !*a16 && self.x.res.addresses {
                let mut regs = Vec::new();
                for a in addr {
                    if let Operand::Vector { reg, dwords } = *a {
                        regs.extend((0..dwords).map(|k| reg + k));
                    }
                }
                let words = regs
                    .iter()
                    .map(|&r| {
                        let v = inst.read(Reg::V(r)).ok_or("a ray")?;
                        self.i32_of(v)
                    })
                    .collect::<Result<Vec<V>, Why>>()?;
                let words: [V; 12] = words.try_into().map_err(|_| "a ray")?;
                let r = self.intersect(&words, guard);
                for (k, x) in r.into_iter().enumerate() {
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let none = self.s.m.i32(u32::MAX);
            for k in 0..4 {
                self.write(inst, Reg::V(reg + k), none, guard);
            }
            return Ok(());
        }
        // A load or sample with no channel enabled returns the first.
        let dmask = if *dmask == 0 { &1 } else { dmask };
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let (kind, coords) = match dim {
            0 => (1, 1),
            1 => (2, 2),
            2 => (4, 3),
            // Sampled (face coordinates and face, see `cube_direction`) or
            // sized only.
            3 if name.starts_with("image_sample")
                || name.starts_with("image_gather")
                || name == "image_get_resinfo" =>
            {
                (5, 3)
            }
            4 => (6, 2),
            5 => (7, 3),
            6 if name == "image_load" || name == "image_get_resinfo" => (3, 2),
            7 if name == "image_load" || name == "image_get_resinfo" => (8, 3),
            _ => return Err("an image dimension (cube loads or MSAA)"),
        };
        // Address VGPRs in order.
        let mut regs = Vec::new();
        for a in addr {
            if let Operand::Vector { reg, dwords } = *a {
                regs.extend((0..dwords).map(|k| reg + k));
            }
        }
        // Full 32-bit address VGPR at `reg_idx` (offset, bias, z-compare).
        let addr_u32 = |l: &mut Self, reg_idx: usize| -> Result<V, Why> {
            let r = *regs.get(reg_idx).ok_or("an image address")?;
            let v = inst.read(Reg::V(r)).ok_or("an image address")?;
            l.i32_of(v)
        };
        // With a16, two 16-bit components per VGPR starting at `base_reg`, low half first.
        let addr_packed = |l: &mut Self, base_reg: usize, k: usize| -> Result<V, Why> {
            let x = addr_u32(l, base_reg + if *a16 { k / 2 } else { k })?;
            if !*a16 {
                return Ok(x);
            }
            let c = l.cur;
            let (sh, mask) = (l.s.m.i32(16 * (k as u32 % 2)), l.s.m.i32(0xFFFF));
            let x = l.s.m.bin(c, Bin::LShr, x, sh);
            Ok(l.s.m.bin(c, Bin::And, x, mask))
        };
        let addr_val = |l: &mut Self, k: usize| -> Result<V, Why> { addr_packed(l, 0, k) };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked image")?;
        let store = name.starts_with("image_store");
        let uav = store
            || self
                .used_kind(src)
                .is_some_and(|u| u.kind == ResourceType::Uav);
        let props = (kind | if uav { 0x1000 } else { 0 }, 0x409);
        if let Some(op) = name.strip_prefix("image_atomic_") {
            // GLC returns the old value. DXIL implements subtraction by adding
            // the two's complement of the operand at the same width.
            let code = match op {
                "add" | "sub" => 0,
                "and" => 1,
                "or" => 2,
                "xor" => 3,
                "smin" => 4,
                "smax" => 5,
                "umin" => 6,
                "umax" => 7,
                "swap" => 8,
                "cmpswap" => u32::MAX,
                _ => return Err(name),
            };
            let Fields::Image { cache, .. } = &inst.fields else {
                unreachable!()
            };
            if *d16 {
                return Err("a 16-bit image atomic");
            }
            // Compare-exchange consumes a replacement followed by a comparand.
            let compare = code == u32::MAX;
            let wide = match (compare, *dmask) {
                (false, 1) | (true, 3) => false,
                (false, 3) | (true, 15) => true,
                _ => return Err("an image atomic data mask"),
            };
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image atomic data");
            };
            let width = if wide { 2 } else { 1 };
            let vals = self.vgprs(inst, data, if compare { 2 * width } else { width })?;
            let (ty, component) = if wide {
                self.s.flags |= flags::INT64 | flags::ATOMIC_TYPED64 | flags::ATOMIC_HEAP64;
                (self.s.m.int(64), 0x107) // one uint64_t in an R32G32_UINT view
            } else {
                (self.s.t.i32, 0x105)
            };
            let original = if wide {
                self.join64(vals[0], vals[1])
            } else {
                vals[0]
            };
            let value = if op == "sub" {
                let zero = self.s.m.int_const(ty, 0);
                self.s.m.bin(c, Bin::Sub, zero, original)
            } else {
                original
            };
            let h = self.handle(site, src, false, (kind | 0x1000, component))?;
            let mut args = vec![h];
            let (intrinsic, opcode) = if compare {
                let comparand = if wide {
                    self.join64(vals[2], vals[3])
                } else {
                    vals[1]
                };
                args.extend(cs);
                args.extend([comparand, value]);
                let intrinsic = if wide {
                    "atomicCompareExchange.i64"
                } else {
                    "atomicCompareExchange.i32"
                };
                (intrinsic, 79)
            } else {
                args.push(self.s.m.i32(code));
                args.extend(cs);
                args.push(value);
                (
                    if wide {
                        "atomicBinOp.i64"
                    } else {
                        "atomicBinOp.i32"
                    },
                    78,
                )
            };
            let at = self.begin_guarded(guard);
            let r = self
                .s
                .op(self.cur, intrinsic, Attr::NoUnwind, ty, opcode, &args);
            if cache.glc {
                let old = self.end_guarded(at, &[(r, original)])[0];
                if wide {
                    let (lo, hi) = self.split64(old);
                    self.write(inst, Reg::V(reg), lo, None);
                    self.write(inst, Reg::V(reg + 1), hi, None);
                } else {
                    self.write(inst, Reg::V(reg), old, None);
                }
            } else {
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        if store {
            if name != "image_store" {
                return Err("an image store to a mip level");
            }
            // shortcut: stores are taken to write all four channels of a
            // float format (the console converts by the view's format). The
            // data VGPRs hold the dmask's channels in order; the others are
            // written as 0.
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image store data");
            };
            let mut vals = Vec::new();
            for ch in 0..4u8 {
                if dmask >> ch & 1 == 0 {
                    vals.push(self.s.m.f32(0));
                    continue;
                }
                let k = (dmask & ((1 << ch) - 1)).count_ones() as u8;
                // d16: two f16 channels per VGPR, low half first.
                let r = if *d16 { k / 2 } else { k };
                let v = inst.read(Reg::V(reg + r)).ok_or("image store data")?;
                vals.push(if *d16 {
                    let mut x = self.i32_of(v)?;
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(self.cur, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    self.s
                        .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
                } else {
                    self.float_of(v)?
                });
            }
            let h = self.handle(site, src, false, props)?;
            let at = self.begin_guarded(guard);
            let mut args = vec![h];
            args.extend(cs);
            args.extend(vals);
            args.push(self.s.m.i8(0xF));
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                "textureStore.f32",
                Attr::NoUnwind,
                void,
                67,
                &args,
            );
            self.end_guarded(at, &[]);
            return Ok(());
        }
        // Samples and gathers: the DXIL operation follows from the form.
        let form = sample_form(name);
        let gather = form.is_some_and(|f| f.gather);
        let (op, sample) = match name {
            "image_get_lod" if self.s.pixel => (81, true),
            "image_load" | "image_load_mip" => (66, false),
            "image_get_resinfo" => (72, false),
            _ if form.is_some() => (0, true),
            _ => return Err(name),
        };
        if op == 72 {
            // Width, height, depth or layers, levels (per dmask) of the level
            // in the first address: DXIL's dimensions; a channel DXIL leaves
            // out reads 1 (UAVs: one level).
            let h = self.handle(site, src, false, props)?;
            let mip = if uav || matches!(kind, 3 | 8) {
                self.s.m.undef(self.s.t.i32)
            } else {
                addr_val(self, 0)?
            };
            let i32t = self.s.t.i32;
            let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
            let d = self
                .s
                .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
            let one = self.s.m.i32(1);
            let present = match kind {
                1 => [true, false, false, true],
                2 | 3 | 5 | 6 => [true, true, false, true],
                _ => [true, true, true, true],
            };
            let Operand::Vector { reg, .. } = *data else {
                return Err("an image result");
            };
            for (k, ch) in (0..4u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                let x = if ch == 3 && uav || !present[ch as usize] {
                    one
                } else {
                    self.s.m.extract(c, d, ch)
                };
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        if uav && op != 66 {
            return Err("sampling a UAV");
        }
        let h = self.handle(site, src, false, props)?;
        let fu = self.s.m.undef(self.s.t.f32);
        let iu = self.s.m.undef(self.s.t.i32);
        let r = if sample {
            let ss = self
                .descriptors
                .get(&(site, true))
                .copied()
                .ok_or("an untracked sampler")?;
            let f = form.unwrap_or_default();
            if f.c && dmask.count_ones() != 1 {
                return Err("a comparison sample of several channels");
            }
            // Comparison: the sampler is a comparison sampler.
            let sh = self.handle(site, ss, true, (14 | if f.c { 0x8000 } else { 0 }, 0))?;
            // Coordinates the size divides (not the array slice).
            let spatial = coords - usize::from(kind == 6 || kind == 7);
            // Address dwords in order (LLVM ImageDimIntrinsics.td, Mesa
            // ac_build_image_opcode): the texel offset, bias and comparison
            // reference, a dword each (an a16 bias is an f16 in the low
            // half, LLVM SIISelLowering::lowerImage); the derivatives, x
            // then y; the coordinates, then the level or level clamp.
            // Derivatives are 16-bit in the g16 opcodes, coordinates and
            // levels with a16 (GFX10.3 keeps the two apart, as LLVM's
            // lowerImage does where hasG16): two per dword, low half first,
            // each direction starting a dword.
            let whole = usize::from(f.o) + usize::from(f.b) + usize::from(f.c);
            let bias_at = (usize::from(f.o), a16.then_some(0));
            // A cube's gradients are those of its two face coordinates.
            let grads = match (f.d, kind) {
                (false, _) => 0,
                (true, 5) => 2,
                (true, _) => spatial,
            };
            let g16 = f.g16;
            let per_direction = if g16 { grads.div_ceil(2) } else { grads };
            let grad_at = |dir: usize, k: usize| {
                if g16 {
                    (whole + dir * per_direction + k / 2, Some(k as u32 % 2))
                } else {
                    (whole + dir * grads + k, None)
                }
            };
            let first = whole + 2 * per_direction;
            // Coordinate k; k = coords is the level or the level clamp.
            let coord_at = |k: usize| {
                if *a16 {
                    (first + k / 2, Some(k as u32 % 2))
                } else {
                    (first + k, None)
                }
            };
            let raw = |l: &mut Self, at: usize| -> Result<V, Why> {
                let r = *regs.get(at).ok_or("an image address")?;
                let v = inst.read(Reg::V(r)).ok_or("an image address")?;
                l.i32_of(v)
            };
            // A float address: a dword, or an f16 half of one.
            let float = |l: &mut Self, (at, half): (usize, Option<u32>)| -> Result<V, Why> {
                let x = raw(l, at)?;
                let Some(h) = half else {
                    return Ok(l.as_float(x));
                };
                let c = l.cur;
                let (sh, mask) = (l.s.m.i32(16 * h), l.s.m.i32(0xFFFF));
                let x = l.s.m.bin(c, Bin::LShr, x, sh);
                let x = l.s.m.bin(c, Bin::And, x, mask);
                let f32t = l.s.t.f32;
                Ok(l.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]))
            };
            // A float address known at translation (an f16 only when 0).
            let known =
                |l: &mut Self, (at, half): (usize, Option<u32>)| -> Result<Option<f32>, Why> {
                    let x = raw(l, at)?;
                    Ok(l.s.m.known(x).and_then(|bits| match half {
                        None => Some(f32::from_bits(bits as u32)),
                        Some(h) => (bits >> (16 * h) & 0x7FFF == 0).then_some(0.0),
                    }))
                };
            let level = if f.l {
                known(self, coord_at(coords))?
            } else {
                f.lz.then_some(0.0)
            };
            let level_zero = level == Some(0.0);
            // A gather at a level not known to be 0 (`image_gather4_l`):
            // DXIL's gathers read level 0 only, so it is emulated below with
            // four loads at that level.
            let gather_at_level = f.gather && f.l && !f.c && !level_zero;
            // The DXIL operation. SM 6.6 has no comparison at a level, with
            // a bias or with gradients (SampleCmpLevel is SM 6.7,
            // SampleCmpBias and SampleCmpGrad 6.8) and gathers level 0 only;
            // those forms are taken where a level or bias known to be 0 makes
            // them one SM 6.6 has.
            let op = if op == 81 {
                81
            } else if f.gather {
                if f.b {
                    return Err("a gather with a level bias");
                }
                if !f.l && !f.lz {
                    return Err("a gather with implicit level");
                }
                if !level_zero && !gather_at_level {
                    return Err("a gather at a level not known to be 0");
                }
                if gather_at_level && kind != 2 {
                    return Err("a gather with explicit level on a non-2D texture");
                }
                if dmask.count_ones() != 1 {
                    return Err("a gather of several channels");
                }
                if !matches!(kind, 2 | 5 | 7) {
                    return Err("a gather from a 1D or 3D image");
                }
                if f.c { 74 } else { 73 }
            } else if f.c {
                if f.d {
                    return Err("a comparison sample with explicit derivatives");
                }
                if f.l && !level_zero {
                    return Err("a comparison sample at a level not known to be 0");
                }
                if f.b && known(self, bias_at)?.is_none_or(|b| b != 0.0) {
                    return Err("a comparison sample with a level bias");
                }
                if f.l || f.lz {
                    65
                } else if self.s.pixel || self.s.threads.is_some() && self.s.mesh.is_none() {
                    // Implicit derivatives: pixel and compute shaders.
                    64
                } else {
                    return Err(
                        "a comparison sample with implicit derivatives outside pixel and compute shaders",
                    );
                }
            } else if f.d {
                63
            } else if f.l || f.lz {
                62
            } else if f.b {
                61
            } else {
                60
            };
            let mut cs = vec![fu; 4];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = float(self, coord_at(k))?;
            }
            let face = cs[2];
            if kind == 5 {
                if *unorm || op == 81 {
                    return Err("a cube sample with texel coordinates or a level query");
                }
                if f.o {
                    return Err("a texel offset on a cube image");
                }
                let dir = self.cube_direction(cs[0], cs[1], face);
                cs[..3].copy_from_slice(&dir);
            }
            if *unorm {
                // Texel coordinates: divided by the size of level 0.
                let i32t = self.s.t.i32;
                let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                let z = self.s.m.i32(0);
                let d = self
                    .s
                    .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, z]);
                for (k, slot) in cs.iter_mut().enumerate().take(spatial) {
                    let n = self.s.m.extract(c, d, k as u32);
                    let n = self.s.m.cast(c, Cast::UiToFp, n, self.s.t.f32);
                    *slot = self.s.m.bin(c, Bin::SDiv, *slot, n);
                }
            }
            if op == 81 {
                // The clamped then the unclamped level of detail.
                let Operand::Vector { reg, .. } = *data else {
                    return Err("an image result");
                };
                let f32t = self.s.t.f32;
                // The size's dimensions only: an array slice is undef.
                let mut lc = [fu; 3];
                let n = lod_dims(kind);
                lc[..n].copy_from_slice(&cs[..n]);
                for (k, ch) in (0..2u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                    let clamped = self.s.m.bool(ch == 0);
                    let lod = self.s.op(
                        c,
                        "calculateLOD.f32",
                        Attr::ReadOnly,
                        f32t,
                        81,
                        &[h, sh, lc[0], lc[1], lc[2], clamped],
                    );
                    let x = self.as_int(lod);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let offsets = if kind == 5 { 0 } else { spatial.min(3) };
            let zero = self.s.m.i32(0);
            let mut os = vec![iu; 3];
            for o in os.iter_mut().take(offsets) {
                *o = zero;
            }
            if f.o {
                // Texel offsets: x in bits 0-5, y in 8-13, z in 16-21, signed.
                let w = raw(self, 0)?;
                let field = |w: i64, k: usize| (w as i32) << (26 - 8 * k) >> 26;
                let immediate = self
                    .s
                    .m
                    .known(w)
                    .filter(|&w| (0..offsets).all(|k| (-8..=7).contains(&field(w, k))));
                if let (Some(w), false) = (immediate, f.gather) {
                    for (k, o) in os.iter_mut().enumerate().take(offsets) {
                        *o = self.s.m.i32(field(w, k) as u32);
                    }
                } else {
                    // Bits at..at+5 sign-extended: shift up, then down.
                    let k26 = self.s.m.i32(26);
                    for (k, o) in os.iter_mut().enumerate().take(offsets) {
                        let left = self.s.m.i32(26 - 8 * k as u32);
                        let t = self.s.m.bin(c, Bin::Shl, w, left);
                        *o = self.s.m.bin(c, Bin::AShr, t, k26);
                    }
                }
                if !f.gather && immediate.is_none() {
                    // A sample's offsets are immediates in -8..7 (gathers
                    // take any, per lane). At a level known here, the
                    // coordinates move by offset / that level's size instead:
                    // the same texels, the coordinate rounded once more (the
                    // console adds the offset to the texel coordinate). A
                    // sampler whose LOD clamp or bias moves the sampled
                    // level away from the one given, or a level past the
                    // view's last, differs.
                    let Some(n) = level.filter(|n| *n >= 0.0 && n.fract() == 0.0) else {
                        return Err(
                            "a texel offset past -8..7 or per lane at a level not known here",
                        );
                    };
                    let i32t = self.s.t.i32;
                    let f32t = self.s.t.f32;
                    let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                    let n = self.s.m.i32(n as u32);
                    let d = self
                        .s
                        .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, n]);
                    for (k, o) in os.iter_mut().enumerate().take(offsets) {
                        let size = self.s.m.extract(c, d, k as u32);
                        let size = self.s.m.cast(c, Cast::UiToFp, size, f32t);
                        let texels = self.s.m.cast(c, Cast::SiToFp, *o, f32t);
                        let shift = self.s.m.bin(c, Bin::SDiv, texels, size);
                        cs[k] = self.s.m.bin(c, Bin::Add, cs[k], shift);
                        *o = zero;
                    }
                }
            }
            // The level clamp, after the coordinates.
            let clamp = if f.cl {
                float(self, coord_at(coords))?
            } else {
                fu
            };
            let mut args = vec![h, sh];
            args.extend(&cs);
            match op {
                60 | 61 if !self.s.pixel => {
                    // Outside pixel shaders the console still takes the level
                    // from each quad of lanes (lane 1 minus lane 0 across,
                    // lane 2 minus lane 0 down); DXIL before validator 1.8
                    // has no implicit derivatives there, so the quad's
                    // differences become explicit gradients. A bias scales
                    // them by 2^bias, the same level for isotropic filtering.
                    self.cross_lane(true)?;
                    let scale = if op == 61 {
                        let bias = float(self, bias_at)?;
                        Some(self.unary(21, bias))
                    } else {
                        None
                    };
                    let g = self.quad_gradients(&cs[..spatial], scale);
                    args.extend(&os);
                    args.extend(g);
                    args.push(clamp);
                    self.s.op(
                        c,
                        "sampleGrad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        63,
                        &args,
                    )
                }
                60 => {
                    args.extend(&os);
                    args.push(clamp);
                    self.s
                        .op(c, "sample.f32", Attr::ReadOnly, self.s.t.res_ret, 60, &args)
                }
                61 => {
                    let bias = float(self, bias_at)?;
                    args.extend(&os);
                    args.extend([bias, clamp]);
                    self.s.op(
                        c,
                        "sampleBias.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        61,
                        &args,
                    )
                }
                63 => {
                    // shortcut: the coarse forms (`_cd`) compute one level
                    // per quad; the host's per pixel is the same when the
                    // gradients agree across the quad.
                    let mut g = vec![fu; 6];
                    for k in 0..grads {
                        g[k] = float(self, grad_at(0, k))?;
                        g[3 + k] = float(self, grad_at(1, k))?;
                    }
                    if kind == 5 {
                        // The host takes the gradients of the direction.
                        let x = self.cube_gradient(g[0], g[1], face);
                        let y = self.cube_gradient(g[3], g[4], face);
                        g[..3].copy_from_slice(&x);
                        g[3..].copy_from_slice(&y);
                    }
                    args.extend(&os);
                    args.extend(g);
                    args.push(clamp);
                    self.s.op(
                        c,
                        "sampleGrad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        63,
                        &args,
                    )
                }
                64 | 65 => {
                    let at = usize::from(f.o) + usize::from(f.b);
                    let reference = float(self, (at, None))?;
                    args.extend(&os);
                    args.push(reference);
                    let name = if op == 64 {
                        args.push(clamp);
                        "sampleCmp.f32"
                    } else {
                        "sampleCmpLevelZero.f32"
                    };
                    self.s
                        .op(c, name, Attr::ReadOnly, self.s.t.res_ret, op, &args)
                }
                62 => {
                    let lod = if f.l {
                        float(self, coord_at(coords))?
                    } else {
                        self.s.m.f32(0)
                    };
                    args.extend(&os);
                    args.push(lod);
                    self.s.op(
                        c,
                        "sampleLevel.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        62,
                        &args,
                    )
                }
                _ => {
                    if gather_at_level {
                        // Console-only Gather at an explicit mip level: DXIL's
                        // textureGather has no LOD argument, so emulate with
                        // four clamped textureLoad calls at that mip in HLSL
                        // gather order: (-,+), (+,+), (+,-), (-,-).
                        let lod = float(self, coord_at(coords))?;
                        let i32t = self.s.t.i32;
                        let f32t = self.s.t.f32;
                        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                        let d0 = self
                            .s
                            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, zero]);
                        let levels = self.s.m.extract(c, d0, 3);
                        let one = self.s.m.i32(1);
                        let max_mip = self.s.m.bin(c, Bin::Sub, levels, one);
                        let fzero = self.s.m.f32(0);
                        let lod = self.binary(35, lod, fzero);
                        let mip = self.s.m.cast(c, Cast::FpToUi, lod, i32t);
                        let over = self.s.m.cmp(c, pred::IUGT, mip, max_mip);
                        let mip = self.s.m.select(c, over, max_mip, mip);
                        let dm = self
                            .s
                            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
                        let w = self.s.m.extract(c, dm, 0);
                        let ht = self.s.m.extract(c, dm, 1);
                        let half = self.s.m.f32(0.5f32.to_bits());
                        let clamp_pair = |l: &mut Self, coord: V, dim: V, off: V| {
                            let df = l.s.m.cast(c, Cast::UiToFp, dim, f32t);
                            let p = l.s.m.bin(c, Bin::Mul, coord, df);
                            let p = l.s.m.bin(c, Bin::Sub, p, half);
                            let fl = l.s.op(c, "unary.f32", Attr::ReadNone, f32t, 27, &[p]);
                            let t0 = l.s.m.cast(c, Cast::FpToSi, fl, i32t);
                            let t0 = l.s.m.bin(c, Bin::Add, t0, off);
                            let t1 = l.s.m.bin(c, Bin::Add, t0, one);
                            let hi = l.s.m.bin(c, Bin::Sub, dim, one);
                            let mut cl = |x: V| {
                                let lo_c = l.s.m.cmp(c, pred::SLT, x, zero);
                                let x = l.s.m.select(c, lo_c, zero, x);
                                let hi_c = l.s.m.cmp(c, pred::SGT, x, hi);
                                l.s.m.select(c, hi_c, hi, x)
                            };
                            (cl(t0), cl(t1))
                        };
                        let (x0, x1) = clamp_pair(self, cs[0], w, os[0]);
                        let (y0, y1) = clamp_pair(self, cs[1], ht, os[1]);
                        let ch = dmask.trailing_zeros();
                        let Operand::Vector { reg, .. } = *data else {
                            return Err("an image result");
                        };
                        let mut xs = Vec::with_capacity(4);
                        for (xi, yi) in [(x0, y1), (x1, y1), (x1, y0), (x0, y0)] {
                            let tl = self.s.op(
                                c,
                                "textureLoad.f32",
                                Attr::ReadOnly,
                                self.s.t.res_ret,
                                66,
                                &[h, mip, xi, yi, iu, iu, iu, iu],
                            );
                            let x = self.s.m.extract(c, tl, ch);
                            xs.push(if *d16 {
                                self.s
                                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[x])
                            } else {
                                self.as_int(x)
                            });
                        }
                        if *d16 {
                            let k16 = self.s.m.i32(16);
                            xs = xs
                                .chunks(2)
                                .map(|pair| {
                                    let lo = pair[0];
                                    match pair.get(1) {
                                        Some(&hi) => {
                                            let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                                            self.s.m.bin(c, Bin::Or, lo, hi)
                                        }
                                        None => lo,
                                    }
                                })
                                .collect();
                        }
                        for (k, x) in xs.into_iter().enumerate() {
                            self.write(inst, Reg::V(reg + k as u8), x, guard);
                        }
                        return Ok(());
                    }
                    args.extend(&os[..2]);
                    args.push(self.s.m.i32(dmask.trailing_zeros()));
                    let name = if op == 74 {
                        let reference = float(self, (usize::from(f.o), None))?;
                        args.push(reference);
                        "textureGatherCmp.f32"
                    } else {
                        "textureGather.f32"
                    };
                    self.s
                        .op(c, name, Attr::ReadOnly, self.s.t.res_ret, op, &args)
                }
            }
        } else {
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let mip = if name == "image_load_mip" || matches!(kind, 3 | 8) {
                addr_val(self, coords)?
            } else if uav {
                // Our declarations have four channels.
                self.s.flags |= flags::TYPED_UAV_LOADS;
                iu
            } else {
                self.s.m.i32(0)
            };
            let mut args = vec![h, mip];
            args.extend(&cs);
            args.extend([iu, iu, iu]);
            self.s.op(
                c,
                "textureLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                66,
                &args,
            )
        };
        let Operand::Vector { reg, .. } = *data else {
            return Err("an image result");
        };
        let channels: Vec<u32> = if gather {
            vec![0, 1, 2, 3]
        } else {
            (0..4).filter(|ch| dmask >> ch & 1 != 0).collect()
        };
        let mut xs = Vec::new();
        for ch in channels {
            let x = self.s.m.extract(c, r, ch);
            xs.push(if *d16 {
                // shortcut: d16 results are taken as floats (rounded to
                // nearest even); integer formats would keep 16 bits.
                let i32t = self.s.t.i32;
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[x])
            } else {
                self.as_int(x)
            });
        }
        if *d16 {
            // Two channels per VGPR, low half first (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            xs = xs
                .chunks(2)
                .map(|p| match *p {
                    [lo, hi] => {
                        let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                        self.s.m.bin(c, Bin::Or, lo, hi)
                    }
                    _ => p[0],
                })
                .collect();
        }
        for (k, x) in xs.into_iter().enumerate() {
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// The direction a cube sample's face coordinates stand for: the
    /// console takes s and t in [1, 2] (`v_cubesc`/`v_cubetc` over
    /// `|v_cubema|`, plus 1.5) and the face id (`v_cubeid`), which the
    /// host's cube sampling turns back into the same texel.
    // shortcut: the view is taken as a cube, not a cube array (whose face
    // operand adds 8 × the slice).
    fn cube_direction(&mut self, s: V, t: V, face: V) -> [V; 3] {
        let c = self.cur;
        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
        let (two, three) = (k(self, 2.0), k(self, 3.0));
        let u = self.s.m.bin(c, Bin::Mul, s, two);
        let u = self.s.m.bin(c, Bin::Sub, u, three);
        let v = self.s.m.bin(c, Bin::Mul, t, two);
        let v = self.s.m.bin(c, Bin::Sub, v, three);
        let z = self.s.m.f32(0x8000_0000);
        let (nu, nv) = (
            self.s.m.bin(c, Bin::Sub, z, u),
            self.s.m.bin(c, Bin::Sub, z, v),
        );
        let (one, mone) = (k(self, 1.0), k(self, -1.0));
        let f = self.s.m.cast(c, Cast::FpToUi, face, self.s.t.i32);
        // Per face +x, -x, +y, -y, +z, -z.
        self.cube_face(
            f,
            [
                [one, mone, u, u, u, nu],
                [nv, nv, one, mone, nv, nv],
                [nu, u, v, nv, one, mone],
            ],
        )
    }

    /// The gradient of [`Self::cube_direction`] for the gradients `ds`, `dt`
    /// of the face coordinates: the direction is linear in them (twice
    /// each, signed per face) and constant along the face's own axis.
    fn cube_gradient(&mut self, ds: V, dt: V, face: V) -> [V; 3] {
        let c = self.cur;
        let two = self.s.m.f32(2.0f32.to_bits());
        let u = self.s.m.bin(c, Bin::Mul, ds, two);
        let v = self.s.m.bin(c, Bin::Mul, dt, two);
        let z = self.s.m.f32(0x8000_0000);
        let (nu, nv) = (
            self.s.m.bin(c, Bin::Sub, z, u),
            self.s.m.bin(c, Bin::Sub, z, v),
        );
        let o = self.s.m.f32(0);
        let f = self.s.m.cast(c, Cast::FpToUi, face, self.s.t.i32);
        self.cube_face(
            f,
            [
                [o, o, u, u, u, nu],
                [nv, nv, o, o, nv, nv],
                [nu, u, v, nv, o, o],
            ],
        )
    }

    /// Each row's entry for the face (an integer, +x -x +y -y +z -z).
    fn cube_face(&mut self, face: V, table: [[V; 6]; 3]) -> [V; 3] {
        let c = self.cur;
        table.map(|row| {
            let mut r = row[0];
            for (n, &x) in row.iter().enumerate().skip(1) {
                let kn = self.s.m.i32(n as u32);
                let is = self.s.m.cmp(c, pred::EQ, face, kn);
                r = self.s.m.select(c, is, x, r);
            }
            r
        })
    }

    fn raw_dword(&mut self, h: V, coords: [V; 2]) -> V {
        let c = self.cur;
        let (mask, align) = (self.s.m.i8(1), self.s.m.i32(4));
        let r = self.s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, self.s.t.res_ret, 139,
            &[h, coords[0], coords[1], mask, align]);
        let x = self.s.m.extract(c, r, 0);
        self.as_int(x)
    }

    fn buffer_compare_exchange(&mut self, h: V, coords: [V; 3], old: V, new: V, wide: bool) -> V {
        let name = if wide { "atomicCompareExchange.i64" } else { "atomicCompareExchange.i32" };
        let ty = self.s.m.type_of(old);
        self.s.op(self.cur, name, Attr::NoUnwind, ty, 79,
            &[h, coords[0], coords[1], coords[2], old, new])
    }

    /// A CAS must run on a real lane and at an in-range address: outside
    /// those cases DXIL can return an unspecified old value.
    fn buffer_bounds(&mut self, h: V, index: Option<V>, off: V, stride: u32, size: u32) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let unused = self.s.m.undef(i32t);
        let dimensions = self.s.op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, unused]);
        let length = self.s.m.extract(c, dimensions, 0);
        if let Some(index) = index {
            let record = self.s.m.cmp(c, pred::IULT, index, length);
            let limit = self.s.m.i32(stride - (size - 1));
            let member = self.s.m.cmp(c, pred::IULT, off, limit);
            return self.s.m.bin(c, Bin::And, record, member);
        }
        let tail = self.s.m.i32(size - 1);
        let last = self.s.m.bin(c, Bin::Add, off, tail);
        let fits = self.s.m.cmp(c, pred::IULT, last, length);
        let no_wrap = self.s.m.cmp(c, pred::IUGE, last, off);
        self.s.m.bin(c, Bin::And, fits, no_wrap)
    }

    fn buffer_store_guard(&mut self, bounds: V, guard: Option<V>) -> V {
        let c = self.cur;
        let g = guard.map_or(bounds, |g| self.s.m.bin(c, Bin::And, g, bounds));
        if !self.s.pixel {
            return g;
        }
        let helper = self.s.op(c, "isHelperLane.i1", Attr::ReadOnly, self.s.t.i1, 221, &[]);
        let no = self.s.m.bool(false);
        self.s.m.select(c, helper, no, g)
    }

    fn buffer(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Buffer {
            data,
            addr,
            soffset,
            offset,
            offen,
            idxen,
            format,
            lds,
            tfe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if *lds || *tfe {
            return Err("a buffer access to LDS or with tfe");
        }
        if format.is_some() && !name.starts_with("tbuffer_load_format_") {
            return Err("an MTBUF store requiring a packed-format write");
        }
        if name.contains("_format_d16_hi_") {
            return Err("a typed buffer access to the high half with an untracked numeric format");
        }
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let store = name.starts_with("buffer_store_");
        let atomic = name.strip_prefix("buffer_atomic_");
        let wide = atomic.is_some_and(|n| n.ends_with("_x2"));
        let atomic = atomic.map(|n| n.strip_suffix("_x2").unwrap_or(n));
        // DXIL atomic operations: add, and, or, xor, signed and unsigned
        // min and max, exchange; compare-exchange apart.
        let native_atomic = match atomic {
            None => None,
            Some("add" | "sub") => Some(0),
            Some("and") => Some(1),
            Some("or") => Some(2),
            Some("xor") => Some(3),
            Some("smin") => Some(4),
            Some("smax") => Some(5),
            Some("umin") => Some(6),
            Some("umax") => Some(7),
            Some("swap") => Some(8),
            Some("cmpswap" | "inc" | "dec" | "csub") => None,
            Some("fcmpswap" | "fmin" | "fmax") => {
                // Mesa aco_insert_fp_mode.cpp, vmem_default_needs.
                return Err("a buffer floating-point atomic requiring the console's denormal mode");
            }
            Some(_) => return Err(name),
        };
        let sub_load = name.strip_prefix("buffer_load_").and_then(subword_load);
        let sub_store = match name {
            "buffer_store_byte" | "buffer_store_byte_d16_hi" => Some(8),
            "buffer_store_short" | "buffer_store_short_d16_hi" => Some(16),
            _ => None,
        };
        // d16: two 16-bit channels per VGPR, low half first (buffer and
        // tbuffer formats alike).
        let d16 = name.contains("_format_d16_");
        let plain = name.replace("_format_d16_", "_format_");
        let (typed, count) = match plain
            .strip_prefix("buffer_load_")
            .or(plain.strip_prefix("buffer_store_"))
            .or(plain.strip_prefix("tbuffer_load_"))
        {
            _ if atomic.is_some() || sub_load.is_some() || sub_store.is_some() => (false, 1),
            Some("dword") => (false, 1),
            Some("dwordx2") => (false, 2),
            Some("dwordx3") => (false, 3),
            Some("dwordx4") => (false, 4),
            Some("format_x") => (true, 1),
            Some("format_xy") => (true, 2),
            Some("format_xyz") => (true, 3),
            Some("format_xyzw") => (true, 4),
            _ => return Err(name),
        };
        let Operand::Vector { reg: areg, .. } = *addr else {
            return Err("a buffer address");
        };
        let vaddr = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a buffer address")?;
            l.i32_of(v)
        };
        let index = if *idxen { Some(vaddr(self, 0)?) } else { None };
        let mut off = self.s.m.i32(u32::from(*offset));
        if *offen {
            let o = vaddr(self, u8::from(*idxen))?;
            off = self.s.m.bin(c, Bin::Add, off, o);
        }
        if !matches!(
            soffset,
            Operand::Scalar { code: NULL, .. } | Operand::Const(128)
        ) {
            let so = self.operand(inst, soffset)?;
            off = self.s.m.bin(c, Bin::Add, off, so);
        }
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked buffer")?;
        let used = self.used_kind(src);
        let stride = used.map_or(0, |u| u.stride);
        if self.by_address(src)
            && let Fields::Buffer { rsrc, .. } = &inst.fields
        {
            if typed && format.is_none() {
                return Err("a typed buffer access by address");
            }
            let (h, start, stride) = self.made_buffer(inst, rsrc)?;
            self.made.insert(site, h);
            off = self.s.m.bin(c, Bin::Add, off, start);
            if let Some(i) = index {
                let rows = self.s.m.bin(c, Bin::Mul, i, stride);
                off = self.s.m.bin(c, Bin::Add, off, rows);
            }
        }
        let index = if self.by_address(src) { None } else { index };
        let (base, extra) = if self.by_address(src) {
            (0, None)
        } else {
            self.built_offset(src)?
        };
        if base != 0 {
            let b = self.s.m.i32(base);
            off = self.s.m.bin(c, Bin::Add, off, b);
        }
        if let Some(x) = extra {
            off = self.s.m.bin(c, Bin::Add, off, x);
        }
        if used.is_some_and(|u| u.kind == ResourceType::Cbv) && !typed {
            if store || atomic.is_some() {
                return Err("a write or atomic operation on a constant buffer");
            }
            // A constant buffer read per lane (a dynamically indexed array).
            // shortcut: the console's CBV descriptor is taken to have a 16-byte
            // stride, as the vector loads of whole rows suggest.
            let h = self.handle(site, src, false, (13, 0x1_0000))?;
            let byte = match index {
                Some(i) => {
                    let sixteen = self.s.m.i32(16);
                    let rows = self.s.m.bin(c, Bin::Mul, i, sixteen);
                    self.s.m.bin(c, Bin::Add, rows, off)
                }
                None => off,
            };
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            if let Some((width, signed, half)) = sub_load {
                let x = self.read_subword(byte, width, signed, |l, at| Ok(l.cbuffer_dword(h, at)))?;
                self.write_subword_result(inst, reg, x, half, guard)?;
                return Ok(());
            }
            for k in 0..count {
                let four = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, byte, four);
                let x = self.cbuffer_dword(h, a);
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        // A dword access by index through a descriptor built in code: the
        // stride is the code's own, so the element becomes a byte offset in
        // the raw root view.
        let (index, off) = match index {
            Some(i) if !typed && format.is_none() => match self.built_element(src, i, off)? {
                Some(byte) => (None, byte),
                // Structured buffers of 16-byte multiples are raw views of
                // their bytes on the host: D3D12 takes structured views only
                // on multiples of the stride, and AMD reads element 0 for all
                // of a raw view's. (Others find a start they fit, and raw
                // views are to start on 16 bytes.)
                None if stride >= 16 && stride.is_multiple_of(16) => {
                    let st = self.s.m.i32(stride);
                    let first = self.s.m.bin(c, Bin::Mul, i, st);
                    (None, self.s.m.bin(c, Bin::Add, first, off))
                }
                None => (Some(i), off),
            },
            x => (x, off),
        };
        let uav = store || atomic.is_some() || used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        let iu = self.s.m.undef(self.s.t.i32);
        if sub_load.is_some() || sub_store.is_some() {
            let data = data.as_ref().ok_or("buffer sub-dword data")?;
            let Operand::Vector { reg, .. } = *data else {
                return Err("buffer sub-dword data");
            };
            let h = if index.is_some() {
                if stride < 4 || !stride.is_multiple_of(4) {
                    return Err("a structured sub-dword buffer access without a dword-aligned stride");
                }
                self.handle(site, src, false, (12 | u, stride))?
            } else {
                self.handle(site, src, false, (11 | u, 0))?
            };
            let coords = |byte| index.map_or([byte, iu], |i| [i, byte]);
            if let Some((width, signed, half)) = sub_load {
                // Root SRVs have no size in DXIL.
                let bounds = (!matches!(src, Source::Built { .. } | Source::Address))
                    .then(|| self.buffer_bounds(h, index, off, stride, width / 8));
                let at = self.begin_guarded(bounds);
                let x = self.read_subword(off, width, signed, |l, at| Ok(l.raw_dword(h, coords(at))))?;
                let zero = self.s.m.i32(0);
                let x = self.end_guarded(at, &[(x, zero)])[0];
                self.write_subword_result(inst, reg, x, half, guard)?;
            } else if let Some(width) = sub_store {
                let mut x = self.vgprs(inst, data, 1)?[0];
                if name.ends_with("_hi") {
                    let sixteen = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, sixteen);
                }
                let bounds = self.buffer_bounds(h, index, off, stride, width / 8);
                let active = self.buffer_store_guard(bounds, guard);
                let at = self.begin_guarded(Some(active));
                self.write_subword(off, width, x, |l, byte, mask, bits| {
                    let [a, b] = coords(byte);
                    let zero = l.s.m.i32(0);
                    l.atomic_update(zero, |l, old| {
                        let x = l.atomic_new_value(AtomicUpdate::MaskOr, old, mask, bits);
                        l.buffer_compare_exchange(h, [a, b, iu], old, x, false)
                    });
                });
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        if let Some(op) = atomic {
            let returns = matches!(inst.fields, Fields::Buffer { cache, .. } if cache.glc);
            if op == "csub" && !returns {
                // LLVM BUFInstructions.td only selects the returning form
                // on GFX10.3.
                return Err("buffer_atomic_csub without the required GLC bit");
            }
            let d = data.as_ref().ok_or("buffer atomic data")?;
            let words = if wide { 2 } else { 1 };
            let vals = self.vgprs(inst, d, if op == "cmpswap" { 2 * words } else { words })?;
            let val = if wide { self.join64(vals[0], vals[1]) } else { vals[0] };
            let ty = self.s.m.type_of(val);
            let zero = self.s.m.int_const(ty, 0);
            let (h, coords) = match index {
                Some(index) => {
                    if stride >= 4 && stride.is_multiple_of(4) {
                        if wide && !stride.is_multiple_of(8) {
                            return Err("a 64-bit buffer atomic without an eight-byte-aligned stride");
                        }
                        let h = self.handle(site, src, false, (12 | u, stride))?;
                        (h, [index, off, iu])
                    } else {
                        if wide {
                            return Err("a 64-bit buffer atomic through a structured buffer of unknown stride");
                        }
                        let byte = self.runtime_stride_offset(src, index, off)?;
                        (self.handle(site, src, false, (11 | u, 0))?, [byte, iu, iu])
                    }
                }
                None => (self.handle(site, src, false, (11 | u, 0))?, [off, iu, iu]),
            };
            let Operand::Vector { reg, .. } = *d else {
                return Err("buffer atomic data");
            };
            if wide {
                self.s.flags |= flags::INT64 | flags::ATOMIC_HEAP64;
            }
            let active = if wide || matches!(op, "sub" | "inc" | "dec" | "csub") {
                let bounds = self.buffer_bounds(h, index, off, stride, 4 * u32::from(words));
                Some(self.buffer_store_guard(bounds, guard))
            } else {
                guard
            };
            let at = self.begin_guarded(active);
            let r = if op == "cmpswap" {
                // cmpswap: data is (new, compare).
                let compare = if wide { self.join64(vals[2], vals[3]) } else { vals[1] };
                self.buffer_compare_exchange(h, coords, compare, val, wide)
            } else if let Some(code) = native_atomic {
                let name = if wide { "atomicBinOp.i64" } else { "atomicBinOp.i32" };
                let code = self.s.m.i32(code);
                let data = if op == "sub" { self.s.m.bin(self.cur, Bin::Sub, zero, val) } else { val };
                self.s.op(self.cur, name, Attr::NoUnwind, ty, 78,
                    &[h, code, coords[0], coords[1], coords[2], data])
            } else {
                let update = match op {
                    "inc" => AtomicUpdate::Inc,
                    "dec" => AtomicUpdate::Dec,
                    "csub" => AtomicUpdate::ClampSub,
                    _ => unreachable!(),
                };
                self.atomic_update(zero, |l, old| {
                    let x = l.atomic_new_value(update, old, val, zero);
                    l.buffer_compare_exchange(h, coords, old, x, wide)
                })
            };
            if returns {
                let r = self.end_guarded(at, &[(r, zero)])[0];
                self.write_atomic_result(inst, reg, r, wide, guard);
            } else {
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        if store {
            let regs = if d16 { count.div_ceil(2) } else { count };
            let mut vals =
                self.vgprs(inst, data.as_ref().ok_or("buffer store data")?, regs as u8)?;
            if d16 {
                // shortcut: d16 data is taken as f16 (integer formats would
                // keep 16 bits), widened for the 32-bit float store.
                let mut wide = Vec::new();
                for k in 0..count {
                    let mut x = vals[(k / 2) as usize];
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                    wide.push(self.as_int(f));
                }
                vals = wide;
            }
            let mut args = Vec::new();
            let (op, name) = if typed {
                // shortcut: typed stores are taken to write a float format.
                // The channels the opcode leaves out are written as 0: the
                // validator has typed stores write all four.
                if *offen || *offset != 0 || matches!(off, V::Inst(_)) {
                    return Err("a typed buffer store at an offset");
                }
                args.push(self.handle(site, src, false, (10 | u, 0x409))?);
                args.extend([index.unwrap_or(self.s.m.i32(0)), iu]);
                for x in vals {
                    args.push(self.as_float(x));
                }
                args.resize(3 + 4, self.s.m.f32(0));
                (69, "bufferStore.f32")
            } else {
                match index {
                    Some(index) => {
                        let stride = used.map_or(0, |u| u.stride);
                        if stride >= 4 && stride.is_multiple_of(4) {
                            args.push(self.handle(site, src, false, (12 | u, stride))?);
                            args.extend([index, off]);
                        } else {
                            let byte = self.runtime_stride_offset(src, index, off)?;
                            args.push(self.handle(site, src, false, (11 | u, 0))?);
                            args.extend([byte, iu]);
                        }
                    }
                    None => {
                        args.push(self.handle(site, src, false, (11 | u, 0))?);
                        args.extend([off, iu]);
                    }
                }
                args.extend(&vals);
                args.resize(3 + 4, iu);
                (140, "rawBufferStore.i32")
            };
            args.push(self.s.m.i8(if typed { 0xF } else { (1u8 << count) - 1 }));
            if !typed {
                args.push(self.s.m.i32(4));
            }
            let at = self.begin_guarded(guard);
            let void = self.s.t.void;
            self.s.op(self.cur, name, Attr::NoUnwind, void, op, &args);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        if let Some(f) = *format {
            if index.is_some() {
                return Err("an indexed MTBUF load requiring a raw view of a structured descriptor");
            }
            let h = self.handle(site, src, false, (11 | u, 0))?;
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            let mut xs = self.format_load(h, off, f, count)?;
            if d16 {
                xs = self.d16_channels(f, xs)?;
            }
            for (k, x) in xs.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let r = if typed {
            if *offen || matches!(off, V::Inst(_)) || *offset != 0 {
                return Err("a typed buffer load at an offset");
            }
            if uav {
                self.s.flags |= flags::TYPED_UAV_LOADS;
            }
            let h = self.handle(site, src, false, (10 | u, 0x409))?;
            let index = index.unwrap_or(self.s.m.i32(0));
            self.s.op(
                c,
                "bufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                68,
                &[h, index, iu],
            )
        } else {
            let mask = self.s.m.i8((1u8 << count) - 1);
            let align = self.s.m.i32(4);
            match index {
                Some(index) => {
                    let stride = used.map_or(0, |u| u.stride);
                    if stride >= 4 && stride.is_multiple_of(4) {
                        let h = self.handle(site, src, false, (12 | u, stride))?;
                        self.s.op(
                            c,
                            "rawBufferLoad.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            139,
                            &[h, index, off, mask, align],
                        )
                    } else {
                        let byte = self.runtime_stride_offset(src, index, off)?;
                        let h = self.handle(site, src, false, (11 | u, 0))?;
                        self.s.op(
                            c,
                            "rawBufferLoad.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            139,
                            &[h, byte, iu, mask, align],
                        )
                    }
                }
                None => {
                    let h = self.handle(site, src, false, (11 | u, 0))?;
                    self.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, off, iu, mask, align],
                    )
                }
            }
        };
        let Some(Operand::Vector { reg, .. }) = *data else {
            return Err("a buffer result");
        };
        if d16 {
            // Two f16 channels per VGPR (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            let i32t = self.s.t.i32;
            for k in 0..count.div_ceil(2) {
                let mut x = None;
                for ch in (2 * k..(2 * k + 2).min(count)).rev() {
                    let f = self.s.m.extract(c, r, ch);
                    let h = self
                        .s
                        .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
                    x = Some(match x {
                        Some(hi) => {
                            let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                            self.s.m.bin(c, Bin::Or, h, hi)
                        }
                        None => h,
                    });
                }
                self.write(inst, Reg::V(reg + k as u8), x.expect("a channel"), guard);
            }
            return Ok(());
        }
        for k in 0..count {
            let x = self.s.m.extract(c, r, k);
            let x = self.as_int(x);
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    fn export(&mut self, b: usize, i: usize) -> Result<(), Why> {
        let e = *self
            .exports
            .get(&(b, i))
            .ok_or("an export I/O raising did not see")?;
        if e.compressed {
            return Err("a compressed export");
        }
        let guard = self.guard(b, i)?;
        let mesh = self.s.mesh.is_some();
        let c = self.cur;
        if let Some(g) = guard
            && !mesh
        {
            if !(e.done && self.s.pixel) {
                return Err("a conditional export");
            }
            let t = self.s.m.bool(true);
            let kill = self.s.m.bin(c, Bin::Xor, g, t);
            self.s
                .op(c, "discard", Attr::NoUnwind, self.s.t.void, 82, &[kill]);
        }
        if mesh && e.target == Target::Primitive {
            return self.emit_indices(e, guard);
        }
        let param_row = |usage: Usage, row: u8| match usage {
            // The 2020 compiler leaves a mesh program's first entry zeroed
            // (usage "position"): its row is the one no other entry names.
            Usage::Other(0) if mesh => self.unnamed_row(),
            _ => Some(row),
        };
        // (output element, channel) per exported channel; per-primitive
        // outputs are elements of `prims`.
        let mut stores = Vec::new();
        let primitive = matches!(e.target, Target::PrimitiveParam { .. });
        let outputs = if primitive {
            &self.s.prims
        } else {
            &self.s.outputs
        };
        let by_kind = |kind: u8| outputs.iter().position(|o| o.kind == kind);
        // 16-bit vertex outputs: the halves each dword carries, as (output
        // element, channel, high half) per channel.
        let halves = match e.target {
            Target::Param { usage, row } => {
                param_row(usage, row).and_then(|r| self.half_outputs.get(&r).map(|&h| (r, h)))
            }
            _ => None,
        };
        if let Some((row, (pack2, b))) = halves {
            for ch in 0..4u8 {
                let Some(value) = e.values[usize::from(ch)] else {
                    continue;
                };
                for high in [false, true] {
                    let h = 2 * ch + u8::from(high);
                    let src = if pack2 {
                        packed_half(outputs, row, b, h)
                    } else {
                        (!high).then_some((row, ch))
                    };
                    let found = src.and_then(|(r, c)| {
                        outputs.iter().position(|o| {
                            o.row == Some(u32::from(r)) && c >= o.col && c < o.col + o.cols
                        })
                    });
                    if let Some(id) = found {
                        let col = src.expect("found").1 - outputs[id].col;
                        stores.push((id, col, value, Some(high)));
                    }
                }
            }
        }
        for ch in 0..4u8 {
            let Some(value) = e.values[usize::from(ch)] else {
                continue;
            };
            if halves.is_some() {
                break;
            }
            let found = match e.target {
                Target::Color(n) => outputs
                    .iter()
                    .position(|o| o.kind == 16 && o.row == Some(u32::from(n)))
                    .map(|id| (id, ch)),
                Target::Position(0) => by_kind(3).map(|id| (id, ch)),
                Target::Depth => match ch {
                    0 => by_kind(17)
                        .or(by_kind(18))
                        .or(by_kind(19))
                        .map(|id| (id, 0)),
                    1 => by_kind(20).map(|id| (id, 0)),
                    2 => by_kind(14).map(|id| (id, 0)),
                    _ => None,
                },
                Target::Param { usage, row } | Target::PrimitiveParam { usage, row } => {
                    let row = param_row(usage, row);
                    outputs
                        .iter()
                        .position(|o| {
                            row.is_some_and(|r| o.row == Some(u32::from(r)))
                                && ch >= o.col
                                && ch < o.col + o.cols
                        })
                        .map(|id| (id, ch - outputs[id].col))
                }
                Target::Null | Target::Unread => None,
                Target::Position(n @ 1..=2) => {
                    // shortcut: without the misc vector (point size, render
                    // target or viewport index) pos1 and pos2 carry the clip
                    // then cull distances, in semantic order.
                    if outputs.iter().any(|o| matches!(o.kind, 4 | 5)) {
                        return Err("render-target or viewport index exports");
                    }
                    let mut dists: Vec<usize> = (0..outputs.len())
                        .filter(|&i| matches!(outputs[i].kind, 6 | 7))
                        .collect();
                    dists.sort_by_key(|&i| (outputs[i].kind, outputs[i].index));
                    let flat: Vec<(usize, u8)> = dists
                        .iter()
                        .flat_map(|&i| (0..outputs[i].cols).map(move |c| (i, c)))
                        .collect();
                    // Channels past the distances the signature declares are
                    // ones the clipper does not enable (`PA_CL_VS_OUT_CNTL`).
                    let k = usize::from(n - 1) * 4 + usize::from(ch);
                    flat.get(k).copied()
                }
                Target::Position(_) => return Err("clip, cull or point-size exports"),
                Target::Primitive => return Err("a primitive export"),
            };
            if let Some((id, col)) = found {
                stores.push((id, col, value, None));
            }
        }
        fn elements<'l>(l: &'l mut Lower, primitive: bool) -> &'l mut Vec<SigElement> {
            if primitive {
                &mut l.s.prims
            } else {
                &mut l.s.outputs
            }
        }
        if stores
            .iter()
            .any(|s| s.3.is_none() && elements(self, primitive)[s.0].half)
        {
            return Err("a 16-bit output the stage data does not mark");
        }
        // A mesh lane writes its own vertex or primitive, where it exports.
        let at = if mesh {
            self.begin_guarded(guard)
        } else {
            None
        };
        let (name, opcode) = match (mesh, primitive) {
            (false, _) => ("storeOutput", 5),
            (true, false) => ("storeVertexOutput", 171),
            (true, true) => ("storePrimitiveOutput", 172),
        };
        for (id, col, value, half) in stores {
            let mut x = self.i32_of(value)?;
            if let Some(high) = half {
                // An f16 half, declared a 32-bit float on the host.
                let c = self.cur;
                if high {
                    let k16 = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, k16);
                }
                let f32t = self.s.t.f32;
                let f = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                x = self.as_int(f);
            }
            let el = &mut elements(self, primitive)[id];
            el.usage |= 1 << col;
            let (suffix, x) = match el.comp {
                9 if half.is_some() => (".f32", self.as_float(x)),
                9 => {
                    let f = match self.floats.get(&value) {
                        Some(&f) => f,
                        None => self.as_float(x),
                    };
                    (".f32", f)
                }
                _ => (".i32", x),
            };
            let mut args = vec![
                self.s.m.i32(id as u32),
                self.s.m.i32(0),
                self.s.m.i8(col),
                x,
            ];
            if mesh {
                args.push(self.mesh_thread());
            }
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                &format!("{name}{suffix}"),
                Attr::NoUnwind,
                void,
                opcode,
                &args,
            );
        }
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: `exp prim` as the lane's primitive. Bit 31 culls it (all
    /// indices 0: no area).
    fn emit_indices(&mut self, e: &Export, guard: Option<V>) -> Result<(), Why> {
        let v = e.values[0].ok_or("a primitive export without its indices")?;
        let x = self.i32_of(v)?;
        let at = self.begin_guarded(guard);
        let c = self.cur;
        let z = self.s.m.i32(0);
        let culled = self.s.m.cmp(c, pred::SLT, x, z);
        let mut args = vec![self.mesh_thread()];
        for k in 0..3 {
            let (shift, mask) = (self.s.m.i32(10 * k), self.s.m.i32(0x3FF));
            let i = self.s.m.bin(c, Bin::LShr, x, shift);
            let i = self.s.m.bin(c, Bin::And, i, mask);
            args.push(self.s.m.select(c, culled, z, i));
        }
        let void = self.s.t.void;
        self.s
            .op(c, "emitIndices", Attr::NoUnwind, void, 169, &args);
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// The per-vertex arbitrary row no mesh output entry names but one
    /// (the 2020 compiler's zeroed entry).
    fn unnamed_row(&self) -> Option<u8> {
        let StageIo::Ms(ms) = &self.x.native.io else {
            return None;
        };
        let named = |r: u32| {
            ms.outputs
                .iter()
                .any(|o| o.param & 0x80 == 0 && o.usage != Usage::Other(0) && u32::from(o.row) == r)
        };
        let free: std::collections::BTreeSet<u32> = self
            .s
            .outputs
            .iter()
            .filter(|o| o.kind == 0)
            .filter_map(|o| o.row)
            .filter(|&r| !named(r))
            .collect();
        match free.into_iter().collect::<Vec<_>>()[..] {
            [r] => u8::try_from(r).ok(),
            _ => None,
        }
    }

    /// The program's natural loops (from their reachable back edges), and
    /// for each that lanes leave through one block (every edge into it from
    /// the loop): the scalar values written in the loop that code after it
    /// reads (see [`Lower::leave_loops`]).
    fn find_loop_exits(&mut self) {
        let p = self.x.p;
        let reachable = |l: &Self, q: usize| l.block[q].is_some();
        let mut succs: Vec<Vec<usize>> = vec![Vec::new(); p.blocks.len()];
        for (b, block) in p.blocks.iter().enumerate() {
            if reachable(self, b) {
                for &q in block.preds.iter().filter(|&&q| reachable(self, q)) {
                    succs[q].push(b);
                }
            }
        }
        let scalar = |r: Reg| match r {
            Reg::S(c) => c != EXEC && c != EXEC + 1,
            Reg::Scc => true,
            _ => false,
        };
        let mut headers: Vec<usize> = self.counters.keys().copied().collect();
        headers.sort_unstable();
        for h in headers {
            let mut body: HashSet<usize> = HashSet::from([h]);
            let mut work: Vec<usize> = p.blocks[h]
                .preds
                .iter()
                .copied()
                .filter(|&q| reachable(self, q) && self.rpo[h] <= self.rpo[q])
                .collect();
            while let Some(q) = work.pop() {
                if body.insert(q) {
                    work.extend(
                        p.blocks[q]
                            .preds
                            .iter()
                            .copied()
                            .filter(|&r| reachable(self, r)),
                    );
                }
            }
            let mut exits: Vec<usize> = body
                .iter()
                .flat_map(|&b| succs[b].iter().copied())
                .filter(|t| !body.contains(t))
                .collect();
            exits.sort_unstable();
            exits.dedup();
            let single = match exits[..] {
                [e] if p.blocks[e]
                    .preds
                    .iter()
                    .all(|q| !reachable(self, *q) || body.contains(q)) =>
                {
                    Some(e)
                }
                _ => None,
            };
            // A waterfall loop runs once with each lane's own descriptor (its
            // compares hold in every lane), so its scalars stay each lane's.
            let waterfall = self
                .x
                .res
                .waterfall
                .compares
                .iter()
                .any(|site| body.contains(&site.0));
            if let (Some(e), true) = (single, waterfall) {
                // Listed without values: the exit joins nothing either.
                self.loop_exits.entry(e).or_default();
            } else if let Some(e) = single {
                let mut written = HashSet::new();
                for &b in &body {
                    let block = &p.blocks[b];
                    let phis = block.phis.iter().map(|phi| (phi.reg, phi.value));
                    let writes = block.insts.iter().flat_map(|i| i.writes.iter().copied());
                    written.extend(
                        phis.chain(writes)
                            .filter(|&(r, _)| scalar(r))
                            .map(|(_, v)| v),
                    );
                }
                let mut after = Vec::new();
                let mut seen = HashSet::new();
                for (b, block) in p.blocks.iter().enumerate() {
                    if body.contains(&b) || !reachable(self, b) {
                        continue;
                    }
                    let reads = block
                        .insts
                        .iter()
                        .flat_map(|i| i.reads.iter().map(|r| r.1))
                        .chain(block.phis.iter().flat_map(|phi| phi.args.iter().copied()));
                    for v in reads {
                        if written.contains(&v) && seen.insert(v) {
                            after.push(v);
                        }
                    }
                }
                self.loop_exits.entry(e).or_default().push((h, after));
            }
            self.loop_bodies.insert(h, body);
        }
    }

    /// Entering `b`, a join of branches (not a loop header): its scalar
    /// joins as the wave has them. The console runs each path some lane
    /// takes, in code order, with one set of scalar registers, so after the
    /// join every lane reads what the last path run left; here each lane
    /// reads what its own path left. Each lane's path is ranked by its code
    /// order, and the lanes of the highest rank hold the wave's values.
    // shortcut: a scalar both paths change is the later path's own result,
    // not that path run on the earlier one's.
    fn join_scalars(&mut self, b: usize) {
        let p = self.x.p;
        if self.counters.contains_key(&b) {
            return;
        }
        let scalar = |r: Reg| match r {
            Reg::S(c) => c != EXEC && c != EXEC + 1,
            Reg::Scc => true,
            _ => false,
        };
        let joins: Vec<usize> = (0..p.blocks[b].phis.len())
            .filter(|&i| {
                let phi = &p.blocks[b].phis[i];
                scalar(phi.reg)
                    && matches!(self.vals.get(&phi.value), Some(Val::I32(_) | Val::Bit(_)))
            })
            .collect();
        if joins.is_empty() {
            return;
        }
        // Reachable predecessors and the DXIL block each edge arrives from.
        let mut preds: Vec<(usize, usize)> = Vec::new();
        for &q in &p.blocks[b].preds {
            let Some(tail) = self.tail[q] else { continue };
            let via = self.edges.get(&(q, b)).copied().unwrap_or(tail);
            preds.push((q, via));
        }
        if preds.len() < 2 {
            return;
        }
        let mut order: Vec<usize> = preds.iter().map(|&(q, _)| q).collect();
        order.sort_by_key(|&q| (p.blocks[q].offset, self.rpo[q]));
        order.dedup();
        let c = self.cur;
        let i32t = self.s.t.i32;
        let rank = self.s.m.phi(c, i32t);
        let incoming: Vec<(V, usize)> = preds
            .iter()
            .map(|&(q, via)| {
                let r = order.iter().position(|&x| x == q).expect("a predecessor") as u32;
                (self.s.m.i32(r), via)
            })
            .collect();
        self.s.m.set_phi(rank, incoming);
        let (max, unsigned) = (self.s.m.i8(3), self.s.m.i8(1));
        let wave_max = |l: &mut Self, x: V| {
            l.s.op(
                c,
                "waveActiveOp.i32",
                Attr::NoUnwind,
                i32t,
                119,
                &[x, max, unsigned],
            )
        };
        let last = wave_max(self, rank);
        let mine = self.s.m.cmp(c, pred::EQ, rank, last);
        self.s.flags |= flags::WAVE_OPS;
        let zero = self.s.m.i32(0);
        for i in joins {
            let v = p.blocks[b].phis[i].value;
            let f = match self.vals[&v] {
                Val::I32(x) => {
                    let x = self.s.m.select(c, mine, x, zero);
                    Val::I32(wave_max(self, x))
                }
                Val::Bit(x) => {
                    let x = self.s.m.cast(c, Cast::ZExt, x, i32t);
                    let x = self.s.m.select(c, mine, x, zero);
                    let w = wave_max(self, x);
                    Val::Bit(self.s.m.cmp(c, pred::NE, w, zero))
                }
                Val::Missing(_) => continue,
            };
            self.vals.insert(v, f);
        }
    }

    /// Entering `b`, the block lanes leave loops through: the scalar values
    /// those loops wrote, as the wave's last iteration left them. The console
    /// runs a divergent loop until its last lane leaves, so afterwards every
    /// lane reads the last iteration's scalars; here each lane leaves with
    /// those of its own last iteration. The lanes that ran the most
    /// iterations hold the wave's (a scalar is the same in all of them).
    // shortcut: a value lowered after this block (a loop block later in
    // reverse postorder) keeps each lane's own.
    fn leave_loops(&mut self, b: usize) {
        let Some(loops) = self.loop_exits.get(&b).cloned() else {
            return;
        };
        let c = self.cur;
        let i32t = self.s.t.i32;
        for (h, values) in loops {
            let (count, _) = self.counters[&h];
            let (max, unsigned) = (self.s.m.i8(3), self.s.m.i8(1));
            let wave_max = |l: &mut Self, x: V| {
                l.s.op(
                    c,
                    "waveActiveOp.i32",
                    Attr::NoUnwind,
                    i32t,
                    119,
                    &[x, max, unsigned],
                )
            };
            let last = wave_max(self, count);
            let mine = self.s.m.cmp(c, pred::EQ, count, last);
            self.s.flags |= flags::WAVE_OPS;
            let zero = self.s.m.i32(0);
            let final_of = |l: &mut Self, val: Val| -> Option<Val> {
                Some(match val {
                    Val::I32(x) => {
                        let x = l.s.m.select(c, mine, x, zero);
                        Val::I32(wave_max(l, x))
                    }
                    Val::Bit(x) => {
                        let x = l.s.m.cast(c, Cast::ZExt, x, i32t);
                        let x = l.s.m.select(c, mine, x, zero);
                        let w = wave_max(l, x);
                        Val::Bit(l.s.m.cmp(c, pred::NE, w, zero))
                    }
                    Val::Missing(_) => return None,
                })
            };
            for v in values {
                let Some(val) = self.vals.get(&v).copied() else {
                    continue;
                };
                if let Some(f) = final_of(self, val) {
                    self.finals.insert(v, (f, h));
                }
            }
            // This block's joins of the loop's scalars.
            for phi in &self.x.p.blocks[b].phis {
                let scalar = matches!(phi.reg, Reg::S(r) if r != EXEC && r != EXEC + 1)
                    || phi.reg == Reg::Scc;
                if !scalar {
                    continue;
                }
                if let Some(val) = self.vals.get(&phi.value).copied()
                    && let Some(f) = final_of(self, val)
                {
                    self.vals.insert(phi.value, f);
                }
            }
        }
    }

    fn exit(&mut self, b: usize) -> Result<(), Stop> {
        let offset = self.x.p.blocks[b]
            .insts
            .last()
            .map_or(self.x.p.blocks[b].offset, |i| i.machine.offset);
        let term = match self.x.st.exits[b] {
            LaneExit::Jump(t) if self.counters.contains_key(&t) && self.rpo[t] <= self.rpo[b] => {
                let ok = self.count_iteration(t, pred::IULT);
                let abort = self.abort_block();
                Term::CondBr(ok, self.block[t].expect("reachable"), abort)
            }
            LaneExit::Jump(t) => {
                self.enter_loop(b, t, self.cur);
                Term::Br(self.block[t].expect("reachable"))
            }
            LaneExit::Branch {
                taken, not_taken, ..
            } if self.console_cf => {
                let cond = self.console_branch(b).map_err(|w| unsupported(offset, w))?;
                self.branch_term(b, cond, taken, not_taken)
            }
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                // A negated condition branches the other way round.
                let (cond, taken, not_taken) = match self.x.st.bools.node(when) {
                    Some((atom, Bool::TRUE, Bool::FALSE)) => {
                        let a = self
                            .atom(atom, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (a, not_taken, taken)
                    }
                    _ => {
                        let c = self
                            .boolean(when, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (c, taken, not_taken)
                    }
                };
                self.branch_term(b, cond, taken, not_taken)
            }
            LaneExit::End => {
                if self.s.pixel && !self.exported[b] && !self.has_done_export(b) {
                    // A lane leaving without its final export was killed.
                    let t = self.s.m.bool(true);
                    self.s
                        .op(self.cur, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
                }
                Term::Ret
            }
            LaneExit::Indirect => return Err(unsupported(offset, "an indirect jump")),
        };
        let _ = Exit::End;
        self.s.m.terminate(self.cur, term);
        Ok(())
    }

    /// The terminator of block `b` branching to `taken` where `cond` holds.
    fn branch_term(&mut self, b: usize, cond: V, taken: usize, not_taken: usize) -> Term {
        // A back edge is taken only below the iteration limit.
        let back = |l: &Self, to: usize| l.counters.contains_key(&to) && l.rpo[to] <= l.rpo[b];
        // (No `xor` of a compare: a loop exit testing one hung the
        // AMD driver.)
        let cond = if back(self, taken) {
            let ok = self.count_iteration(taken, pred::IULT);
            self.s.m.bin(self.cur, Bin::And, cond, ok)
        } else if back(self, not_taken) {
            let stop = self.count_iteration(not_taken, pred::IUGE);
            self.s.m.bin(self.cur, Bin::Or, cond, stop)
        } else {
            cond
        };
        // Critical edges get a block of their own, so loops have
        // dedicated exits and preheaders as DXC emits them.
        let (t, f) = (self.edge(b, taken), self.edge(b, not_taken));
        for (to, via) in [(taken, t), (not_taken, f)] {
            if !back(self, to) {
                // From an edge block, or from here.
                let from = if Some(via) == self.block[to] {
                    self.cur
                } else {
                    via
                };
                self.enter_loop(b, to, from);
            }
        }
        Term::CondBr(cond, t, f)
    }

    /// Console flow: whether block `b`'s `s_cbranch_*` is taken, the same
    /// in both halves (wave-wide mask tests through [`Self::any_lane`]).
    fn console_branch(&mut self, b: usize) -> Result<V, Why> {
        let inst = self.x.p.blocks[b].insts.last().ok_or("a branch")?;
        let name = inst.name();
        let c = self.cur;
        let (any, set) = match name {
            "s_cbranch_scc0" | "s_cbranch_scc1" => {
                let v = inst.read(Reg::Scc).ok_or("a branch's SCC")?;
                let scc = self.scc_bit(v)?;
                (scc, name == "s_cbranch_scc1")
            }
            "s_cbranch_vccz" | "s_cbranch_vccnz" | "s_cbranch_execz" | "s_cbranch_execnz" => {
                let reg = if name.contains("vcc") {
                    crate::ir::VCC
                } else {
                    EXEC
                };
                let lo = inst.read(Reg::S(reg)).ok_or("a branch's mask")?;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&lo)
                    .ok_or("a branch on an untracked mask")?;
                (self.any_lane(m)?, name.ends_with("nz"))
            }
            _ => return Err("a branch the console flow does not model"),
        };
        Ok(if set {
            any
        } else {
            let t = self.s.m.bool(true);
            self.s.m.bin(c, Bin::Xor, any, t)
        })
    }

    /// Console flow: an SCC value as a bit. A mask operation's SCC (result
    /// not zero) is whether some lane of the console wave has its bit.
    fn scc_bit(&mut self, v: Value) -> Result<V, Why> {
        if let Some(&Val::Bit(x)) = self.vals.get(&v) {
            return Ok(x);
        }
        let b = *self
            .def_block
            .get(&v)
            .ok_or("an SCC from the program's start")?;
        let inst = self.x.p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == v))
            .ok_or("an SCC without its instruction")?;
        let mask = inst
            .writes
            .iter()
            .filter(|w| matches!(w.0, Reg::S(_)))
            .find_map(|w| self.x.st.masks.get(&w.1).copied())
            .ok_or("an SCC the emitter did not compute")?;
        self.any_lane(mask)
    }

    /// Console flow: whether `m` holds in some lane of the console wave.
    fn any_lane(&mut self, m: Bool) -> Result<V, Why> {
        let c = self.cur;
        let bit = self.boolean(m, c)?;
        let (lo, hi) = self.console_ballot(c, bit);
        let or = self.s.m.bin(c, Bin::Or, lo, hi);
        let z = self.s.m.i32(0);
        Ok(self.s.m.cmp(c, pred::NE, or, z))
    }

    /// Console flow: the console wave's ballot of `bit`, low and high word.
    fn console_ballot(&mut self, at: usize, bit: V) -> (V, V) {
        if self.split {
            let word = self.ballot_word(at, bit);
            return self.exchange(at, word);
        }
        // One host wave is the console wave (the high word is 0 in waves
        // of 32).
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        (self.s.m.extract(at, r, 0), self.s.m.extract(at, r, 1))
    }

    /// The host wave's ballot of `bit` (32 lanes: the first word).
    fn ballot_word(&mut self, at: usize, bit: V) -> V {
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        self.s.m.extract(at, r, 0)
    }

    /// The console thread's flat id in the group (see [`Self::thread_id`]).
    fn console_flat(&mut self) -> V {
        if let Some(f) = self.console_flat {
            return f;
        }
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let flat = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let first = self.s.op(
            e,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[flat],
        );
        let wave = self.s.wave_size.expect("set with console lanes");
        let mask = self.s.m.i32(!(wave - 1));
        let base = self.s.m.bin(e, Bin::And, first, mask);
        let lane = self.s.op(
            e,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        );
        let f = self.s.m.bin(e, Bin::Add, base, lane);
        self.s.flags |= flags::WAVE_OPS;
        self.console_flat = Some(f);
        f
    }

    /// This lane's console lane (0..63, 0..31 in waves of 32) and its
    /// half of a wave64 (0, 1).
    fn console_lane(&mut self) -> (V, V) {
        let f = self.console_flat();
        let e = self.s.entry;
        let (mask, k5) = (self.s.m.i32(self.wave() - 1), self.s.m.i32(5));
        let lane = self.s.m.bin(e, Bin::And, f, mask);
        let half = self.s.m.bin(e, Bin::LShr, lane, k5);
        (lane, half)
    }

    /// Console flow: each half gives `x` (the same in all its lanes) and
    /// both get (the lower half's, the upper half's): through two group-
    /// shared dwords between group barriers, which every thread of the
    /// group reaches (see [`CONSOLE_FLOW`]). A group of 32 threads or
    /// fewer has no upper half: its lanes count as inactive there (0).
    fn exchange(&mut self, at: usize, x: V) -> (V, V) {
        let threads: u32 = self.s.threads.expect("compute").iter().product();
        let z = self.s.m.i32(0);
        if threads <= 32 {
            return (x, z);
        }
        let (_, half) = self.console_lane();
        let slots = match self.xchg {
            Some(g) => g,
            None => {
                let arr = self.s.m.array(2, self.s.t.i32);
                let g = self.s.m.global("xchg", arr, 3);
                self.xchg = Some(g);
                g
            }
        };
        let void = self.s.t.void;
        // Thread-group sync with a group-shared memory fence.
        let mode = self.s.m.i32(1 | 8);
        let mine = self.s.m.element(at, slots, half);
        self.s.m.store(at, mine, x);
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        let one = self.s.m.i32(1);
        let p0 = self.s.m.element(at, slots, z);
        let p1 = self.s.m.element(at, slots, one);
        let lo = self.s.m.load(at, p0);
        let hi = self.s.m.load(at, p1);
        // Both read before the next exchange writes.
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        (lo, hi)
    }

    /// A forward edge into loop header `to` (from DXIL block `via`) starts
    /// its count at 0.
    fn enter_loop(&mut self, _from: usize, to: usize, via: usize) {
        if let Some((_, incoming)) = self.counters.get_mut(&to) {
            let z = self.s.m.i32(0);
            incoming.push((z, via));
        }
    }

    /// A back edge to header `to` (a back edge keeps its block, no
    /// critical-edge block): the count goes up; the result compares it with
    /// [`LOOP_LIMIT`] by `p` (below: go on; at or above: stop).
    fn count_iteration(&mut self, to: usize, p: u8) -> V {
        let c = self.cur;
        let (phi, _) = self.counters[&to];
        let one = self.s.m.i32(1);
        let next = self.s.m.bin(c, Bin::Add, phi, one);
        let limit = self.s.m.i32(super::loop_limit());
        self.counters
            .get_mut(&to)
            .expect("header")
            .1
            .push((next, c));
        self.s.m.cmp(c, p, next, limit)
    }

    /// The block a lane stopped by [`LOOP_LIMIT`] leaves through.
    fn abort_block(&mut self) -> usize {
        if let Some(a) = self.abort {
            return a;
        }
        let a = self.s.m.new_block();
        if self.s.pixel {
            let t = self.s.m.bool(true);
            self.s
                .op(a, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
        }
        self.s.m.terminate(a, Term::Ret);
        self.abort = Some(a);
        a
    }

    /// The DXIL block a branch from IR block `from` to `to` goes to.
    fn edge(&mut self, from: usize, to: usize) -> usize {
        let target = self.block[to].expect("reachable");
        let preds = self.x.p.blocks[to]
            .preds
            .iter()
            .filter(|&&q| self.block[q].is_some())
            .count();
        // Back edges stay as they are: the exiting block is the latch.
        if preds < 2 || self.rpo[to] <= self.rpo[from] {
            return target;
        }
        let e = self.s.m.new_block();
        self.s.m.terminate(e, Term::Br(target));
        self.edges.insert((from, to), e);
        e
    }

    /// A dword a scalar load writes, where the load is a descriptor the
    /// program uses as a resource or sampler.
    fn descriptor_dword(&self, v: Value) -> bool {
        let Some(load) = self.x.p.blocks.iter().enumerate().find_map(|(b, block)| {
            block
                .insts
                .iter()
                .position(|i| i.writes.iter().any(|w| w.1 == v))
                .map(|i| (b, i))
        }) else {
            return false;
        };
        self.x.p.blocks[load.0].insts[load.1]
            .name()
            .starts_with("s_load_dword")
            && self.x.res.descriptors.iter().any(|d| {
                d.source
                    .leaves()
                    .iter()
                    .any(|l| matches!(l, Source::Loaded { load: at, .. } if *at == load))
            })
    }

    fn fill_phis(&mut self) -> Result<(), Stop> {
        self.fill_paths().map_err(|w| unsupported(0, w))?;
        let p = self.x.p;
        // A phi that only another read phi consumes is live too. Without this, a value with no DXIL form that
        // reaches a read through two joins is taken for dead at the first join and replaced by zero below, and the
        // lowering accepts what it refuses when the same value is read directly.
        let mut work: Vec<Value> = self.read_phis.iter().copied().collect();
        while let Some(v) = work.pop() {
            let Some(&(b, i)) = self.phi_at.get(&v) else {
                continue;
            };
            for &arg in &p.blocks[b].phis[i].args {
                if self.phi_at.contains_key(&arg) && self.read_phis.insert(arg) {
                    work.push(arg);
                }
            }
        }
        for (v, b, i) in std::mem::take(&mut self.data_phis) {
            let phi = &p.blocks[b].phis[i];
            let bit = phi.reg == Reg::Scc;
            // A register initialized only on one branch may carry a dead
            // descriptor dword left over from earlier register reuse on the
            // other branch (guarded by a separate predicate). Only treat a
            // descriptor-dword `Missing` as fatal when no incoming edge carries
            // a real data value.
            let has_data = !bit
                && p.blocks[b]
                    .preds
                    .iter()
                    .enumerate()
                    .filter(|&(_, &q)| self.tail[q].is_some())
                    .any(|(n, _)| matches!(self.vals.get(&phi.args[n]), Some(Val::I32(_))));
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let arg = phi.args[n];
                if self.read_phis.contains(&phi.value) && !self.vals.contains_key(&arg) {
                    let _ = self.lazy_barycentric(arg);
                }
                // A root parameter no instruction read.
                if !self.vals.contains_key(&arg)
                    && let Some(&(param, dword)) = self.params.get(&arg)
                    && self.read_phis.contains(&phi.value)
                {
                    let val = self.user(param, dword);
                    self.vals.insert(arg, val);
                }
                // An edge from outside a loop takes the loop's final value.
                let looked = match self.finals.get(&arg) {
                    Some(&(f, h)) if !self.loop_bodies[&h].contains(&q) => Some(f),
                    _ => self.vals.get(&arg).copied(),
                };
                let x = match (looked, bit) {
                    (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => x,
                    (Some(Val::Missing(why)), false)
                        if has_data
                            && matches!(
                                why,
                                "a descriptor's address or format read as data"
                                    | "a descriptor read only as data"
                            ) =>
                    {
                        self.s.m.i32(0)
                    }
                    (Some(Val::Bit(_)), false) if self.read_phis.contains(&phi.value) => {
                        return Err(Stop::Retry(phi.value, "an SCC joined with SGPR values"));
                    }
                    // A descriptor's dwords reaching a join that also takes
                    // data (the registers are reused after the last access).
                    // shortcut: undefined on that edge, assuming the program
                    // reads the joined value only after the data edge.
                    (Some(Val::Missing(_)), _) if self.descriptor_dword(arg) => {
                        if bit {
                            self.s.m.undef(self.s.t.i1)
                        } else {
                            self.s.m.undef(self.s.t.i32)
                        }
                    }
                    // SCC of a mask operation joined with a comparison's: the
                    // edge past a section no lane is in, where it is clear.
                    (Some(Val::Missing(why)), true)
                        if why == LANE_MASK
                            && phi
                                .args
                                .iter()
                                .any(|a| matches!(self.vals.get(a), Some(Val::Bit(_)))) =>
                    {
                        self.s.m.bool(false)
                    }
                    // As above, of a lane mask: the registers held one before.
                    // shortcut: as above, zero there.
                    (Some(Val::Missing(why)), false)
                        if has_data
                            && (why == LANE_MASK
                                || self.x.st.masks.contains_key(&arg)
                                || self.x.st.highs.contains_key(&arg)) =>
                    {
                        self.s.m.i32(0)
                    }
                    (Some(Val::Missing(why)), _) if self.read_phis.contains(&phi.value) => {
                        return Err(Stop::Retry(phi.value, why));
                    }
                    _ if self.read_phis.contains(&phi.value) => {
                        return Err(Stop::Retry(phi.value, "a join of mismatched values"));
                    }
                    _ if bit => self.s.m.bool(false),
                    _ => self.s.m.i32(0),
                };
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        for (phi, incoming) in std::mem::take(&mut self.counters).into_values() {
            self.s.m.set_phi(phi, incoming);
        }
        while let Some((v, b, i)) = self.pending_masks.pop() {
            let phi = &p.blocks[b].phis[i];
            let offset = p.blocks[b].offset;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                // Console flow: every lane arrives by every edge, so a join
                // also takes masks that are constants or built from scalars.
                let arg = &phi.args[n];
                // shortcut: a plain scalar on the edge is the register in
                // another use, taken for a mask no lane is in; right as long
                // as the program does not read the join after that edge.
                let x = match self.x.st.masks.get(arg).or_else(|| self.x.st.others.get(arg)) {
                    Some(&m) => self.boolean(m, qb).map_err(|w| unsupported(offset, w))?,
                    None => self.s.m.bool(false),
                };
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        Ok(())
    }
}
