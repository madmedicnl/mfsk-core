//! LDPC(240, 74) codec with CRC-24, for the FST4W beacon.
//!
//! FST4 and FST4W share one frame (160 symbols: 40 sync + 120 data) and
//! differ only in the message layer and the code: FST4 carries 77
//! information bits under LDPC(240, 101), while **FST4W carries a 50-bit
//! payload** (callsign + locator + power, the WSJT 50-bit types) plus a
//! 24-bit CRC, encoded under **LDPC(240, 74)**. WSJT-X's `genfst4.f90`
//! selects the second with `iwspr=1`; the encoder/decoder are
//! `encode240_74`/`decode240_74`.
//!
//! The implementation is shared, not duplicated: like [`super::ldpc240_101`]
//! this module only carries the protocol-specific tables (in [`tables`])
//! and CRC-24 helpers, and pins the generic
//! [`crate::engine::protocol::FecCodec`] bodies to
//! [`Ldpc240_74Params`] through the [`Ldpc240_74`] wrapper.

pub mod tables;

use crate::engine::protocol::BpPooledFec;
use crate::engine::{FecCodec, FecOpts, FecResult};
use crate::fec::ldpc::bp::{
    BpScratch, bp_decode_generic_kind, bp_decode_generic_kind_with_scratch, bp_llr_zsum,
    bp_llr_zsum_with_scratch,
};
use crate::fec::ldpc::osd::{ldpc_encode_generic, osd_decode_generic};
use crate::fec::ldpc::params::Ldpc240_74Params;

pub const LDPC_N: usize = 240;
pub const LDPC_K: usize = 74;
pub const LDPC_M: usize = LDPC_N - LDPC_K; // 166

// ────────────────────────────────────────────────────────────────────
// CRC-24

/// CRC-24Q as used by WSJT-X FST4 and FST4W: polynomial 0x100065B,
/// applied bit-serially over the message padded with 24 zeros.
///
/// Identical to [`super::ldpc240_101::crc24`] — the CRC is the same
/// code; FST4W simply computes it over 50 message bits rather than 77.
/// Kept as its own copy so this module is self-contained; the two are
/// byte-for-byte the same routine.
///
/// Matches the `get_crc24` subroutine in WSJT-X `lib/fst4/get_crc24.f90`.
pub fn crc24(bits: &[u8]) -> u32 {
    let mut r = [0u8; 25];
    for (i, slot) in r.iter_mut().enumerate() {
        *slot = if i < bits.len() { bits[i] & 1 } else { 0 };
    }
    const POLY: [u8; 25] = [
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1,
    ];
    let n = bits.len().saturating_sub(25);
    for i in 0..=n {
        if i + 25 <= bits.len() {
            r[24] = bits[i + 24] & 1;
        } else {
            r[24] = 0;
        }
        let top = r[0];
        if top != 0 {
            for (rv, pv) in r.iter_mut().zip(POLY.iter()) {
                *rv ^= *pv;
            }
        }
        let first = r[0];
        for k in 0..24 {
            r[k] = r[k + 1];
        }
        r[24] = first;
    }
    let mut v = 0u32;
    for &b in &r[..24] {
        v = (v << 1) | (b as u32);
    }
    v
}

/// Verify CRC-24 for a 74-bit decoded word (50 message + 24 CRC).
///
/// The FST4W sibling of [`super::ldpc240_101::check_crc24`]: same CRC,
/// message length 50 rather than 77. Suitable as a
/// `MessageCodec::verify_info` implementation passed through
/// `FecOpts::verify_info`.
pub fn check_crc24(decoded: &[u8]) -> bool {
    if decoded.len() != LDPC_K {
        return false;
    }
    let mut with_zero = [0u8; LDPC_K];
    with_zero[..50].copy_from_slice(&decoded[..50]);
    let expected = crc24(&with_zero);

    let mut got = 0u32;
    for &b in &decoded[50..74] {
        got = (got << 1) | (b as u32 & 1);
    }
    expected == got
}

// ────────────────────────────────────────────────────────────────────
// FecCodec impl

/// Zero-sized LDPC(240, 74) codec — a thin wrapper that pins the
/// generic [`crate::fec::ldpc::params::LdpcParams`]-based
/// implementation to [`Ldpc240_74Params`].
#[derive(Copy, Clone, Debug, Default)]
pub struct Ldpc240_74;

impl crate::engine::protocol::sealed::Sealed for Ldpc240_74 {}
impl FecCodec for Ldpc240_74 {
    const N: usize = LDPC_N;
    const K: usize = LDPC_K;

    fn encode(&self, info: &[u8], codeword: &mut [u8]) {
        assert_eq!(info.len(), LDPC_K, "info must be {} bits", LDPC_K);
        assert_eq!(codeword.len(), LDPC_N, "codeword must be {} bits", LDPC_N);
        ldpc_encode_generic::<Ldpc240_74Params>(info, codeword);
    }

    fn decode_soft(&self, llr: &[f32], opts: &FecOpts<'_>) -> Option<FecResult> {
        let llr_arr = prepare_llr(llr);

        if let Some(r) = bp_decode_generic_kind::<Ldpc240_74Params>(
            &llr_arr,
            None,
            opts.bp_max_iter,
            opts.verify_info,
            opts.bp_kind,
        ) {
            return Some(FecResult {
                info: r.info,
                hard_errors: r.hard_errors,
                iterations: r.iterations,
            });
        }

        if opts.osd_depth == 0 {
            return None;
        }

        if let Some(r) = osd_decode_generic::<Ldpc240_74Params>(
            &llr_arr,
            opts.osd_depth.min(4) as u8,
            LDPC_K,
            opts.verify_info,
            false,
        ) {
            return Some(FecResult {
                info: r.info,
                hard_errors: r.hard_errors,
                iterations: 0,
            });
        }

        // zsum-seeded OSD retry, as the 101 code does (issue #146): WSJT-X
        // feeds OSD the running sum of BP's soft estimate rather than the
        // raw channel LLR. FST4W does no AP decoding, so there is no mask
        // to skip for.
        let zsum = bp_llr_zsum::<Ldpc240_74Params>(&llr_arr, 2);
        if let Some(r) =
            osd_decode_generic::<Ldpc240_74Params>(&zsum, opts.osd_depth.min(4) as u8, LDPC_K, opts.verify_info, false)
        {
            return Some(FecResult {
                info: r.info,
                hard_errors: r.hard_errors,
                iterations: 0,
            });
        }

        None
    }
}

impl BpPooledFec for Ldpc240_74 {
    type Scratch = BpScratch<Ldpc240_74Params, f32>;

    fn decode_soft_pooled(
        &self,
        llr: &[f32],
        opts: &FecOpts<'_>,
        scratch: &mut Self::Scratch,
    ) -> Option<FecResult> {
        let llr_arr = prepare_llr(llr);

        if let Some(r) = bp_decode_generic_kind_with_scratch::<Ldpc240_74Params>(
            scratch,
            &llr_arr,
            None,
            opts.bp_max_iter,
            opts.verify_info,
            opts.bp_kind,
        ) {
            return Some(FecResult {
                info: r.info,
                hard_errors: r.hard_errors,
                iterations: r.iterations,
            });
        }

        if opts.osd_depth == 0 {
            return None;
        }

        if let Some(r) = osd_decode_generic::<Ldpc240_74Params>(
            &llr_arr,
            opts.osd_depth.min(4) as u8,
            LDPC_K,
            opts.verify_info,
            false,
        ) {
            return Some(FecResult {
                info: r.info,
                hard_errors: r.hard_errors,
                iterations: 0,
            });
        }

        let zsum = bp_llr_zsum_with_scratch::<Ldpc240_74Params>(scratch, &llr_arr, 2);
        if let Some(r) =
            osd_decode_generic::<Ldpc240_74Params>(zsum, opts.osd_depth.min(4) as u8, LDPC_K, opts.verify_info, false)
        {
            return Some(FecResult {
                info: r.info,
                hard_errors: r.hard_errors,
                iterations: 0,
            });
        }

        None
    }
}

/// Copy the incoming LLR slice into the fixed 240-array the generic
/// bodies expect. FST4W does no a-priori decoding, so unlike the 101
/// code's `prepare_ap_llr` there is no mask to apply.
fn prepare_llr(llr: &[f32]) -> [f32; LDPC_N] {
    assert_eq!(llr.len(), LDPC_N, "llr must be {} values", LDPC_N);
    let mut a = [0f32; LDPC_N];
    a.copy_from_slice(llr);
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fec::ldpc::bp::bp_decode_generic;
    use crate::fec::ldpc::osd::ldpc_encode_generic;

    /// Round-trip: encode a 74-bit info word (50 message bits + 24 CRC),
    /// feed perfect LLRs, and recover it. Exercises the generator
    /// sub-matrix and the BP path through the generic helpers.
    #[test]
    fn roundtrip_perfect_llr() {
        let mut info = [0u8; LDPC_K];
        for i in 0..50 {
            info[i] = ((i * 5 + 1) & 1) as u8;
        }
        let crc = crc24(&info);
        for i in 0..24 {
            info[50 + i] = ((crc >> (23 - i)) & 1) as u8;
        }

        let mut cw = [0u8; LDPC_N];
        ldpc_encode_generic::<Ldpc240_74Params>(&info, &mut cw);
        assert_eq!(&cw[..LDPC_K], &info[..], "systematic encode keeps info");

        let mut llr = vec![0f32; LDPC_N];
        for i in 0..LDPC_N {
            llr[i] = if cw[i] == 1 { 8.0 } else { -8.0 };
        }
        let r = bp_decode_generic::<Ldpc240_74Params>(&llr, None, 30, Some(check_crc24))
            .expect("BP converges on perfect LLR");
        assert_eq!(&r.info[..50], &info[..50]);
        assert!(check_crc24(&r.info));
    }
}
