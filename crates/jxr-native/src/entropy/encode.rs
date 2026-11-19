//! Entropy writing: the inverse of each parsing primitive in this module.
//!
//! Every writer drives the same adaptive state as its parsing twin, in the
//! same order, so a decoder reading the result reaches identical state.

mod coefficients;
mod refinement;

pub(crate) use coefficients::{
    dc_has_abs_level, encode_dc_coefficient, encode_hp_block, encode_lp_block, encode_ordered_block,
};
pub(crate) use refinement::{
    check_magnitudes, encode_flex_block, encode_lp_refinement, encode_lp_refinement_at, vlc_part,
};

#[cfg(test)]
pub(crate) mod test_support {
    /// Deterministic generator for round-trip tests; no external dependency.
    pub(crate) struct Lcg(pub(crate) u64);

    impl Lcg {
        /// The state after `state`: the step every test generator of the
        /// crate takes, whichever bits of it each one then uses.
        pub(crate) const fn advance(state: u64) -> u64 {
            state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407)
        }

        pub(crate) fn next(&mut self) -> u64 {
            self.0 = Self::advance(self.0);
            self.0 >> 24
        }

        pub(crate) fn below(&mut self, bound: u64) -> u64 {
            self.next() % bound
        }

        /// A coefficient whose magnitude class is itself random, so small
        /// values, escapes, and wide refinements all occur.
        pub(crate) fn coefficient(&mut self, max_bits: u32) -> i32 {
            let bits = self.below(u64::from(max_bits) + 1);
            let magnitude = if bits == 0 { 0 } else { self.below(1 << bits) };
            let magnitude = i32::try_from(magnitude).unwrap();
            if self.below(2) == 0 {
                magnitude
            } else {
                -magnitude
            }
        }
    }
}
