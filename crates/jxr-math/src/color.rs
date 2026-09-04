//! Exact color-transform arithmetic.

/// Clamp a reconstructed sample to an inclusive range.
#[must_use]
pub const fn clamp_sample(sample: i32, minimum: i32, maximum: i32) -> i32 {
    if sample < minimum {
        minimum
    } else if sample > maximum {
        maximum
    } else {
        sample
    }
}
