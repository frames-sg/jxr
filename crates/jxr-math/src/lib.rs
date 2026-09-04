//! Exact, allocation-free integer building blocks used by JPEG XR decode paths.
//!
//! This crate is the scalar arithmetic oracle for the workspace. The functions
//! here define overflow and rounding behavior, but do not by themselves claim
//! to implement every transform or table in ITU-T T.832.

#![no_std]
#![forbid(unsafe_code)]

pub mod alpha;
pub mod arithmetic;
pub mod color;
pub mod overlap;
pub mod prediction;
pub mod quantization;
pub mod rgbe;
pub mod sampling;
pub mod tables;
pub mod transform;

pub use arithmetic::MathError;
