//! Output-formatting failures.

use core::fmt;

/// Failure while converting reconstructed samples to their declared output representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputFormatError {
    /// The requested input/output colour, depth, or storage combination is not implemented.
    UnsupportedCombination {
        /// Stable description of the unsupported combination.
        combination: &'static str,
    },
    /// The number of supplied reconstructed planes does not match the internal colour format.
    ComponentCount {
        /// Required number of primary component planes.
        expected: usize,
        /// Supplied number of primary component planes.
        actual: usize,
    },
    /// A plane's dimensions, stride, or backing slice are invalid.
    InvalidPlane {
        /// Index of the invalid primary plane, or `None` for the alpha plane.
        component: Option<usize>,
        /// Stable description of the violated plane invariant.
        reason: &'static str,
    },
    /// The requested crop extends outside a required plane.
    CropOutsidePlane {
        /// Index of the primary plane, or `None` for the alpha plane.
        component: Option<usize>,
    },
    /// A layout requiring alpha was requested without an alpha plane, or vice versa.
    AlphaMismatch,
    /// A checked sample or allocation calculation overflowed.
    ArithmeticOverflow {
        /// Stable description of the operation that overflowed.
        operation: &'static str,
    },
    /// A reconstructed sample cannot be represented by the declared floating-point syntax.
    InvalidFloatingPointSample,
}

impl OutputFormatError {
    pub(crate) const fn arithmetic(operation: &'static str) -> Self {
        Self::ArithmeticOverflow { operation }
    }
}

impl fmt::Display for OutputFormatError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedCombination { combination } => {
                write!(
                    formatter,
                    "unsupported JPEG XR output combination: {combination}"
                )
            }
            Self::ComponentCount { expected, actual } => {
                write!(
                    formatter,
                    "expected {expected} component planes, received {actual}"
                )
            }
            Self::InvalidPlane { component, reason } => match component {
                Some(component) => {
                    write!(formatter, "invalid component plane {component}: {reason}")
                }
                None => write!(formatter, "invalid alpha plane: {reason}"),
            },
            Self::CropOutsidePlane { component } => match component {
                Some(component) => write!(
                    formatter,
                    "crop extends outside component plane {component}"
                ),
                None => formatter.write_str("crop extends outside alpha plane"),
            },
            Self::AlphaMismatch => {
                formatter.write_str("output channel layout does not match alpha input")
            }
            Self::ArithmeticOverflow { operation } => {
                write!(formatter, "overflow while {operation}")
            }
            Self::InvalidFloatingPointSample => {
                formatter.write_str("sample is outside the declared floating-point representation")
            }
        }
    }
}

impl std::error::Error for OutputFormatError {}
