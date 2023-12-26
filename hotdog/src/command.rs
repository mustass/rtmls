use winit::dpi::PhysicalSize;
use crate::control_panel::TrueLabel;
pub enum Command {
    Resize { new_size: PhysicalSize<u32> },
    TrainMode { value: bool },
    Evaluate {value: bool },
    SubmitLabel { value: bool },
    TrueLabel { value: TrueLabel },
    KeyEventW,
    KeyEventA,
    KeyEventS,
    KeyEventD,
    KeyEventQ,
    KeyEventE,
    KeyEventComma,
    KeyEventPeriod,
    Shutdown { value: bool },
}
