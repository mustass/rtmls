use eframe::egui;
use eframe::run_native;

mod app;

use app::HotNotDogApp;

fn main() {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([600.0, 800.0]),
        ..Default::default()
    };
    let _ = run_native(
        "HotNotDog",
        native_options,
        Box::new(|cc| {
            // Add the egui_extras crate as a dependency
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Box::new(HotNotDogApp::new(cc))
        }),
    ); // Added closing parenthesis here
}
