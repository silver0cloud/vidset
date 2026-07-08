// main.rs
// -------
// Tauri entry point. Spawns the Python FastAPI backend as a sidecar
// process on a free local port, waits for it to become healthy, then
// emits the chosen port to the frontend so api/client.ts can connect.

use std::net::TcpListener;
use std::time::Duration;

use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_shell::process::CommandEvent;
use tauri_plugin_shell::ShellExt;

/// Finds a free TCP port on localhost by binding to port 0 and reading
/// back the OS-assigned port, then immediately releasing it.
fn find_free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("failed to bind to find a free port")
        .local_addr()
        .expect("failed to read local addr")
        .port()
}

/// Polls the backend's /api/health endpoint until it responds or times out.
async fn wait_for_backend(port: u16) -> bool {
    let url = format!("http://127.0.0.1:{port}/api/health");
    let client = reqwest::Client::new();

    for _ in 0..60 {
        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                return true;
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    false
}

fn spawn_backend(app: &AppHandle, port: u16) {
    let shell = app.shell();

    let port_arg = port.to_string();
    let (mut rx, _child) = shell
        .sidecar("tts-backend")
        .expect("failed to create tts-backend sidecar command")
        .args([port_arg.as_str()])
        .spawn()
        .expect("failed to spawn tts-backend sidecar");

    // Forward backend stdout/stderr into the app log so issues are visible
    // in `tauri dev` console output during development.
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(line) => {
                    print!("[backend] {}", String::from_utf8_lossy(&line));
                }
                CommandEvent::Stderr(line) => {
                    eprint!("[backend:err] {}", String::from_utf8_lossy(&line));
                }
                CommandEvent::Terminated(payload) => {
                    eprintln!("[backend] exited with code {:?}", payload.code);
                }
                _ => {}
            }
        }
    });
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let port = find_free_port();
            let handle = app.handle().clone();

            spawn_backend(&handle, port);

            // Wait for the backend to come up, then tell the frontend
            // which port to talk to via a Tauri event.
            tauri::async_runtime::spawn(async move {
                let ready = wait_for_backend(port).await;
                if ready {
                    handle
                        .emit("backend-ready", port)
                        .expect("failed to emit backend-ready event");
                } else {
                    handle
                        .emit("backend-failed", ())
                        .expect("failed to emit backend-failed event");
                }
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Open TTS Studio");
}
