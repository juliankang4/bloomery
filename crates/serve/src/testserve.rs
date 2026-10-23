//! The in-crate tests' server: a mock engine behind a real socket, and one
//! HTTP round trip to it. Tests of `api` and of the modules converted onto it
//! start their servers here, so a field `ServerConfig` gains is set in one
//! place ([`mock_config`]).

use std::cell::RefCell;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::{Arc, mpsc};

use super::{End, Engine, FATAL_LINGER, Server, ServerConfig, State};
use crate::flag::ApiKeys;

thread_local! {
    pub(super) static CLIENT_PROBE: RefCell<Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>> = const {
        RefCell::new(None)
    };
}

/// The mock's chat template: each message as `<role>\n` and its content,
/// then `<assistant>\n` when a generation prompt is asked for.
pub(super) const TEMPLATE: &str = concat!(
    "{%- for message in messages %}",
    "{{- '<' + message.role + '>\\n' + message.content }}",
    "{%- endfor %}",
    "{%- if add_generation_prompt %}{{- '<assistant>\\n' }}{%- endif %}",
);

/// The configuration every in-crate test server starts from.
pub(super) fn mock_config() -> ServerConfig {
    ServerConfig {
        model_alias: "mock".to_owned(),
        model_path: "mock.gguf".to_owned(),
        chat_template: TEMPLATE.to_owned(),
        chat_preparation: None,
        sampler: None,
        fatal_linger: FATAL_LINGER,
        slot_save_path: None,
        api_keys: ApiKeys::default(),
    }
}

/// A mock server whose key check is on, `keys` set: the server every key
/// test asks a key of.
pub(super) fn spawn_keyed(keys: ApiKeys) -> (SocketAddr, Arc<State>, mpsc::Receiver<End>) {
    let mut config = mock_config();
    config.api_keys = keys;
    spawn(Box::new(crate::MockEngine::new(64)), config)
}

/// `engine` served on a free loopback port under `config`: the address, the
/// server's state and its end channel.
pub(super) fn spawn(
    engine: Box<dyn Engine>,
    config: ServerConfig,
) -> (SocketAddr, Arc<State>, mpsc::Receiver<End>) {
    let server = Server::bind("127.0.0.1:0", engine, config).expect("bind");
    let addr = server.local_addr().expect("addr");
    let (state, ended) = server.start().expect("start");
    (addr, state, ended)
}

/// One HTTP/1.0 request with `headers` past the JSON content type, read to
/// the connection's end: the status and the body (a stream's whole body).
pub(super) fn roundtrip(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &str,
) -> (u16, String) {
    let (status, _head, body) = roundtrip_head(addr, method, path, headers, body);
    (status, body)
}

/// One HTTP/1.0 request with `headers` past the JSON content type, written
/// over a fresh connection that is returned with its answer unread: the form
/// a test that closes the connection, or reads it a piece at a time, needs.
pub(super) fn send_request(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &str,
) -> TcpStream {
    let mut s = TcpStream::connect(addr).expect("connect");
    let extra: String = headers
        .iter()
        .map(|(k, v)| format!("{k}: {v}\r\n"))
        .collect();
    let req = format!(
        "{method} {path} HTTP/1.0\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
         {extra}Content-Length: {}\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).expect("write");
    s
}

/// [`roundtrip`] with the whole reply head beside the body: the status,
/// every header line as sent, and the body — the form a test that pins a
/// response's headers needs.
pub(super) fn roundtrip_head(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &str,
) -> (u16, String, String) {
    let mut s = send_request(addr, method, path, headers, body);
    let mut raw = String::new();
    s.read_to_string(&mut raw).expect("read");
    let (head, body) = raw.split_once("\r\n\r\n").expect("a head");
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .expect("a status line");
    (status, head.to_owned(), body.to_owned())
}
