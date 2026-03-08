use std::{
    env, fs,
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Result};
use hf_hub::api::sync::{ApiError, ApiRepo};
use tracing::{info, warn};

use super::FileListCache;

#[derive(Clone, Debug)]
pub(crate) struct RemoteAccessIssue {
    pub status_code: Option<u16>,
    pub message: String,
}

/// Resolve the Hugging Face home directory.
///
/// Precedence:
/// 1. HF_HOME
/// 2. ~/.cache/huggingface
pub fn hf_home_dir() -> Option<PathBuf> {
    let dir = env::var("HF_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".cache").join("huggingface")));

    if let Some(ref dir) = dir {
        if let Err(err) = fs::create_dir_all(dir) {
            warn!(
                "Could not create Hugging Face home directory `{}`: {err}",
                dir.display()
            );
        }
    }

    dir
}

/// Resolve the Hugging Face Hub cache directory.
///
/// Precedence:
/// 1. HF_HUB_CACHE
/// 2. HF_HOME/hub
/// 3. ~/.cache/huggingface/hub
pub fn hf_hub_cache_dir() -> Option<PathBuf> {
    let dir = env::var("HF_HUB_CACHE")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_home_dir().map(|home| home.join("hub")));

    if let Some(ref dir) = dir {
        if let Err(err) = fs::create_dir_all(dir) {
            warn!(
                "Could not create Hugging Face hub cache directory `{}`: {err}",
                dir.display()
            );
        }
    }

    dir
}

/// Resolve the Hugging Face token file path.
pub fn hf_token_path() -> Option<PathBuf> {
    hf_home_dir().map(|home| home.join("token"))
}

fn cache_dir() -> PathBuf {
    hf_hub_cache_dir().unwrap_or_else(|| PathBuf::from("./"))
}

fn cache_file_for_model(model_id: &Path) -> PathBuf {
    let sanitized_id = model_id.display().to_string().replace('/', "-");
    cache_dir().join(format!("{sanitized_id}_repo_list.json"))
}

fn read_cached_repo_files(cache_file: &Path) -> Option<Vec<String>> {
    if !cache_file.exists() {
        return None;
    }

    let mut file = match fs::File::open(cache_file) {
        Ok(file) => file,
        Err(err) => {
            warn!(
                "Could not open Hugging Face repo cache file `{}`: {err}",
                cache_file.display()
            );
            return None;
        }
    };

    let mut contents = String::new();
    if let Err(err) = file.read_to_string(&mut contents) {
        warn!(
            "Could not read Hugging Face repo cache file `{}`: {err}",
            cache_file.display()
        );
        return None;
    }

    match serde_json::from_str::<FileListCache>(&contents) {
        Ok(cache) => {
            info!("Read from cache file `{}`", cache_file.display());
            Some(cache.files)
        }
        Err(err) => {
            warn!(
                "Could not parse Hugging Face repo cache file `{}`: {err}",
                cache_file.display()
            );
            None
        }
    }
}

fn write_cached_repo_files(cache_file: &Path, files: &[String]) {
    let cache = FileListCache {
        files: files.to_vec(),
    };
    match serde_json::to_string_pretty(&cache) {
        Ok(json) => {
            if let Err(err) = fs::write(cache_file, json) {
                warn!(
                    "Could not write Hugging Face repo cache file `{}`: {err}",
                    cache_file.display()
                );
            } else {
                info!("Write to cache file `{}`", cache_file.display());
            }
        }
        Err(err) => warn!(
            "Could not serialize Hugging Face repo cache for `{}`: {err}",
            cache_file.display()
        ),
    }
}

pub(crate) fn parse_status_code(message: &str) -> Option<u16> {
    let marker = "status code ";
    let (_, tail) = message.split_once(marker)?;
    let digits = tail
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>();
    digits.parse().ok()
}

pub(crate) fn api_error_status_code(err: &ApiError) -> Option<u16> {
    match err {
        ApiError::TooManyRetries(inner) => api_error_status_code(inner),
        _ => parse_status_code(&err.to_string()),
    }
}

pub(crate) fn should_propagate_api_error(err: &ApiError) -> bool {
    matches!(api_error_status_code(err), Some(401 | 403 | 404))
}

pub(crate) fn remote_issue_from_api_error(
    model_id: &Path,
    file: Option<&str>,
    err: &ApiError,
) -> RemoteAccessIssue {
    let target = match file {
        Some(file) => format!("`{file}` for `{}`", model_id.display()),
        None => format!("`{}`", model_id.display()),
    };
    RemoteAccessIssue {
        status_code: api_error_status_code(err),
        message: format!("Failed to access {target}: {err}"),
    }
}

pub(crate) fn hf_access_error(model_id: &Path, issue: &RemoteAccessIssue) -> anyhow::Error {
    match issue.status_code {
        Some(code @ (401 | 403)) => anyhow!(
            "Could not access `{}` on Hugging Face (HTTP {code}). You may need to run `mistralrs login` or set HF_TOKEN.",
            model_id.display()
        ),
        Some(404) => anyhow!(
            "Model `{}` was not found or is not accessible on Hugging Face (HTTP 404). Check the model ID and your access token.",
            model_id.display()
        ),
        Some(code) => anyhow!(
            "Failed to access `{}` on Hugging Face (HTTP {code}): {}",
            model_id.display(),
            issue.message
        ),
        None => anyhow!(
            "Failed to access `{}` on Hugging Face: {}",
            model_id.display(),
            issue.message
        ),
    }
}

pub(crate) fn hf_api_error(model_id: &Path, file: Option<&str>, err: &ApiError) -> anyhow::Error {
    let status_code = api_error_status_code(err);
    let file_context = file
        .map(|f| format!(" while fetching `{f}`"))
        .unwrap_or_default();
    match status_code {
        Some(code @ (401 | 403)) => anyhow!(
            "Could not access `{}` on Hugging Face (HTTP {code}){file_context}. You may need to run `mistralrs login` or set HF_TOKEN.",
            model_id.display()
        ),
        Some(404) => anyhow!(
            "Model `{}` was not found or is not accessible on Hugging Face (HTTP 404){file_context}. Check the model ID and your access token.",
            model_id.display()
        ),
        Some(code) => anyhow!(
            "Failed to access `{}` on Hugging Face (HTTP {code}){file_context}: {err}",
            model_id.display()
        ),
        None => anyhow!(
            "Failed to access `{}` on Hugging Face{file_context}: {err}",
            model_id.display()
        ),
    }
}

pub(crate) fn local_file_missing_error(model_id: &Path, file: &str) -> anyhow::Error {
    anyhow!(
        "File `{file}` was not found at local model path `{}`.",
        model_id.display()
    )
}

pub(crate) fn list_repo_files(
    api: &ApiRepo,
    model_id: &Path,
    should_error: bool,
) -> Result<Vec<String>> {
    if model_id.exists() {
        let listing = fs::read_dir(model_id).map_err(|err| {
            anyhow!(
                "Cannot list local model directory `{}`: {err}",
                model_id.display()
            )
        })?;
        let files = listing
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                entry
                    .path()
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(std::string::ToString::to_string)
            })
            .collect::<Vec<_>>();
        return Ok(files);
    }

    let cache_file = cache_file_for_model(model_id);
    if let Some(files) = read_cached_repo_files(&cache_file) {
        return Ok(files);
    }

    match api.info() {
        Ok(repo) => {
            let files = repo
                .siblings
                .iter()
                .map(|x| x.rfilename.clone())
                .collect::<Vec<_>>();
            write_cached_repo_files(&cache_file, &files);
            Ok(files)
        }
        Err(err) => {
            if should_error || should_propagate_api_error(&err) {
                Err(hf_api_error(model_id, None, &err))
            } else {
                warn!(
                    "Could not get directory listing from Hugging Face for `{}`: {err}",
                    model_id.display()
                );
                Ok(Vec::new())
            }
        }
    }
}

// ── Parallel (multi-connection) downloader ────────────────────────────────────
//
// HuggingFace serves model files through a CDN (Cloudflare / S3) that caps
// each TCP connection at roughly 20-30 MB/s.  By opening N simultaneous
// connections with non-overlapping `Range:` headers we can saturate the
// available bandwidth.
//
// Only weight files (`.safetensors`, `.bin`, `.gguf`, `.pt`, `.pth`) that are
// larger than PARALLEL_THRESHOLD are handled this way.  Small metadata files
// continue to go through hf-hub's built-in downloader unchanged.
//
// Cache layout used (matches Python huggingface_hub, so the files are
// interchangeable):
//   <hub>/models--{org}--{name}/blobs/<etag>          — final file
//   <hub>/models--{org}--{name}/snapshots/<commit>/<filename>  — symlink
//   <hub>/models--{org}--{name}/refs/main             — commit hash

/// Minimum file size to use multi-connection download (64 MB).
const PARALLEL_THRESHOLD: u64 = 64 * 1024 * 1024;

/// Maximum simultaneous TCP connections per file.
const MAX_CONNS: usize = 8;

/// Minimum bytes per connection.  We don't spawn more connections than this
/// allows.
const MIN_CHUNK: u64 = 32 * 1024 * 1024; // 32 MB

fn conn_count(size: u64) -> usize {
    ((size / MIN_CHUNK) as usize).max(2).min(MAX_CONNS)
}

/// Metadata returned by the HF LFS proxy for a single file.
struct HfMeta {
    /// The blob's content hash (used as the cache filename).
    etag: String,
    /// The repository commit hash (used for the snapshot directory).
    commit_hash: String,
    /// Total file size in bytes.
    size: u64,
}

/// Fetch `etag`, `x-repo-commit`, and `Content-Length` for `url` without
/// downloading the file body.
///
/// Uses a `Range: bytes=0-0` trick (same as hf-hub's `metadata()`) to get
/// the `Content-Range: bytes 0-0/<total>` response that carries the total
/// size.  A non-redirecting first request captures the HF-server-level headers
/// (`x-linked-etag`, `x-repo-commit`); a second request (to the CDN redirect
/// target if needed) captures `Content-Range`.
///
/// Returns `Err` with a message containing `"xet backend"` when the file is
/// stored on HuggingFace's XetHub storage (`cas-bridge.xethub.hf.co`).  The
/// Xet protocol is content-addressable and not amenable to plain HTTP range
/// requests, so the caller falls back to `api.get()` which handles Xet via
/// hf-hub's built-in downloader.
fn fetch_meta(url: &str, token: Option<&str>) -> Result<HfMeta> {
    let no_redir = ureq::AgentBuilder::new().redirects(0).build();

    let hf_get = |agent: &ureq::Agent, target: &str| {
        let mut req = agent.get(target).set("Range", "bytes=0-0");
        if let Some(t) = token {
            req = req.set("Authorization", &format!("Bearer {t}"));
        }
        req.call()
    };

    let r1 = hf_get(&no_redir, url)
        .map_err(|e| anyhow!("metadata request for {url}: {e}"))?;

    // Extract HF-server headers (only present before the external redirect).
    let etag = r1
        .header("x-linked-etag")
        .or_else(|| r1.header("etag"))
        .ok_or_else(|| anyhow!("no etag in response for {url}"))?
        .trim_matches('"')
        .to_owned();

    let commit_hash = r1
        .header("x-repo-commit")
        .ok_or_else(|| anyhow!("no x-repo-commit in response for {url}"))?
        .to_owned();

    // Content-Range gives us the total size.  When HF redirects to an
    // external CDN (absolute Location), the first response won't have
    // Content-Range; inspect the Location before following it.
    let size = if let Some(cr) = r1.header("Content-Range") {
        parse_content_range(cr)?
    } else if r1.status() / 100 == 3 {
        let loc = r1
            .header("Location")
            .ok_or_else(|| anyhow!("3xx with no Location for {url}"))?;

        // ── Xet backend detection ──────────────────────────────────────────
        // HuggingFace's XetHub storage (`cas-bridge.xethub.hf.co`) uses a
        // content-addressable protocol that does NOT support plain HTTP Range
        // requests.  Attempting one fails with a DNS error or a 403.  Detect
        // the Xet host from the redirect Location BEFORE making a second
        // request, and bail cleanly so the caller can use `api.get()` which
        // handles Xet through hf-hub's own implementation.
        if loc.contains("xethub.hf.co") || loc.contains("cas-bridge") {
            anyhow::bail!("xet backend — file is on HuggingFace XetHub storage \
                           (parallel range requests not supported; \
                           using single-connection fallback)");
        }

        let r2 = hf_get(&ureq::agent(), loc)
            .map_err(|e| anyhow!("CDN follow for {url}: {e}"))?;
        let cr = r2
            .header("Content-Range")
            .ok_or_else(|| anyhow!("no Content-Range from CDN for {url}"))?;
        parse_content_range(cr)?
    } else {
        // Fallback: Content-Length from r1 (non-redirect responses).
        r1.header("Content-Length")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| anyhow!("cannot determine size for {url}"))?
    };

    Ok(HfMeta { etag, commit_hash, size })
}

fn parse_content_range(cr: &str) -> Result<u64> {
    // "bytes 0-0/12345678"  or  "bytes */12345678"
    cr.rsplit('/')
        .next()
        .and_then(|s| s.trim().parse().ok())
        .ok_or_else(|| anyhow!("unparseable Content-Range: {cr}"))
}

/// Download the byte range `[start, end]` (inclusive) from `url` into
/// `chunk_path` (a regular file created or truncated on each call).
///
/// Using a dedicated chunk file per connection (rather than `write_at` into a
/// shared blob) has two advantages:
///
/// 1. **Accurate progress**: the progress monitor (`progress.rs`) measures
///    bytes on disk by summing file sizes.  With `write_at` into a
///    pre-allocated sparse blob the monitor would see 100% immediately.
///    With chunk files it sees a smooth increase from 0 → model_size.
///
/// 2. **Crash recovery**: surviving chunk files from a prior interrupted run
///    can be purged individually.
fn download_chunk_to_file(
    url: &str,
    start: u64,
    end: u64,
    chunk_path: &Path,
    token: Option<&str>,
) -> Result<()> {
    use std::io::{Read as _, Write as _};

    let mut req = ureq::get(url).set("Range", &format!("bytes={start}-{end}"));
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let resp = req
        .call()
        .map_err(|e| anyhow!("chunk {start}-{end}: {e}"))?;

    let mut reader = resp.into_reader();
    let mut file  = std::fs::File::create(chunk_path)
        .map_err(|e| anyhow!("create chunk file {}: {e}", chunk_path.display()))?;
    let mut buf   = vec![0u8; 1024 * 1024]; // 1 MB write buffer

    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| anyhow!("read chunk {start}-{end}: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| anyhow!("write chunk {start}-{end}: {e}"))?;
    }

    Ok(())
}

/// Register a completed blob in the hf-hub on-disk cache so that subsequent
/// calls to `api.get(filename)` return it from cache without re-downloading.
///
/// Creates:
///   • `snapshots/<commit>/<filename>` → `../../blobs/<etag>`  (symlink)
///   • `refs/main`  = commit_hash                               (ref file)
fn register_in_cache(
    cache_root: &Path,
    folder: &str,
    etag: &str,
    commit_hash: &str,
    filename: &str,
) -> Result<()> {
    let model_dir   = cache_root.join(folder);
    let snap_dir    = model_dir.join("snapshots").join(commit_hash);
    let pointer     = snap_dir.join(filename);

    // Pointer symlink
    if !pointer.exists() {
        fs::create_dir_all(&snap_dir)?;
        // Relative path from pointer → blob: ../../blobs/<etag>
        let rel = PathBuf::from("../..").join("blobs").join(etag);
        #[cfg(unix)]
        std::os::unix::fs::symlink(&rel, &pointer)?;
        #[cfg(windows)]
        {
            let blob_path = model_dir.join("blobs").join(etag);
            std::os::windows::fs::symlink_file(&blob_path, &pointer)?;
        }
    }

    // refs/main  (revision = "main" is the default; update only if file exists
    // so we don't clobber a custom revision checkout)
    let refs_dir  = model_dir.join("refs");
    let refs_main = refs_dir.join("main");
    if !refs_main.exists() {
        fs::create_dir_all(&refs_dir)?;
        fs::write(&refs_main, commit_hash)?;
    }

    info!(
        "Registered `{filename}` in hf cache  (etag={etag}, commit={commit_hash})"
    );
    Ok(())
}

/// Try to download `file` belonging to `model_id` using N parallel TCP
/// connections, then register the result in the hf-hub on-disk cache.
///
/// Returns the pointer path (same shape as what `api.get(filename)` returns)
/// or an error.  On any error the caller falls back to `api.get()`.
fn parallel_download(api: &ApiRepo, model_id: &Path, file: &str) -> Result<PathBuf> {
    use crate::GLOBAL_HF_CACHE;

    let token   = read_hf_token();
    let tok_ref = token.as_deref();
    let url     = api.url(file);

    // ── Metadata ─────────────────────────────────────────────────────────────
    let meta = fetch_meta(&url, tok_ref)?;

    // ── Cache layout ─────────────────────────────────────────────────────────
    let folder = format!(
        "models--{}",
        model_id.display().to_string().replace('/', "--")
    );
    let cache      = GLOBAL_HF_CACHE.get().cloned().unwrap_or_default();
    let blobs_dir  = cache.path().join(&folder).join("blobs");
    let blob_path  = blobs_dir.join(&meta.etag);
    let pointer    = cache.path()
        .join(&folder).join("snapshots")
        .join(&meta.commit_hash).join(file);

    // ── Already on disk? ─────────────────────────────────────────────────────
    if blob_path.exists() && blob_path.metadata().map_or(false, |m| m.len() == meta.size) {
        register_in_cache(cache.path(), &folder, &meta.etag, &meta.commit_hash, file)
            .unwrap_or_else(|e| warn!("cache registration failed: {e}"));
        if pointer.exists() {
            return Ok(pointer);
        }
    }

    // ── Small files: let hf-hub handle them ──────────────────────────────────
    if meta.size < PARALLEL_THRESHOLD {
        // Returning an error here causes the caller to fall back to api.get().
        anyhow::bail!("file too small for parallel download ({} B)", meta.size);
    }

    // ── Chunk map ────────────────────────────────────────────────────────────
    fs::create_dir_all(&blobs_dir)?;
    let n          = conn_count(meta.size);
    let chunk_size = (meta.size + n as u64 - 1) / n as u64;
    // chunk_path[i] = blobs/<etag>.chunk.<i>
    // The `.chunk.<i>` suffix is recognised by `scan_blobs` as an in-flight
    // download, so the progress monitor shows correct bytes and a live count
    // of active connections.
    let chunk_paths: Vec<PathBuf> = (0..n)
        .map(|i| blobs_dir.join(format!("{}.chunk.{i}", meta.etag)))
        .collect();
    let chunks: Vec<(u64, u64, &Path)> = (0..n)
        .map(|i| {
            let start = i as u64 * chunk_size;
            let end   = (start + chunk_size - 1).min(meta.size - 1);
            (start, end, chunk_paths[i].as_path())
        })
        .collect();

    // Delete any stale chunk files from a previous interrupted run.
    for cp in &chunk_paths {
        if cp.exists() {
            let _ = fs::remove_file(cp);
        }
    }

    info!(
        "Downloading `{file}` ({:.1} GB) with {} parallel connections",
        meta.size as f64 / 1_073_741_824.0,
        n
    );

    // ── Parallel download ────────────────────────────────────────────────────
    // Each chunk is saved to its own `.chunk.<i>` file.  The progress monitor
    // sums all files in the blobs dir (including these) so bytes_done grows
    // smoothly from 0 → model_size as chunks complete.
    //
    // We use `std::thread::scope` instead of `rayon::par_iter` here.  Chunk
    // downloads are I/O-bound (network recv syscalls); rayon threads that
    // block on I/O do NOT yield, so using `par_iter` for chunks INSIDE an
    // outer `par_iter` (the per-shard parallelism in `paths.rs`) can saturate
    // the shared rayon thread pool and starve other shards.  OS threads
    // (`thread::scope`) are kernel-scheduled and handled correctly by the
    // kernel's I/O subsystem regardless of pool pressure.
    let url_str = url.as_str();
    std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .iter()
            .map(|&(start, end, cp)| {
                s.spawn(move || download_chunk_to_file(url_str, start, end, cp, tok_ref))
            })
            .collect();

        // Collect results; surface the first error (remaining threads continue
        // to completion because `thread::scope` joins ALL handles regardless).
        let mut first_err: Option<anyhow::Error> = None;
        for h in handles {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => { first_err.get_or_insert(e); }
                Err(_)     => { first_err.get_or_insert_with(|| anyhow!("chunk thread panicked")); }
            }
        }
        first_err.map_or(Ok(()), Err)
    })?;

    // ── Stream-concatenate chunk files into the final blob ───────────────────
    // Chunks are deleted in order AFTER their bytes are appended to the blob.
    // Peak disk usage = model_size (chunk files) + model_size (growing blob)
    // briefly, but each chunk file is deleted immediately after being consumed
    // so total overhead is at most one extra chunk_size at any moment.
    info!("All chunks downloaded — assembling `{file}` blob…");
    {
        use std::io::{Write as _};
        let mut blob = std::fs::OpenOptions::new()
            .write(true).create(true).truncate(true)
            .open(&blob_path)?;
        let mut read_buf = vec![0u8; 4 * 1024 * 1024]; // 4 MB copy buffer
        for (i, cp) in chunk_paths.iter().enumerate() {
            let mut chunk_file = std::fs::File::open(cp)
                .map_err(|e| anyhow!("open chunk {i}: {e}"))?;
            loop {
                use std::io::Read as _;
                let n_read = chunk_file.read(&mut read_buf)
                    .map_err(|e| anyhow!("read chunk {i}: {e}"))?;
                if n_read == 0 { break; }
                blob.write_all(&read_buf[..n_read])
                    .map_err(|e| anyhow!("write blob (chunk {i}): {e}"))?;
            }
            drop(chunk_file);
            let _ = fs::remove_file(cp); // delete chunk after appending
        }
    }

    info!("Blob assembled for `{file}` — registering in hf cache");

    // ── Register in cache ────────────────────────────────────────────────────
    register_in_cache(cache.path(), &folder, &meta.etag, &meta.commit_hash, file)?;

    Ok(pointer)
}

/// Download (or return the cached copy of) a single model file, with:
///
/// 1. **Resume**: `hf_hub` already sends `Range: bytes=N-` when a `.part` file
///    from a previous interrupted download exists — no extra work needed.
///
/// 2. **Corruption recovery**: after every cache hit or completed download we
///    verify the file is non-empty and (for `.safetensors`) has a parseable
///    header.  If the check fails we delete the pointer + blob and retry.
///
/// 3. **`InvalidResume` recovery**: happens when a `.part` file is *larger*
///    than the server's declared file size (can occur after a model is
///    re-uploaded with a size change, or after rare disk corruption).  We
///    locate the oversized `.part` via a lightweight HEAD request (using the
///    same `ureq` agent already compiled in via hf-hub), delete it, and retry.
///
/// At most `MAX_DOWNLOAD_ATTEMPTS` total attempts are made before giving up.
pub(crate) fn get_file(api: &ApiRepo, model_id: &Path, file: &str) -> Result<PathBuf> {
    // ── Local directory path ──────────────────────────────────────────────────
    // For local models the caller is responsible for file integrity; we just
    // check existence and pass through.
    if model_id.exists() {
        let path = model_id.join(file);
        if !path.exists() {
            return Err(local_file_missing_error(model_id, file));
        }
        info!("Loading `{file}` locally at `{}`", path.display());
        return Ok(path);
    }

    // ── HF cache download with repair loop ───────────────────────────────────
    const MAX_ATTEMPTS: usize = 3;
    let mut last_err = String::new();

    // Weight files (`.safetensors`, `.gguf`, `.bin`, `.pt`, `.pth`) are
    // downloaded with N parallel TCP connections on the first attempt.  If the
    // fast path fails for any reason (network error, unsupported server
    // response, permission error) we fall through to `api.get()`.
    let is_weight = file.ends_with(".safetensors")
        || file.ends_with(".gguf")
        || file.ends_with(".bin")
        || file.ends_with(".pt")
        || file.ends_with(".pth");

    for attempt in 0..MAX_ATTEMPTS {
        if attempt > 0 {
            warn!(
                "[{}/{}] Retrying download of `{file}` for `{}`…",
                attempt + 1,
                MAX_ATTEMPTS,
                model_id.display()
            );
        }

        // Fast path: multi-connection download for large weight files.
        // On the first attempt use our parallel downloader; on retries fall
        // back to hf-hub (which supports resume from a `.part` file).
        let result = if is_weight && attempt == 0 {
            match parallel_download(api, model_id, file) {
                Ok(path) => {
                    info!("Parallel download complete for `{file}`");
                    Ok(path)
                }
                Err(e) => {
                    // Determine whether this is an expected / silent fallback
                    // or a genuine unexpected failure.
                    let msg = e.to_string();
                    let silent = msg.contains("too small") || msg.contains("xet backend");
                    if silent {
                        info!(
                            "Using single-connection download for `{file}` \
                             (xet backend or small file)"
                        );
                    } else {
                        warn!(
                            "Parallel download of `{file}` failed ({e}); \
                             falling back to single-connection download"
                        );
                    }
                    api.get(file).map_err(|ae| hf_api_error(model_id, Some(file), &ae))
                }
            }
        } else {
            api.get(file).map_err(|ae| match ae {
                // Re-wrap for the InvalidResume handler below.
                ref e if is_invalid_resume(e) => {
                    // Return as-is so the outer match can pattern-match on it.
                    hf_api_error(model_id, Some(file), e)
                }
                e => hf_api_error(model_id, Some(file), &e),
            })
        };

        match result {
            Ok(path) => {
                // ── Verify the cached / freshly-downloaded file ───────────────
                match verify_cached_file(&path, file) {
                    Ok(()) => {
                        info!("Verified `{file}` at `{}`", path.display());
                        return Ok(path);
                    }
                    Err(reason) => {
                        warn!(
                            "File `{file}` in cache is corrupt ({reason}); \
                             removing and re-downloading (attempt {}/{MAX_ATTEMPTS}).",
                            attempt + 1
                        );
                        purge_cached_file(&path);
                        last_err = reason;
                    }
                }
            }
            Err(e) => {
                // Surface unrecoverable errors immediately;
                // wrap InvalidResume in a recoverable path.
                if is_invalid_resume_str(&e.to_string()) {
                    warn!(
                        "InvalidResume for `{file}`: partial download is larger than \
                         expected.  Locating and removing the corrupt .part file…"
                    );
                    if let Err(clean_err) =
                        purge_part_file_via_head(api, model_id, file)
                    {
                        warn!("Could not clean up .part file: {clean_err}");
                    }
                    last_err = e.to_string();
                } else {
                    return Err(e);
                }
            }
        }
    }

    Err(anyhow!(
        "Failed to obtain a valid copy of `{file}` for `{}` after \
         {MAX_ATTEMPTS} attempts: {last_err}",
        model_id.display()
    ))
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Returns `true` if the `ApiError` is an `InvalidResume` (the `.part` file
/// on disk is larger than the server's declared `Content-Length`).
fn is_invalid_resume(err: &ApiError) -> bool {
    matches!(err, ApiError::InvalidResume)
        || is_invalid_resume_str(&err.to_string())
}

fn is_invalid_resume_str(s: &str) -> bool {
    let lower = s.to_lowercase();
    lower.contains("invalidresume") || lower.contains("invalid resume")
}

/// Verify a cached file is readable, non-empty, and — for `.safetensors` —
/// has a self-consistent header.
fn verify_cached_file(path: &Path, filename: &str) -> Result<(), String> {
    // Resolve the pointer symlink to the actual blob.
    let real = path
        .canonicalize()
        .map_err(|e| format!("cannot resolve symlink: {e}"))?;

    let meta = real
        .metadata()
        .map_err(|e| format!("cannot stat blob: {e}"))?;

    if meta.len() == 0 {
        return Err("file is empty (0 bytes)".into());
    }

    if filename.ends_with(".safetensors") {
        verify_safetensor_header(&real, meta.len())?;
    }

    Ok(())
}

/// Check that the first 8 bytes of a safetensors file parse as a plausible
/// header-size value (must be > 0 and ≤ file_size − 8).
fn verify_safetensor_header(path: &Path, file_size: u64) -> Result<(), String> {
    use std::io::Read as _;
    let mut f = std::fs::File::open(path).map_err(|e| format!("cannot open: {e}"))?;
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)
        .map_err(|e| format!("cannot read header length prefix: {e}"))?;
    let header_size = u64::from_le_bytes(buf);
    if header_size == 0 {
        return Err("safetensors header-size prefix is 0".into());
    }
    // The 8-byte prefix + header JSON must fit inside the file.
    if 8u64.saturating_add(header_size) > file_size {
        return Err(format!(
            "safetensors header-size ({header_size} B) + 8-byte prefix \
             exceeds file size ({file_size} B): file is truncated"
        ));
    }
    Ok(())
}

/// Delete the pointer symlink and the blob it points to, so the next
/// `api.get()` call triggers a fresh download.
fn purge_cached_file(pointer_path: &Path) {
    // On macOS/Linux the pointer is a symlink; canonicalize() gives the blob.
    // On Windows hf-hub copies instead; we just remove what we have.
    if let Ok(blob) = pointer_path.canonicalize() {
        if let Err(e) = std::fs::remove_file(&blob) {
            warn!("Could not remove corrupt blob `{}`: {e}", blob.display());
        } else {
            info!("Removed corrupt blob `{}`", blob.display());
        }
    }
    if let Err(e) = std::fs::remove_file(pointer_path) {
        warn!(
            "Could not remove pointer `{}`: {e}",
            pointer_path.display()
        );
    } else {
        info!("Removed corrupt pointer `{}`", pointer_path.display());
    }
}

/// Make a HEAD request to resolve the ETag for `file`, then delete the
/// corresponding `<etag>.part` file from the blobs directory.
///
/// The ETag is the blob filename used by hf-hub's cache layout:
///   `<cache>/<model-folder>/blobs/<etag>`        — complete blob
///   `<cache>/<model-folder>/blobs/<etag>.part`   — in-progress download
///
/// We use `ureq` directly (already compiled in as a transitive dep of
/// hf-hub) to send the HEAD request with the same auth token hf-hub would
/// use, so we avoid any extra network round-trips for normal downloads.
fn purge_part_file_via_head(api: &ApiRepo, model_id: &Path, file: &str) -> Result<()> {
    use crate::GLOBAL_HF_CACHE;

    let url = api.url(file);

    // ── HEAD request to resolve the ETag ─────────────────────────────────────
    // We read the HF token from the standard locations so the request works
    // for gated models.
    let token = read_hf_token();
    let etag = {
        let mut req = ureq::head(&url);
        if let Some(ref t) = token {
            req = req.set("Authorization", &format!("Bearer {t}"));
        }
        // ureq follows up to 5 redirects by default (HF CDN redirects HEAD
        // to the actual storage URL).
        let resp = req.call().map_err(|e| anyhow!("HEAD {url}: {e}"))?;

        // HF returns the ETag as a quoted string like `"sha256-abc123..."`.
        let raw = resp
            .header("ETag")
            .or_else(|| resp.header("etag"))
            .ok_or_else(|| anyhow!("No ETag in HEAD response for {url}"))?;
        raw.trim_matches('"').to_owned()
    };

    // ── Locate the blobs directory ────────────────────────────────────────────
    // hf-hub cache layout (matches Python huggingface_hub):
    //   <cache>/models--<org>--<model>/blobs/<etag>
    //
    // `model_id` is a relative path like `Qwen/Qwen3.5-7B`; we map it to the
    // folder name with the same algorithm as `Repo::folder_name()`.
    let folder_name = format!(
        "models--{}",
        model_id.display().to_string().replace('/', "--")
    );
    let cache = GLOBAL_HF_CACHE.get().cloned().unwrap_or_default();
    let blobs_dir = cache.path().join(&folder_name).join("blobs");

    let part_path = blobs_dir.join(format!("{etag}.part"));
    if part_path.exists() {
        std::fs::remove_file(&part_path)
            .map_err(|e| anyhow!("Could not remove `{}`: {e}", part_path.display()))?;
        info!(
            "Removed oversized .part file `{}` (ETag: {etag})",
            part_path.display()
        );
    } else {
        // Nothing to delete — may have been cleaned up already.
        info!(
            "No .part file found at `{}` (ETag: {etag}); nothing to remove.",
            part_path.display()
        );
    }

    Ok(())
}

/// Read the cached HF access token, trying the same locations hf-hub does.
fn read_hf_token() -> Option<String> {
    // 1. Environment variable (highest priority, used in CI / containers).
    if let Ok(t) = std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN")) {
        if !t.is_empty() {
            return Some(t);
        }
    }
    // 2. Token file written by `huggingface-cli login`.
    hf_token_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}
