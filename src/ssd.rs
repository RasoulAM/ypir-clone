//! "Server data on SSD" mode for YPIR+SP (SimplePIR variant).
//!
//! What the online phase of YPIR+SP needs (see
//! `YServer::perform_online_computation_simplepir`):
//!   * the database (DB-dependent),
//!   * `precomp[i].0` row 0 and `precomp[i].1` (the gadget-decomposed
//!     ring-packing precomputation derived from hint_0 = A * DB, i.e.
//!     DB-dependent),
//!   * `precomp[i].2` (NTT automorphism tables) and `y_constants`, which only
//!     depend on the RLWE parameters (database-independent).
//! `hint_0`, `prepacked_lwe` and `fake_pack_pub_params` are not used online.
//!
//! The DB-dependent parts are written to one file and read back per query;
//! the parameter-derived parts stay resident (and are reported).
//!
//! File layout (all little-endian u64 words unless noted):
//!   header: HEADER_WORDS words
//!   precomp: for each RLWE output: precomp_res row 0 (crt_count * poly_len
//!            words), then for each of the precomp_vals matrices
//!            (t_exp_left x 1, condensed form) the first poly_len words of
//!            each row (the condensed upper halves are unused/zero)
//!   database: for col in 0..stored_cols:
//!            db[col * db_rows .. col * db_rows + stored_rows] as raw T
//!            (the DB is stored transposed). Rows >= stored_rows and
//!            cols >= stored_cols are zero padding (YPIR+SP pads every item
//!            to instances * 2048 coefficients of 14 bits and the number of
//!            rows to a power of two) and are re-created on load.

use std::fs::File;
use std::io::{Read, Write};
use std::os::unix::io::AsRawFd;
use std::time::Instant;

use spiral_rs::params::Params;
use spiral_rs::poly::{PolyMatrix, PolyMatrixNTT};

use crate::server::{Precomp, ToM512, ToU64, YServer};

pub const SSD_FILE: &str = "/tmp/ypir_db.bin";
const MAGIC: u64 = 0x3144_5353_5249_5059; // "YPIRSSD1"
const HEADER_WORDS: usize = 16;

/// Which part of the (padded) database holds real data.
#[derive(Clone, Copy, Debug)]
pub struct DbLayout {
    pub db_rows: usize,
    pub db_cols: usize,
    pub stored_rows: usize,
    pub stored_cols: usize,
    pub elem_size: usize,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct LoadTimes {
    pub read_us: usize,
    pub parse_us: usize,
    pub derive_us: usize,
    pub total_us: usize,
}

fn words_as_bytes(w: &[u64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 8) }
}

fn words_as_bytes_mut(w: &mut [u64]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(w.as_mut_ptr() as *mut u8, w.len() * 8) }
}

fn precomp_words_per_output(params: &Params, num_vals: usize) -> usize {
    params.crt_count * params.poly_len + num_vals * params.t_exp_left * params.poly_len
}

/// Evicts the file from the page cache (no-op for pages that are not cached).
pub fn evict_from_page_cache(path: &str) {
    let f = File::open(path).unwrap();
    unsafe {
        libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
    }
}

/// Page-cache residency of the given files (untimed diagnostic): mmaps each
/// file read-only and asks `mincore` which of its pages are resident.
/// Returns `(resident_pages, total_pages)` summed over all files.
pub fn page_cache_residency(paths: &[&str]) -> (usize, usize) {
    use std::os::unix::io::AsRawFd;
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let (mut resident, mut total) = (0usize, 0usize);
    for path in paths {
        let file = std::fs::File::open(path).unwrap();
        let len = file.metadata().unwrap().len() as usize;
        if len == 0 {
            continue;
        }
        let pages = (len + page - 1) / page;
        let mut vec = vec![0u8; pages];
        unsafe {
            let addr = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            );
            assert!(addr != libc::MAP_FAILED, "mmap failed for {}", path);
            let rc = libc::mincore(addr, len, vec.as_mut_ptr());
            libc::munmap(addr, len);
            assert_eq!(rc, 0, "mincore failed for {}", path);
        }
        resident += vec.iter().filter(|&&b| b & 1 != 0).count();
        total += pages;
    }
    (resident, total)
}

/// Prints the page-cache residency of the server-state file(s); call after
/// eviction and before the timed load.
pub fn print_page_cache_residency_before_load(paths: &[&str]) {
    let (resident, total) = page_cache_residency(paths);
    let pct = if total == 0 { 0.0 } else { 100.0 * resident as f64 / total as f64 };
    println!(
        "Server Page Cache Residency Before Load: {}/{} pages ({:.3}%)",
        resident, total, pct
    );
}

/// Writes the DB-dependent server state, fdatasyncs it and evicts it from the
/// page cache. Returns the file size in bytes.
pub fn write_state<'a, T>(
    path: &str,
    params: &'a Params,
    y_server: &YServer<'a, T>,
    precomp: &Precomp<'a>,
    layout: DbLayout,
) -> usize
where
    T: Sized + Copy + ToU64 + Default,
    *const T: ToM512,
{
    let poly_len = params.poly_len;
    let num_outputs = precomp.len();
    let num_vals = precomp[0].1.len();
    assert_eq!(layout.db_rows, y_server.db_rows_padded());
    assert_eq!(layout.db_cols, y_server.db_cols());
    assert_eq!(layout.elem_size, std::mem::size_of::<T>());

    let mut file = File::create(path).unwrap();

    let mut header = [0u64; HEADER_WORDS];
    header[0] = MAGIC;
    header[1] = poly_len as u64;
    header[2] = params.crt_count as u64;
    header[3] = params.t_exp_left as u64;
    header[4] = num_outputs as u64;
    header[5] = num_vals as u64;
    header[6] = layout.db_rows as u64;
    header[7] = layout.db_cols as u64;
    header[8] = layout.stored_rows as u64;
    header[9] = layout.stored_cols as u64;
    header[10] = layout.elem_size as u64;
    file.write_all(words_as_bytes(&header)).unwrap();

    let words_per_output = precomp_words_per_output(params, num_vals);
    let mut buf = Vec::with_capacity(words_per_output);
    for (res, vals, _tables) in precomp.iter() {
        buf.clear();
        assert_eq!(vals.len(), num_vals);
        // Only row 0 of precomp_res is used online (row 1 is overwritten).
        buf.extend_from_slice(res.get_poly(0, 0));
        for v in vals.iter() {
            assert_eq!(v.rows, params.t_exp_left);
            assert_eq!(v.cols, 1);
            for r in 0..v.rows {
                let p = v.get_poly(r, 0);
                // Condensed form: the upper half of every poly is zero.
                debug_assert!(p[poly_len..].iter().all(|&x| x == 0));
                buf.extend_from_slice(&p[..poly_len]);
            }
        }
        assert_eq!(buf.len(), words_per_output);
        file.write_all(words_as_bytes(&buf)).unwrap();
    }

    let db = y_server.db_bytes();
    let es = layout.elem_size;
    if layout.stored_rows == layout.db_rows {
        file.write_all(&db[..layout.stored_cols * layout.db_rows * es])
            .unwrap();
    } else {
        for col in 0..layout.stored_cols {
            let start = col * layout.db_rows * es;
            file.write_all(&db[start..start + layout.stored_rows * es])
                .unwrap();
        }
    }
    file.flush().unwrap();
    // Test hook (smoke tests only): corrupt one word of the stored precomp
    // or database to show that the online phase really uses the loaded data.
    if let Some(what) = std::env::var("YPIR_SSD_SELFTEST_CORRUPT").ok().filter(|w| !w.is_empty()) {
        use std::io::{Seek, SeekFrom};
        let off = match what.as_str() {
            "precomp" => (HEADER_WORDS * 8 + 8 * (params.crt_count * poly_len + 5)) as u64,
            _ => (HEADER_WORDS * 8 + 8 * num_outputs * words_per_output) as u64,
        };
        file.seek(SeekFrom::Start(off)).unwrap();
        file.write_all(&[0x5a, 0x01]).unwrap();
        eprintln!("SELFTEST: corrupted stored {} at byte {}", what, off);
    }
    file.sync_data().unwrap();
    let total_size = file.metadata().unwrap().len() as usize;
    unsafe {
        libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
    }
    total_size
}

/// Loads the DB-dependent server state written by `write_state`: reads the
/// precomputation into one staging buffer and parses it into `Precomp`
/// (attaching the resident, parameter-derived automorphism tables), reads
/// the database directly into a freshly allocated DB buffer of `y_server`
/// and re-creates the zero padding.
pub fn load_state<'a, T>(
    path: &str,
    params: &'a Params,
    y_server: &mut YServer<'a, T>,
    resident_tables: &[Vec<usize>],
) -> (Precomp<'a>, LoadTimes)
where
    T: Sized + Copy + ToU64 + Default,
    *const T: ToM512,
{
    let start = Instant::now();
    let poly_len = params.poly_len;

    // --- read: header + precomp section (one sequential read) ---
    let mut file = File::open(path).unwrap();
    let mut header = [0u64; HEADER_WORDS];
    file.read_exact(words_as_bytes_mut(&mut header)).unwrap();
    assert_eq!(header[0], MAGIC, "bad SSD state file");
    assert_eq!(header[1] as usize, poly_len);
    assert_eq!(header[2] as usize, params.crt_count);
    assert_eq!(header[3] as usize, params.t_exp_left);
    let num_outputs = header[4] as usize;
    let num_vals = header[5] as usize;
    let layout = DbLayout {
        db_rows: header[6] as usize,
        db_cols: header[7] as usize,
        stored_rows: header[8] as usize,
        stored_cols: header[9] as usize,
        elem_size: header[10] as usize,
    };
    assert_eq!(layout.db_rows, y_server.db_rows_padded());
    assert_eq!(layout.db_cols, y_server.db_cols());
    assert_eq!(layout.elem_size, std::mem::size_of::<T>());

    let words_per_output = precomp_words_per_output(params, num_vals);
    let mut staging = vec![0u64; num_outputs * words_per_output];
    file.read_exact(words_as_bytes_mut(&mut staging)).unwrap();
    let mut read_us = start.elapsed().as_micros() as usize;

    // --- parse: precomp ---
    let t_parse = Instant::now();
    let crt_poly = params.crt_count * poly_len;
    let mut precomp: Precomp<'a> = Vec::with_capacity(num_outputs);
    for i in 0..num_outputs {
        let w = &staging[i * words_per_output..(i + 1) * words_per_output];
        let mut res = PolyMatrixNTT::zero(params, 2, 1);
        res.get_poly_mut(0, 0).copy_from_slice(&w[..crt_poly]);
        let mut off = crt_poly;
        let mut vals = Vec::with_capacity(num_vals);
        for _ in 0..num_vals {
            let mut v = PolyMatrixNTT::zero(params, params.t_exp_left, 1);
            for r in 0..params.t_exp_left {
                v.get_poly_mut(r, 0)[..poly_len].copy_from_slice(&w[off..off + poly_len]);
                off += poly_len;
            }
            vals.push(v);
        }
        assert_eq!(off, words_per_output);
        precomp.push((res, vals, resident_tables.to_vec()));
    }
    drop(staging);
    let parse_us = t_parse.elapsed().as_micros() as usize;

    // --- read: database, directly into a fresh DB buffer ---
    let t_read_db = Instant::now();
    let es = layout.elem_size;
    {
        let db = y_server.alloc_db_bytes();
        if layout.stored_rows == layout.db_rows {
            file.read_exact(&mut db[..layout.stored_cols * layout.db_rows * es])
                .unwrap();
        } else {
            for col in 0..layout.stored_cols {
                let s = col * layout.db_rows * es;
                file.read_exact(&mut db[s..s + layout.stored_rows * es])
                    .unwrap();
            }
        }
    }
    read_us += t_read_db.elapsed().as_micros() as usize;
    // The whole file must have been consumed.
    let mut probe = [0u8; 1];
    assert_eq!(file.read(&mut probe).unwrap(), 0, "trailing data in SSD file");

    // --- derive: re-create the zero padding (explicitly touched here so that
    // the page-fault / zeroing cost is part of LOAD, not of COMPUTE) ---
    let t_derive = Instant::now();
    {
        let db = y_server.db_bytes_mut();
        if layout.stored_rows < layout.db_rows {
            for col in 0..layout.stored_cols {
                let s = (col * layout.db_rows + layout.stored_rows) * es;
                let e = (col + 1) * layout.db_rows * es;
                db[s..e].fill(0);
            }
        }
        let tail = layout.stored_cols * layout.db_rows * es;
        db[tail..].fill(0);
    }
    let derive_us = t_derive.elapsed().as_micros() as usize;

    let total_us = start.elapsed().as_micros() as usize;
    (
        precomp,
        LoadTimes {
            read_us,
            parse_us,
            derive_us,
            total_us,
        },
    )
}

/// Bytes of the parameter-derived state that SSD mode keeps resident.
pub fn resident_state_bytes(
    params: &Params,
    y_constants: &(Vec<PolyMatrixNTT>, Vec<PolyMatrixNTT>),
    tables: &[Vec<usize>],
) -> usize {
    let poly_bytes = params.crt_count * params.poly_len * 8;
    let y_bytes = (y_constants.0.len() + y_constants.1.len()) * poly_bytes;
    let table_bytes: usize = tables.iter().map(|t| t.len() * 8).sum();
    // One resident copy of the tables (attached to every RLWE output on load).
    y_bytes + table_bytes
}
