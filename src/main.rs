use std::{
    env,
    fs::File,
    io::{self, BufRead, BufReader},
    slice,
};

// How many hashes do we compute at a time?
const BATCH_SIZE: usize = 16384;
const MAX_LINE_LEN: usize = 256;
const LOG_EVERY_BATCHES: usize = 0;

// Vector of bytes; used to interface w/CUDA code
#[repr(C)]
#[derive(Debug)]
pub struct FfiVector {
    data: *mut u8,
    len: usize,
}

// Vector of FfiVectors; used to interface w/CUDA code
#[repr(C)]
#[derive(Debug)]
pub struct FfiVectorBatched {
    data: *mut FfiVector,
    len: usize,
}

#[link(name = "cudacracker", kind = "static")]
unsafe extern "C" {
    unsafe fn init();
    unsafe fn md5_target_batched_wrapper(msgs: &FfiVectorBatched, target_digest: &FfiVector) -> i32;
}

impl From<Vec<u8>> for FfiVector {
    fn from(value: Vec<u8>) -> Self {
        let len = value.len();
        let data = value.as_ptr() as *mut u8;
        std::mem::forget(value);

        FfiVector { data, len }
    }
}

impl From<FfiVector> for Vec<u8> {
    fn from(value: FfiVector) -> Self {
        let n = value.len;
        let data = value.data;

        unsafe {
            let data_slice = slice::from_raw_parts(data, n);
            
            data_slice.to_vec()
        }
    }
}

impl From<Vec<FfiVector>> for FfiVectorBatched {
    fn from(value: Vec<FfiVector>) -> Self {
        let len = value.len();
        let data = value.as_ptr() as *mut FfiVector;
        std::mem::forget(value);

        FfiVectorBatched { data, len }
    }
}

impl From<Vec<Vec<u8>>> for FfiVectorBatched {
    fn from(value: Vec<Vec<u8>>) -> Self {
        let ffi_vecs: Vec<FfiVector> = value.into_iter().map(|x| FfiVector::from(x)).collect();
        
        FfiVectorBatched::from(ffi_vecs)
    }
}

impl From<FfiVectorBatched> for Vec<Vec<u8>> {
    fn from(value: FfiVectorBatched) -> Self {
        let n = value.len;
        let data = value.data;

        unsafe {
            let data_slice = slice::from_raw_parts(data, n);
            
            data_slice.into_iter().map(|x| slice::from_raw_parts(x.data, x.len).to_vec()).collect()
        }
    }
}

// From the wordlist, find a string whose digest matches the input; if such a string does not exist, return None
fn try_batch(target_digest: &FfiVector, batch: &mut Vec<Vec<u8>>, actual_len: usize) -> Option<Vec<u8>> {
    let mut ffi_vecs: Vec<FfiVector> = batch
        .iter_mut()
        .map(|entry| FfiVector {
            data: entry.as_mut_ptr(),
            len: entry.len(),
        })
        .collect();
    let ffi_batch = FfiVectorBatched {
        data: ffi_vecs.as_mut_ptr(),
        len: ffi_vecs.len(),
    };

    unsafe {
        let idx = md5_target_batched_wrapper(&ffi_batch, target_digest);
        if idx != -1 && (idx as usize) < actual_len {
            return Some(batch[idx as usize].clone());
        }
    }

    None
}

fn crack_stream(reader: impl BufRead, digest: &str) -> io::Result<Option<Vec<u8>>> {
    let mut target_bytes = hex::decode(digest).expect("Failed to decode digest");
    let target_digest = FfiVector {
        data: target_bytes.as_mut_ptr(),
        len: target_bytes.len(),
    };

    let mut batch: Vec<Vec<u8>> = Vec::with_capacity(BATCH_SIZE);
    let mut line = Vec::new();
    let mut reader = reader;
    let mut batch_index: usize = 0;
    let mut total_lines: u64 = 0;
    let mut total_bytes: u64 = 0;
    let mut batch_bytes: u64 = 0;
    let mut batch_max_len: usize = 0;
    let mut skipped_long: u64 = 0;

    loop {
        line.clear();
        let bytes_read = reader.read_until(b'\n', &mut line)?;
        if bytes_read == 0 {
            break;
        }

        if line.ends_with(b"\n") {
            line.pop();
            if line.ends_with(b"\r") {
                line.pop();
            }
        }

        if line.is_empty() {
            continue;
        }

        if line.len() > MAX_LINE_LEN {
            skipped_long += 1;
            continue;
        }

        total_lines += 1;
        total_bytes += line.len() as u64;
        batch_bytes += line.len() as u64;
        batch_max_len = batch_max_len.max(line.len());
        batch.push(std::mem::take(&mut line));

        if batch.len() == BATCH_SIZE {
            batch_index += 1;
            if LOG_EVERY_BATCHES != 0 && batch_index % LOG_EVERY_BATCHES == 0 {
                eprintln!(
                    "batch {batch_index} lines={total_lines} bytes={total_bytes} batch_bytes={batch_bytes} max_len={batch_max_len} skipped={skipped_long}"
                );
            }
            if let Some(found) = try_batch(&target_digest, &mut batch, BATCH_SIZE) {
                return Ok(Some(found));
            }
            batch.clear();
            batch_bytes = 0;
            batch_max_len = 0;
            skipped_long = 0;
        }
    }

    if !batch.is_empty() {
        batch_index += 1;
        if LOG_EVERY_BATCHES != 0 {
            eprintln!(
                "batch {batch_index} lines={total_lines} bytes={total_bytes} batch_bytes={batch_bytes} max_len={batch_max_len} skipped={skipped_long} (final)"
            );
        }
        let actual_len = batch.len();
        if actual_len < BATCH_SIZE {
            batch.resize_with(BATCH_SIZE, Vec::new);
        }
        let found = try_batch(&target_digest, &mut batch, actual_len);
        batch.truncate(actual_len);
        if found.is_some() {
            return Ok(found);
        }
    }

    Ok(None)
}

fn main() -> Result<(), io::Error> {
    unsafe {
        init();
    }
    
    let wordlist_file = File::open(env::args().nth(1).expect("Expected wordlist file name"))?;
    let digest = env::args().nth(2).expect("Expected hash");
    let reader = BufReader::new(wordlist_file);

    if let Some(result) = crack_stream(reader, &digest)? {
        let printable = String::from_utf8_lossy(&result);
        println!("Hash cracked: md5({printable}) = {digest}");
    } else {
        println!("Couldn't crack hash");
    }

    Ok(())
}