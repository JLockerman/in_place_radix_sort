
extern crate byteorder;

use byteorder::{BigEndian, ByteOrder};

pub fn sort<T>(array: &mut [T], key: impl FnMut(&T) -> u64 + Copy) {
    let mut memory: Vec<[usize; 256]> = Vec::with_capacity(8 * 2);
    unsafe { memory.set_len(8 * 2) }
    let (histograms, starts) = memory.split_at_mut(8);
    sort_level(array, histograms, starts, 0, key)
}

fn sort_level<T>(
    array: &mut [T],
    histograms: &mut [[usize; 256]],
    starts: &mut [[usize; 256]],
    level: u8,
    key: impl FnMut(&T) -> u64 + Copy,
) {
    if array.len() <= 1 {
        return
    }

    let mut byte = key_at_level(level, key);

    assert_eq!(starts.len(), histograms.len());
    let (histogram, histograms) = histograms.split_first_mut().unwrap();
    let (start, starts) = starts.split_first_mut().unwrap();

    *histogram = [0; 256];

    for item in &mut *array {
        histogram[byte(item) as usize] += 1;
    }
    start[0] = 0;
    for i in 1..start.len() {
        start[i] = start[i - 1] + histogram[i - 1];
    }
    let ends = histogram;
    for i in 0..ends.len() {
        ends[i] += start[i];
    }

    for dx in 0..start.len() {
        let i = dx as u8;
        while start[dx] < ends[dx] {
            'swap: loop {
                let b = byte(&mut array[start[dx]]);
                if b == i {
                    break 'swap
                }
                array.swap(start[dx], start[b as usize]);
                start[b as usize] += 1;
                debug_assert!(start[b as usize] <= ends[b as usize]);
            }
            start[dx] += 1;
        }
    }

    if level < 7 {
        start[0] = 0;
        start[1..].copy_from_slice(&ends[..255]);
        for i in 0..start.len() {
            if start[i] == ends[i] { continue }
            sort_level(&mut array[start[i]..ends[i]], histograms, starts, level + 1, key);
        }
    }
}

#[inline(always)]
fn key_at_level<T>(level: u8, mut key: impl FnMut(&T) -> u64) -> impl FnMut(&T) -> u8 {
    // move |t| ((key(t) >> level) & 0xff) as u8
    move |t| { 
        let mut bytes = [0; 8];
        BigEndian::write_u64(&mut bytes, key(t));
        bytes[level as usize]
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct BitSet {
    high: u128,
    low: u128,
}

#[allow(dead_code)]
impl BitSet {
    fn new() -> Self {
        BitSet { low: 0, high: 0 }
    }

    #[inline(always)]
    fn set(&mut self, key: u8) {
        if key < 128 {
            self.low |= 1 << (key % 128)
        } else {
            self.high |= 1 << (key % 128)
        }
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.low == 0 && self.high == 0
    }

    #[inline(always)]
    fn take_next(&mut self) -> u8 {
        let (bits, add) =
            if self.low == 0 { (&mut self.high, 128) } else { (&mut self.low, 0) };
        let shift = bits.trailing_zeros() as u8;
        *bits &= !(1 << shift);
        shift + add
    }
}

#[cfg(test)]
pub mod tests {

    extern crate is_sorted;
    extern crate rand;

    use super::*;

    use self::{
        is_sorted::IsSorted,
        rand::prelude::*,
    };

    #[test]
    fn sum() {
        //              [ 0,  3,  8, 10, 11, 11, 37, 50]
        //              [ 3,  8, 10, 11, 11, 37, 50, 53]
        let histogram = [ 3,  5,  2,  1,  0, 26, 13,  3];
        let mut starts = [0, 0, 0, 0, 0, 0, 0, 0];
        
        starts[0] = 0;
        for i in 1..starts.len() {
            starts[i] = starts[i - 1] + histogram[i - 1];
        }
        let mut ends = histogram;
        for i in 0..ends.len() {
            ends[i] += starts[i];
        }
        // println!("{:?}", starts);
        // println!("{:?}", ends);
    }

    #[test]
    fn small_const() {
        let mut vals = [1u64, 0, 22, 5, 36, 2, 1111, 1112, 44];
        sort(&mut vals, |v| *v);
        assert!(vals.iter().is_sorted());
    }

    #[test]
    fn small_random() {
        let mut rng = thread_rng();
        let mut vals: Vec<u64> = (0..100).map(|_| rng.gen_range(0, 10000)).collect();
        sort(&mut vals, |v| *v);
        vals.windows(2).for_each(|w| assert!(w[0] <= w[1], "{} <= {}", w[0], w[1]));
        assert!(vals.iter().is_sorted());
    }

    #[test]
    fn small_random2() {
        let mut vals: Vec<u64> = (0..1000).map(|_| random()).collect();
        sort(&mut vals, |v| *v);
        vals.windows(2).for_each(|w| assert!(w[0] <= w[1], "{} <= {}", w[0], w[1]));
        assert!(vals.iter().is_sorted());
    }

    #[test]
    fn med_random() {
        let mut vals: Vec<u64> = (0..10_000_000).map(|_| random()).collect();
        sort(&mut vals, |v| *v);
        vals.windows(2).for_each(|w| assert!(w[0] <= w[1], "{} <= {}", w[0], w[1]));
        assert!(vals.iter().is_sorted());
    }
}
