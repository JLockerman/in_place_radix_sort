
extern crate byteorder;

use std::{
    // collections::VecDeque,
    mem::{
        size_of,
        uninitialized,
    },
};

use byteorder::{BigEndian, ByteOrder};

pub trait Byter {
    const LEVELS: usize;
    fn bytes(&self, usize) -> u8;
}

impl Byter for u64 {
    const LEVELS: usize = size_of::<Self>();
    #[inline(always)]
    fn bytes(&self, level: usize) -> u8 {
        let mut bytes = [0; size_of::<Self>()];
        BigEndian::write_u64(&mut bytes, *self);
        bytes[level]
        // unsafe {
        //     *bytes.get_unchecked(level)
        // }
    }
}

impl Byter for usize {
    //TODO dynamic set
    const LEVELS: usize = size_of::<u64>();
    #[inline(always)]
    fn bytes(&self, level: usize) -> u8 {
        let mut bytes = [0; size_of::<u64>()];
        BigEndian::write_u64(&mut bytes, *self as u64);
        bytes[level]
    }
}

impl Byter for u32 {
    const LEVELS: usize = size_of::<Self>();
    #[inline(always)]
    fn bytes(&self, level: usize) -> u8 {
        let mut bytes = [0; size_of::<Self>()];
        BigEndian::write_u32(&mut bytes, *self);
        bytes[level]
    }
}

impl Byter for u16 {
    const LEVELS: usize = size_of::<Self>();
    #[inline(always)]
    fn bytes(&self, level: usize) -> u8 {
        let mut bytes = [0; size_of::<Self>()];
        BigEndian::write_u16(&mut bytes, *self);
        bytes[level]
    }
}

impl Byter for u8 {
    const LEVELS: usize = size_of::<Self>();
    #[inline(always)]
    fn bytes(&self, _level: usize) -> u8 {
        *self
    }
}

pub fn sort<T>(array: &mut [T])
where T: Byter + Ord {
    sort_by(array, <T as Byter>::LEVELS, |k, l| <T as Byter>::bytes(k, l))
}

pub fn sort_by<T>(array: &mut [T], num_levels: usize, key: impl FnMut(&T, usize) -> u8 + Copy)
where T: Ord {
    let mut tables = Cache::with_capacity(num_levels * 2);
    sort_level(array, &mut tables, 0, num_levels, key)
}

#[inline]
fn sort_level<T>(
    array: &mut [T],
    tables: &mut Cache,
    level: usize,
    num_levels: usize,
    key: impl FnMut(&T, usize) -> u8 + Copy,
) where T: Ord {
    if array.len() <= 1 {
        return
    }

    assert!(level < num_levels);

    let mut byte = key_at_level(level, key);

    let mut histogram = tables.alloc();
    let mut start = tables.alloc();

    *histogram = [0; 256];

    for item in &mut *array {
        histogram[byte(item) as usize] += 1;
    }

    start[0] = 0;
    for i in 1..start.len() {
        start[i] = start[i - 1] + histogram[i - 1];
    }

    let mut ends = histogram;
    for i in 0..ends.len() {
        ends[i] += start[i];
    }

    for dx in 0..start.len() {
        let i = dx as u8;
        'search: while start[dx] < ends[dx] {
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

    if level + 1 < num_levels {
        start[0] = 0;
        start[1..].copy_from_slice(&ends[..255]);
        for i in 0..start.len() {
            if start[i] == ends[i] { continue }
            if start[i] + 1 == ends[i] { continue }
            let array = &mut array[start[i]..ends[i]];
            match array.len() {
                2 => if &array[0] > &array[1] {
                    array.swap(0, 1);
                },
                l if l <= 25 => array.sort(),
                _ => sort_level(array, tables, level + 1, num_levels, key),
            }
        }
    }

    tables.free(start);
    tables.free(ends);
}

#[inline(always)]
fn key_at_level<T>(level: usize, mut key: impl FnMut(&T, usize) -> u8) -> impl FnMut(&T) -> u8 {
    move |k| key(k, level)
}

struct Cache {
    cache: Vec<Box<[usize; 256]>>,
}

impl Cache {
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            cache: Vec::with_capacity(cap),
        }
    }

    #[inline]
    pub fn alloc(&mut self) -> Box<[usize; 256]> {
        self.cache.pop().unwrap_or_else(|| Box::new(unsafe { uninitialized() }))
    }

    pub fn free(&mut self, table: Box<[usize; 256]>) {
        self.cache.push(table)
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
        sort(&mut vals);
        assert!(vals.iter().is_sorted());
    }

    #[test]
    fn small_random() {
        let mut rng = thread_rng();
        let mut vals: Vec<u64> = (0..100).map(|_| rng.gen_range(0, 10000)).collect();
        println!("{:?}", vals);
        sort(&mut vals);
        println!("{:?}", vals);
        vals.windows(2).for_each(|w| assert!(w[0] <= w[1], "{} <= {}", w[0], w[1]));
        assert!(vals.iter().is_sorted());
    }

    #[test]
    fn small_random2() {
        let mut vals: Vec<u64> = (0..1000).map(|_| random()).collect();
        sort(&mut vals);
        vals.windows(2).for_each(|w| assert!(w[0] <= w[1], "{} <= {}", w[0], w[1]));
        assert!(vals.iter().is_sorted());
    }

    #[test]
    fn med_random() {
        let mut vals: Vec<u64> = (0..10_000_000).map(|_| random()).collect();
        sort(&mut vals);
        vals.windows(2).for_each(|w| assert!(w[0] <= w[1], "{} <= {}", w[0], w[1]));
        assert!(vals.iter().is_sorted());
    }

    #[test]
    fn sort_network_5() {
        let mut rng = thread_rng();
        let mut vals: Vec<u64> = (0..5).map(|_| rng.gen_range(0, 100)).collect();
        println!("{:?}", vals);
        {
            let swap_if_needed = |vals: &mut Vec<_>, i, j| if &vals[i] > &vals[j] {
                vals.swap(i, j);
            };
            swap_if_needed(&mut vals, 0, 1);
            println!("{:?}", vals);
            swap_if_needed(&mut vals, 2, 3);
            println!("{:?}", vals);
            swap_if_needed(&mut vals, 1, 3);
            println!("{:?}", vals);
            swap_if_needed(&mut vals, 2, 4);
            println!("{:?}", vals);
            swap_if_needed(&mut vals, 1, 4);            
            println!("{:?}", vals);
            swap_if_needed(&mut vals, 0, 2);
            println!("{:?}", vals);
            swap_if_needed(&mut vals, 1, 2);
            println!("{:?}", vals);
            swap_if_needed(&mut vals, 3, 4);
            println!("{:?}", vals);
            swap_if_needed(&mut vals, 2, 3);
        }
        println!("{:?}", vals);
        vals.windows(2).for_each(|w| assert!(w[0] <= w[1], "{} <= {}", w[0], w[1]));
        assert!(vals.iter().is_sorted());
    }
}
