
extern crate byteorder;

use byteorder::{BigEndian, ByteOrder};

pub fn sort<T>(array: &mut [T], key: impl FnMut(&mut T) -> u64 + Copy) {
    //TODO better alloc?
    sort_level(array, 0, key)
}

fn sort_level<T>(
    array: &mut [T],
    level: u8,
    key: impl FnMut(&mut T) -> u64 + Copy,
) {
    if array.len() <= 1 {
        return
    }

    let mut histogram = vec![0; 256];
    let mut starts = vec![0; 256];
    assert_eq!(starts.len(), histogram.len());

    let mut byte = key_at_level(level, key);

    for item in &mut *array {
        histogram[byte(item) as usize] += 1;
    }
    starts[0] = 0;
    for i in 1..starts.len() {
        starts[i] = starts[i - 1] + histogram[i - 1];
    }
    let mut ends = histogram;
    for i in 0..ends.len() {
        ends[i] += starts[i];
    }

    for i in 0..ends.len() {
        while starts[i] < ends[i] {
            'swap: loop {
                let b = byte(&mut array[starts[i]]) as usize;
                println!("{} 0x{:2x} 0x{:2x}", level, i, b);
                if b == i {
                    break 'swap
                }
                array.swap(starts[i], starts[b]);
                starts[b] += 1;
                debug_assert!(starts[b] <= ends[b]);
            }
            starts[i] += 1;
        }
    }
    

    println!("start {:?}", starts);
    println!("  end {:?}\n", ends);

    if level < 7 {
        starts[0] = 0;
        for i in 1..starts.len() {
            starts[i] = ends[i - 1];
        }
        for i in 0..starts.len() {
            if starts[i] == ends[i] { continue }
            sort_level(&mut array[starts[i]..ends[i]], level + 1, key);
        }
    }
}

#[inline(always)]
fn key_at_level<T>(level: u8, mut key: impl FnMut(&mut T) -> u64) -> impl FnMut(&mut T) -> u8 {
    // move |t| ((key(t) >> level) & 0xff) as u8
    move |t| { 
        let mut bytes = [0; 8];
        BigEndian::write_u64(&mut bytes, key(t));
        bytes[level as usize]
    }
}

#[allow(dead_code)]
struct BitSet {
    high: u128,
    low: u128,
}

#[allow(dead_code)]
impl BitSet {
    fn new() -> Self {
        BitSet { low: 0, high: 0 }
    }

    fn set(&mut self, key: u8) {
        if key < 128 {
            self.low |= 1 << (key % 128)
        } else {
            self.high |= 1 << (key % 128)
        }
    }

    fn is_empty(&self) -> bool {
        self.low == 0 && self.high == 0
    }

    fn take_next(&mut self) -> u8 {
        let (bits, add) =
            if self.low == 0 { (&mut self.high, 128) } else { (&mut self.low, 0) };
        let ret = bits.trailing_zeros() as u8;
        *bits &= !(1 << (ret + 1));
        ret + add
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
    fn small() {
        let mut vals = [1u64, 0, 22, 5, 36, 2, 1111, 1112, 44];
        sort(&mut vals, |v| *v);
        println!("{:?}", vals);
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

    // #[test]
    // fn med_random() {
    //     let mut vals: Vec<u64> = (0..1_000_000).map(|_| random()).collect();
    //     sort(&mut vals, |v| *v);
    //     vals.windows(2).for_each(|w| assert!(w[0] <= w[1], "{} <= {}", w[0], w[1]));
    //     assert!(vals.iter().is_sorted());
    // }
}
