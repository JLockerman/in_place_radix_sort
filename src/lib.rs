
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

    // macro_rules! sorting_network {
    //     ($array:expr; $(
    //         [$([$i:expr, $j:expr]),*]
    //     )*) => ({
    //         let mut swap_if_needed = |i, j| if &$array[i] > &$array[j] {
    //             $array.swap(i, j);
    //         };
    //         $($(
    //             swap_if_needed($i, $j);
    //         )*)*
    //     });
    // }

    if level + 1 < num_levels {
        start[0] = 0;
        start[1..].copy_from_slice(&ends[..255]);
        for i in 0..start.len() {
            if start[i] == ends[i] { continue }
            if start[i] + 1 == ends[i] { continue }
            //TODO cmove, e.g.:
            // let mut swap_if_needed = |i, j| {
            //     let n = min(array[i], array[j]);
            //     let x = max(array[i], array[j]);
            //     array[i] = n;
            //     array[j] = x;
            // };
            // is noticeably faster, is there way to force it?
            let array = &mut array[start[i]..ends[i]];
            match array.len() {
                2 => if &array[0] > &array[1] {
                    array.swap(0, 1);
                },
                l if l <= 25 => array.sort(),
                // 3 => sorting_network!(array; [[0,1], [0,2]] [[1,2]]),
                // 4 => sorting_network!(array; [[0,1], [2,3]] [[0,2], [1,3]] [[1,2]]),
                // 5 => sorting_network!(array;
                //     [[0,1], [2,3]] [[1,3], [2,4]] [[1,4], [0,2]] [[1,2], [3,4]] [[2,3]]),
                //sorting networks based on http://pages.ripco.net/~jgamble/nw.html
                // 6 => sorting_network!(array;
                //     [[1,2],[4,5]]
                //     [[0,2],[3,5]]
                //     [[0,1],[3,4],[2,5]]
                //     [[0,3],[1,4]]
                //     [[2,4],[1,3]]
                //     [[2,3]]
                // ),
                // 7 => sorting_network!(array;
                //     [[1,2],[3,4],[5,6]]
                //     [[0,2],[3,5],[4,6]]
                //     [[0,1],[4,5],[2,6]]
                //     [[0,4],[1,5]]
                //     [[0,3],[2,5]]
                //     [[1,3],[2,4]]
                //     [[2,3]]
                // ),
                // 8 => sorting_network!(array;
                //     [[0,1],[2,3],[4,5],[6,7]]
                //     [[0,2],[1,3],[4,6],[5,7]]
                //     [[1,2],[5,6],[0,4],[3,7]]
                //     [[1,5],[2,6]]
                //     [[1,4],[3,6]]
                //     [[2,4],[3,5]]
                //     [[3,4]]
                // ),
                // 9 => sorting_network!(array;
                //     [[0,1],[3,4],[6,7]]
                //     [[1,2],[4,5],[7,8]]
                //     [[0,1],[3,4],[6,7],[2,5]]
                //     [[0,3],[1,4],[5,8]]
                //     [[3,6],[4,7],[2,5]]
                //     [[0,3],[1,4],[5,7],[2,6]]
                //     [[1,3],[4,6]]
                //     [[2,4],[5,6]]
                //     [[2,3]]
                // ),
                // 10 => sorting_network!(array;
                //     [[4,9],[3,8],[2,7],[1,6],[0,5]]
                //     [[1,4],[6,9],[0,3],[5,8]]
                //     [[0,2],[3,6],[7,9]]
                //     [[0,1],[2,4],[5,7],[8,9]]
                //     [[1,2],[4,6],[7,8],[3,5]]
                //     [[2,5],[6,8],[1,3],[4,7]]
                //     [[2,3],[6,7]]
                //     [[3,4],[5,6]]
                //     [[4,5]]
                // ),
                // 11 => sorting_network!(array;
                //     [[0,1],[2,3],[4,5],[6,7],[8,9]]
                //     [[1,3],[5,7],[0,2],[4,6],[8,10]]
                //     [[1,2],[5,6],[9,10],[0,4],[3,7]]
                //     [[1,5],[6,10],[4,8]]
                //     [[5,9],[2,6],[0,4],[3,8]]
                //     [[1,5],[6,10],[2,3],[8,9]]
                //     [[1,4],[7,10],[3,5],[6,8]]
                //     [[2,4],[7,9],[5,6]]
                //     [[3,4],[7,8]]
                // ),
                // 12 => sorting_network!(array;
                //     [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]]
                //     [[1,3],[5,7],[9,11],[0,2],[4,6],[8,10]]
                //     [[1,2],[5,6],[9,10],[0,4],[7,11]]
                //     [[1,5],[6,10],[3,7],[4,8]]
                //     [[5,9],[2,6],[0,4],[7,11],[3,8]]
                //     [[1,5],[6,10],[2,3],[8,9]]
                //     [[1,4],[7,10],[3,5],[6,8]]
                //     [[2,4],[7,9],[5,6]]
                //     [[3,4],[7,8]]
                // ),
                // 13 => sorting_network!(array;
                //     [[1,7],[9,11],[3,4],[5,8],[0,12],[2,6]]
                //     [[0,1],[2,3],[4,6],[8,11],[7,12],[5,9]]
                //     [[0,2],[3,7],[10,11],[1,4],[6,12]]
                //     [[7,8],[11,12],[4,9],[6,10]]
                //     [[3,4],[5,6],[8,9],[10,11],[1,7]]
                //     [[2,6],[9,11],[1,3],[4,7],[8,10],[0,5]]
                //     [[2,5],[6,8],[9,10]]
                //     [[1,2],[3,5],[7,8],[4,6]]
                //     [[2,3],[4,5],[6,7],[8,9]]
                //     [[3,4],[5,6]]
                // ),
                // 14 => sorting_network!(array;
                //     [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13]]
                //     [[0,2],[4,6],[8,10],[1,3],[5,7],[9,11]]
                //     [[0,4],[8,12],[1,5],[9,13],[2,6],[3,7]]
                //     [[0,8],[1,9],[2,10],[3,11],[4,12],[5,13]]
                //     [[5,10],[6,9],[3,12],[7,11],[1,2],[4,8]]
                //     [[1,4],[7,13],[2,8],[5,6],[9,10]]
                //     [[2,4],[11,13],[3,8],[7,12]]
                //     [[6,8],[10,12],[3,5],[7,9]]
                //     [[3,4],[5,6],[7,8],[9,10],[11,12]]
                //     [[6,7],[8,9]]
                // ),
                // 15 => sorting_network!(array;
                //     [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13]]
                //     [[0,2],[4,6],[8,10],[12,14],[1,3],[5,7],[9,11]]
                //     [[0,4],[8,12],[1,5],[9,13],[2,6],[10,14],[3,7]]
                //     [[0,8],[1,9],[2,10],[3,11],[4,12],[5,13],[6,14]]
                //     [[5,10],[6,9],[3,12],[13,14],[7,11],[1,2],[4,8]]
                //     [[1,4],[7,13],[2,8],[11,14],[5,6],[9,10]]
                //     [[2,4],[11,13],[3,8],[7,12]]
                //     [[6,8],[10,12],[3,5],[7,9]]
                //     [[3,4],[5,6],[7,8],[9,10],[11,12]]
                //     [[6,7],[8,9]]
                // ),
                // 16 => sorting_network!(array;
                //     [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]]
                //     [[0,2],[4,6],[8,10],[12,14],[1,3],[5,7],[9,11],[13,15]]
                //     [[0,4],[8,12],[1,5],[9,13],[2,6],[10,14],[3,7],[11,15]]
                //     [[0,8],[1,9],[2,10],[3,11],[4,12],[5,13],[6,14],[7,15]]
                //     [[5,10],[6,9],[3,12],[13,14],[7,11],[1,2],[4,8]]
                //     [[1,4],[7,13],[2,8],[11,14],[5,6],[9,10]]
                //     [[2,4],[11,13],[3,8],[7,12]]
                //     [[6,8],[10,12],[3,5],[7,9]]
                //     [[3,4],[5,6],[7,8],[9,10],[11,12]]
                //     [[6,7],[8,9]]
                // ),
                //TODO these aren't great, outline?
                // 17 => sorting_network!(array;
                //     [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[15,16]]
                //     [[0,2],[1,3],[4,6],[5,7],[8,10],[9,11],[14,16]]
                //     [[1,2],[5,6],[0,4],[3,7],[9,10],[14,15],[13,16]]
                //     [[1,5],[2,6],[12,15],[11,16]]
                //     [[1,4],[3,6],[12,14],[13,15],[7,16]]
                //     [[2,4],[3,5],[13,14],[10,15]]
                //     [[3,4],[8,13],[9,14],[11,15]]
                //     [[8,12],[9,13],[11,14],[6,15]]
                //     [[9,12],[10,13],[5,14],[7,15]]
                //     [[10,12],[11,13],[0,9],[7,14]]
                //     [[11,12],[0,8],[1,10],[4,13]]
                //     [[1,9],[2,11],[3,12],[5,13]]
                //     [[1,8],[3,11],[2,9],[6,13]]
                //     [[2,8],[3,10],[7,13],[6,11]]
                //     [[3,9],[5,10],[7,12]]
                //     [[3,8],[4,9],[7,11]]
                //     [[4,8],[5,9],[7,10]]
                //     [[5,8],[6,9]]
                //     [[6,8],[7,9]]
                //     [[7,8]]
                // ),
                // 18 => sorting_network!(array;
                //     [[0,1],[2,3],[4,5],[7,8],[9,10],[11,12],[13,14],[16,17]]
                //     [[0,2],[1,3],[6,8],[9,11],[10,12],[15,17]]
                //     [[1,2],[6,7],[5,8],[10,11],[15,16],[14,17]]
                //     [[4,7],[3,8],[13,16],[12,17]]
                //     [[4,6],[5,7],[13,15],[14,16],[8,17]]
                //     [[5,6],[2,7],[14,15],[11,16]]
                //     [[0,5],[1,6],[3,7],[9,14],[10,15],[12,16]]
                //     [[0,4],[1,5],[3,6],[9,13],[10,14],[12,15],[7,16]]
                //     [[1,4],[2,5],[10,13],[11,14],[0,9],[6,15],[8,16]]
                //     [[2,4],[3,5],[11,13],[12,14],[1,10],[7,15]]
                //     [[3,4],[12,13],[1,9],[2,11],[5,14],[8,15]]
                //     [[3,12],[2,9],[4,13],[7,14]]
                //     [[3,11],[5,13],[8,14]]
                //     [[3,10],[6,13]]
                //     [[3,9],[7,13],[5,10],[6,11]]
                //     [[8,13],[4,9],[7,12]]
                //     [[5,9],[8,12],[7,11]]
                //     [[8,11],[6,9],[7,10]]
                //     [[8,10],[7,9]]
                //     [[8,9]]
                // ),
                // 19 => sorting_network!(array;
                //     [[0,1],[2,3],[4,5],[7,8],[9,10],[12,13],[14,15],[17,18]]
                //     [[0,2],[1,3],[6,8],[11,13],[16,18]]
                //     [[1,2],[6,7],[5,8],[11,12],[10,13],[16,17],[15,18]]
                //     [[4,7],[3,8],[9,12],[14,17],[13,18]]
                //     [[4,6],[5,7],[9,11],[10,12],[14,16],[15,17],[8,18]]
                //     [[5,6],[2,7],[10,11],[15,16],[9,14],[12,17]]
                //     [[0,5],[1,6],[3,7],[10,15],[11,16],[13,17]]
                //     [[0,4],[1,5],[3,6],[10,14],[12,16],[7,17]]
                //     [[1,4],[2,5],[13,16],[11,14],[12,15],[0,10],[8,17]]
                //     [[2,4],[3,5],[13,15],[12,14],[0,9],[1,11],[6,16]]
                //     [[3,4],[13,14],[1,10],[2,12],[5,15],[7,16]]
                //     [[1,9],[3,13],[2,10],[4,14],[8,16],[7,15]]
                //     [[3,12],[2,9],[5,14],[8,15]]
                //     [[3,11],[6,14]]
                //     [[3,10],[7,14],[6,11]]
                //     [[3,9],[8,14],[5,10],[7,12]]
                //     [[4,9],[8,13],[7,11]]
                //     [[5,9],[8,12],[7,10]]
                //     [[8,11],[6,9]]
                //     [[8,10],[7,9]]
                //     [[8,9]]
                // ),
                // 20 => sorting_network!(array;
                //     [[0,1],[3,4],[5,6],[8,9],[10,11],[13,14],[15,16],[18,19]]
                //     [[2,4],[7,9],[12,14],[17,19]]
                //     [[2,3],[1,4],[7,8],[6,9],[12,13],[11,14],[17,18],[16,19]]
                //     [[0,3],[5,8],[4,9],[10,13],[15,18],[14,19]]
                //     [[0,2],[1,3],[5,7],[6,8],[10,12],[11,13],[15,17],[16,18],[9,19]]
                //     [[1,2],[6,7],[0,5],[3,8],[11,12],[16,17],[10,15],[13,18]]
                //     [[1,6],[2,7],[4,8],[11,16],[12,17],[14,18],[0,10]]
                //     [[1,5],[3,7],[11,15],[13,17],[8,18]]
                //     [[4,7],[2,5],[3,6],[14,17],[12,15],[13,16],[1,11],[9,18]]
                //     [[4,6],[3,5],[14,16],[13,15],[1,10],[2,12],[7,17]]
                //     [[4,5],[14,15],[3,13],[2,10],[6,16],[8,17]]
                //     [[4,14],[3,12],[5,15],[9,17],[8,16]]
                //     [[4,13],[3,11],[6,15],[9,16]]
                //     [[4,12],[3,10],[7,15]]
                //     [[4,11],[8,15],[7,12]]
                //     [[4,10],[9,15],[6,11],[8,13]]
                //     [[5,10],[9,14],[8,12]]
                //     [[6,10],[9,13],[8,11]]
                //     [[9,12],[7,10]]
                //     [[9,11],[8,10]]
                //     [[9,10]]
                // ),
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
