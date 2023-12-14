pub fn u8_slice_to_i16_slice(u8_slice: &[u8]) -> &[i16] {
    assert!(u8_slice.len() % 2 == 0);
    let i16_ptr = u8_slice.as_ptr() as *const i16;
    let i16_len = u8_slice.len() / 2;
    unsafe { std::slice::from_raw_parts(i16_ptr, i16_len) }
}
