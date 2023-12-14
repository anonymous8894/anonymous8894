// modified from github.com/LukasKalbertodt/litrs/blob/main/src/escape.rs
// and github.com/LukasKalbertodt/litrs/blob/main/src/parse.rs

use std::{
    error::Error,
    fmt::{Debug, Display},
};

pub fn unescape_string(input: &str) -> Result<String, StrEscapeError> {
    let mut i = 0;
    let mut end_last_escape = 0;
    let mut value = String::new();
    while i < input.len() {
        match input.as_bytes()[i] {
            b'\\' => {
                let (c, len) = unescape(&input[i..input.len() - 1])?;
                value.push_str(&input[end_last_escape..i]);
                value.push(c.into());
                i += len;
                end_last_escape = i;
            }
            b'\r' => {
                if input.as_bytes().get(i + 1) == Some(&b'\n') {
                    value.push_str(&input[end_last_escape..i]);
                    value.push('\n');
                    i += 2;
                    end_last_escape = i;
                } else {
                    return Err(StrEscapeError);
                }
            }
            _ => i += 1,
        }
    }
    value.push_str(&input[end_last_escape..]);
    Ok(value)
}

pub fn unescape(input: &str) -> Result<(char, usize), StrEscapeError> {
    let first = input.as_bytes().get(1).ok_or(StrEscapeError)?;
    let out = match first {
        // Quote escapes
        b'\'' => ('\'', 2),
        b'"' => ('"', 2),

        // Ascii escapes
        b'n' => ('\n', 2),
        b'r' => ('\r', 2),
        b't' => ('\t', 2),
        b'\\' => ('\\', 2),
        b'0' => ('\0', 2),
        b'x' => {
            let hex_string = input.get(2..4).ok_or(StrEscapeError)?.as_bytes();
            let first = hex_digit_value(hex_string[0]).ok_or(StrEscapeError)?;
            let second = hex_digit_value(hex_string[1]).ok_or(StrEscapeError)?;
            let value = second + 16 * first;

            (value.into(), 4)
        }

        // Unicode escape
        b'u' => {
            if input.as_bytes().get(2) != Some(&b'{') {
                return Err(StrEscapeError);
            }

            let closing_pos = input
                .bytes()
                .position(|b| b == b'}')
                .ok_or(StrEscapeError)?;

            let inner = &input[3..closing_pos];
            if inner.as_bytes().first() == Some(&b'_') {
                return Err(StrEscapeError);
            }

            let mut v: u32 = 0;
            let mut digit_count = 0;
            for b in inner.bytes() {
                if b == b'_' {
                    continue;
                }

                let digit = hex_digit_value(b).ok_or(StrEscapeError)?;

                if digit_count == 6 {
                    return Err(StrEscapeError);
                }
                digit_count += 1;
                v = 16 * v + digit as u32;
            }

            let c = std::char::from_u32(v).ok_or(StrEscapeError)?;

            (c, closing_pos + 1)
        }

        _ => return Err(StrEscapeError),
    };

    Ok(out)
}

pub fn hex_digit_value(digit: u8) -> Option<u8> {
    match digit {
        b'0'..=b'9' => Some(digit - b'0'),
        b'a'..=b'f' => Some(digit - b'a' + 10),
        b'A'..=b'F' => Some(digit - b'A' + 10),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrEscapeError;

impl Display for StrEscapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl Error for StrEscapeError {}
