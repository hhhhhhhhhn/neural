use crate::*;

pub const MSE: Error = Error{
    function: |input: &Vec<f32>, expected: &Vec<f32>| -> f32 {
        return input.iter()
            .zip(expected)
            .map(|(i, e)| (e - i)*(e - i))
            .sum::<f32>()
            / (input.len() as f32)
            / 2.
    },
    derivative: |input: &Vec<f32>, expected: &Vec<f32>| -> Vec<f32> {
        return input.iter()
            .zip(expected)
            .map(|(i, e)| i - e)
            .collect()
    },
};
