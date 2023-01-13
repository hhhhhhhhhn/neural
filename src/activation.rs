use crate::Layer;

pub struct Activation {
    pub function: fn(f32) -> f32,
    pub derivative: fn(f32) -> f32
}

impl Layer for Activation {
    fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        return input.iter().map(|x| (self.function)(*x)).collect()
    }

    fn backward(&self, input: &Vec<f32>, output_error: &Vec<f32>) -> Vec<f32> {
        return input.iter()
            .zip(output_error)
            .map(|(x, e)| (self.derivative)(*x)*(*e)).collect()
    }

    fn update_params(&mut self, _: &Vec<f32>, _: &Vec<f32>, _: f32) {
    }
}

impl Activation {
    pub fn new(function: fn(f32) -> f32, derivative: fn(f32) -> f32) -> Self {
        return Activation {function, derivative}
    }
}

pub fn tanh(x: f32) -> f32 {
    return x.tanh()
}

pub fn dtanh(x: f32) -> f32 {
    return 1. - x.tanh().powi(2)
}

pub fn relu(x: f32) -> f32 {
    return x.max(0.)
}

pub fn drelu(x: f32) -> f32 {
    if x > 0. {
        return 1.
    }
    return 0.
}
