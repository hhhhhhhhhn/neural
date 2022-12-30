use crate::Layer;
use rand::prelude::*;

pub struct FullyConnected {
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,
    pub input_size: usize,
    pub output_size: usize,
}

impl Layer for FullyConnected {
    fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = Vec::new();
        for j in 0..self.output_size {
            let mut activation = self.biases[j];
            for i in 0..self.input_size {
                activation += self.weights[j][i]*input[i];
            }
            output.push(activation);
        }
        return output
    }

    fn backward(&self, _input: &Vec<f32>, output_error: &Vec<f32>) -> Vec<f32> {
        let mut input_error: Vec<f32> = Vec::new();
        for i in 0..self.input_size {
            let mut error = 0.;
            for j in 0..self.output_size {
                error += self.weights[j][i]*output_error[j];
            }
            input_error.push(error);
        }
        return input_error
    }

    fn update_params(
        &mut self,
        input: &Vec<f32>,
        output_error: &Vec<f32>,
        learning_rate: f32) {
            for j in 0..self.output_size {
                for i in 0..self.input_size {
                    self.weights[j][i] -= learning_rate*input[i]*output_error[j];
                }
            }
            for j in 0..self.output_size {
                self.biases[j] -= learning_rate*output_error[j];
            }
    }
}

fn random_vec(size: usize) -> Vec<f32> {
    let dist = rand::distributions::Uniform::new(-1., 1.);
    return (0..size).map(|_| (rand::thread_rng()).sample(dist)).collect()
}

impl FullyConnected {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        return FullyConnected{
            input_size,
            output_size,
            biases: random_vec(output_size),
            weights: (0..output_size).map(|_| random_vec(input_size)).collect(),
        };
    }
}
