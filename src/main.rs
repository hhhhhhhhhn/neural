pub mod fully_connected;
pub mod activation;
pub mod errors;

use fully_connected::*;
use activation::*;

pub trait Layer {
    fn forward(&self, input: &Vec<f32>) -> Vec<f32>;
    fn backward(&self, input: &Vec<f32>, output_error: &Vec<f32>) -> Vec<f32>;
    fn update_params(
        &mut self, input: &Vec<f32>, output_error: &Vec<f32>, learning_rate: f32);
}

pub struct Error {
    pub function: fn (input: &Vec<f32>, expected: &Vec<f32>) -> f32,
    pub derivative: fn (input: &Vec<f32>, expected: &Vec<f32>) -> Vec<f32>,
}

pub struct Network<'a> {
    pub layers: Vec<&'a mut dyn Layer>,
    pub error: Error,
}

impl<'a> Network<'a> {
    fn new(layers: Vec<&'a mut dyn Layer>, error: Error) -> Self {
        return Network{layers, error}
    }
}

impl Network<'_> {
    pub fn predict(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut next = input.clone();
        for layer in &self.layers {
            next = layer.forward(&next);
        }
        return next.to_vec()
    }

    pub fn learn(&mut self, input: &Vec<f32>, expected: &Vec<f32>, learning_rate: f32) -> (Vec<f32>, f32) {
        let mut layer_inputs = vec![vec![]; self.layers.len()+1];
        layer_inputs[0] = input.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            layer_inputs[i+1] = layer.forward(&layer_inputs[i]);
        }

        let output = layer_inputs.last().unwrap().clone();
        let error = (self.error.function)(&output, &expected);


        let mut layer_errors = vec![vec![]; self.layers.len()+1];
        *layer_errors.last_mut().unwrap() = (self.error.derivative)(&output, &expected);
        
        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            layer_errors[i] = layer.backward(&layer_inputs[i], &layer_errors[i+1]);
            layer.update_params(&layer_inputs[i], &layer_errors[i+1], learning_rate);
        }
        
        return (output.to_vec(), error)
    }

    pub fn train(&mut self, data: &Vec<(Vec<f32>, Vec<f32>)>, iters: usize, learning_rate: f32) {
        for _ in 0..iters {
            let mut error_sum = 0.;
            for (input, expected) in data.iter() {
                let (_, error) = self.learn(input, expected, learning_rate);
                error_sum += error;
            }
            println!("Error: {}", error_sum/(data.len() as f32))
        }
    }
}

fn main() {
    let mut l1 = FullyConnected::new(2, 3);
    let mut l2 = Activation::new(tahh, dtanh);
    let mut l3 = FullyConnected::new(3, 1);
    let mut l4 = Activation::new(tahh, dtanh);

    let mut network = Network::new(vec![&mut l1, &mut l2, &mut l3, &mut l4], errors::MSE);

    let data = vec![
        (vec![0., 0.], vec![0.]),
        (vec![0., 1.], vec![1.]),
        (vec![1., 0.], vec![1.]),
        (vec![1., 1.], vec![0.]),
    ];
    network.train(&data, 1000, 0.1);

    for (input, _) in data.iter() {
        let output = network.predict(input);
        println!("{:?}", output)
    }
}
