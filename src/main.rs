use pyo3::prelude::*;
use rand::Rng;

struct LinearModel {
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Vec<f64>,
}

impl LinearModel {
    fn new(x: Vec<Vec<f64>>, y: Vec<&str>) -> Self {
        let mut labels_norm = Vec::new();
        let first_label = y[0];
        for i in 0..y.len() {
            if y[i] == first_label {
                labels_norm.push(1.0);
            } else {
                labels_norm.push(-1.0);
            }
        }

        let mut weights = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..x[0].len() + 1 {
            weights.push(rng.gen_range(-1.0..1.0));
        }
        
        LinearModel {
            x: x,
            y: labels_norm,
            weights: weights,
        }
    }

    fn train(&mut self, modele_type: &str, epochs: usize, learning_rate: f64) {
        if *modele_type == *"classification" {
            for _ in 0..epochs {
                let mut rng = rand::thread_rng();
                let i = rng.gen_range(0..self.x.len());

                let random_x = self.x[i].clone();
                let prediction = self.predict(random_x.clone());
                let error = self.y[i] - prediction;

                for j in 0..self.weights.len() - 1 {
                    self.weights[j] += learning_rate * error * random_x[j];
                }
                let last_index = self.weights.len() - 1;
                self.weights[last_index] += learning_rate * error * 1.0; // Biais                
            }
        } else if *modele_type == *"regression" {
            // Regression
        }

        println!("Weights: {:?}", self.weights);
    }


    fn predict(&self, x: Vec<f64>) -> f64 {
        let mut sum: f64 = 0.0;

        let mut x_with_bias = x.clone();
        x_with_bias.push(1.0); // Ajout du biais (1.0)

        for i in 0..x_with_bias.len() {
            sum += x_with_bias[i] * self.weights[i];
        }

        if sum > 0.0 {
            1.0
        } else {
            -1.0
        }
    }
}

fn main() {
    let x = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0], vec![1.0, 1.0]];
    let y = vec!["true", "true", "false", "true"];
    let mut model = LinearModel::new(x, y);
    model.train("classification", 1000000, 0.001);

    println!("Prediction: {:?}", model.predict(vec![1.0, 0.0]));
    println!("Prediction: {:?}", model.predict(vec![0.0, 1.0]));
    println!("Prediction: {:?}", model.predict(vec![0.0, 0.0]));
    println!("Prediction: {:?}", model.predict(vec![1.0, 1.0]));
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn projetannuel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}