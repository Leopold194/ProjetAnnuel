use pyo3::prelude::*;
use rand::Rng;

struct LinearModel {
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Vec<f64>,
}

enum Labels {
    Str(Vec<String>),
    Float(Vec<f64>),
}

// Permet d'utiliser directement Vec<&str> sans intervention de l'utilisateur
impl From<Vec<&str>> for Labels {
    fn from(labels: Vec<&str>) -> Self {
        Labels::Str(labels.into_iter().map(|s| s.to_string()).collect())
    }
}

// Permet d'utiliser directement Vec<f64> sans intervention de l'utilisateur
impl From<Vec<f64>> for Labels {
    fn from(values: Vec<f64>) -> Self {
        Labels::Float(values)
    }
}

impl LinearModel {
    fn new<T: Into<Labels>>(x: Vec<Vec<f64>>, y: T) -> Self {

        let labels_norm = match y.into() {
            Labels::Str(labels) => {
                let first_label = &labels[0];
                labels
                    .iter()
                    .map(|label| {
                        if *label == *first_label {
                            1.0
                        } else {
                            -1.0
                        }
                    })
                    .collect()
            }
            Labels::Float(values) => values,
        };

        let mut weights = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..x[0].len() {
            weights.push(rng.gen_range(-1.0..1.0));
        }
        weights.push(0.0);
        
        LinearModel {
            x: x,
            y: labels_norm,
            weights: weights,
        }
    }

    fn train_classification(&mut self, epochs: usize, learning_rate: f64) {
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

        println!("Weights: {:?}", self.weights);
    }

    // fn train_regression($mut self) {
    //     // for j in 0..self.weights.len() - 1 {
    //     //     self.weights[j] = 
    //     // }
    // }


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
    // let y = vec![1.0, 1.0, -1.0, 1.0];
    println!("x: {:?}", x);
    println!("y: {:?}", y);
    let mut model = LinearModel::new(x, y);
    model.train_classification(1000000, 0.001);

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