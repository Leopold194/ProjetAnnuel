use pyo3::prelude::*;
use rand::Rng;

struct LinearModel {
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Vec<f64>,
    label_map: Option<(String, String)>,
}

enum Labels {
    Str(Vec<String>),
    Float(Vec<f64>),
}

impl From<Vec<&str>> for Labels {
    fn from(labels: Vec<&str>) -> Self {
        Labels::Str(labels.into_iter().map(|s| s.to_string()).collect())
    }
}

impl From<Vec<f64>> for Labels {
    fn from(values: Vec<f64>) -> Self {
        Labels::Float(values)
    }
}

impl From<Vec<i32>> for Labels {
    fn from(values: Vec<i32>) -> Self {
        Labels::Float(values.into_iter().map(|v| v as f64).collect())
    }
}

impl LinearModel {
    fn new<T: Into<Labels>>(x: Vec<Vec<f64>>, y: T) -> Self {
        let mut label_map = None;

        let labels_norm = match y.into() {
            Labels::Str(labels) => {
                let first_label = labels[0].clone();
                let second_label = labels.iter().find(|l| **l != first_label)
                    .unwrap_or(&first_label).clone();
                label_map = Some((first_label.clone(), second_label.clone()));

                labels
                    .iter()
                    .map(|label| if *label == first_label { 1.0 } else { -1.0 })
                    .collect()
            }
            Labels::Float(values) => {
                let mut uniq: Vec<f64> = values.clone();
                uniq.sort_by(|a, b| a.partial_cmp(b).unwrap());
                uniq.dedup();

                if uniq.len() == 2 {
                    label_map = Some((uniq[1].to_string(), uniq[0].to_string())); // 1.0 -> uniq[1], -1.0 -> uniq[0]
                    values
                        .iter()
                        .map(|v| if *v == uniq[1] { 1.0 } else { -1.0 })
                        .collect()
                } else {
                    values
                }
            }
        };

        let mut rng = rand::thread_rng();
        let mut weights = (0..x[0].len()).map(|_| rng.gen_range(-1.0..1.0)).collect::<Vec<f64>>();
        weights.push(0.0); // biais

        LinearModel {
            x,
            y: labels_norm,
            weights,
            label_map,
        }
    }

    fn train_classification(&mut self, epochs: usize, learning_rate: f64) {
        for _ in 0..epochs {
            let mut rng = rand::thread_rng();
            let i = rng.gen_range(0..self.x.len());

            let random_x = self.x[i].clone();
            let prediction = self.predict_value(random_x.clone());
            let error = self.y[i] - prediction;

            for j in 0..self.weights.len() - 1 {
                self.weights[j] += learning_rate * error * random_x[j];
            }
            let last_index = self.weights.len() - 1; 
            self.weights[last_index] += learning_rate * error;
        }

        // println!("Weights: {:?}", self.weights);
    }

    fn predict_value(&self, x: Vec<f64>) -> f64 {
        let mut sum = 0.0;
        let mut x_with_bias = x.clone();
        x_with_bias.push(1.0);

        for i in 0..x_with_bias.len() {
            sum += x_with_bias[i] * self.weights[i];
        }

        if sum > 0.0 { 1.0 } else { -1.0 }
    }

    fn predict(&self, x: Vec<f64>) -> String {
        let prediction = self.predict_value(x);

        if let Some((pos, neg)) = &self.label_map {
            if prediction > 0.0 {
                pos.clone()
            } else {
                neg.clone()
            }
        } else {
            prediction.to_string()
        }
    }
}

fn main() {
    let x = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0], vec![1.0, 1.0]];
    // let y = vec!["true", "true", "false", "true"]; // OR
    let y = vec!["false", "false", "false", "true"]; // AND
    // let y = vec![1, 1, 0, 1]; 

    println!("X: {:?}", x);
    println!("Y: {:?}", y);

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