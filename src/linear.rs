use crate::utils;
use crate::labels;

use pyo3::prelude::*;
use pyo3::types::{ PyString, PyFloat, PyDict };
use rand::Rng;

#[pyclass]
pub struct LinearModel {
    x: Vec<Vec<f64>>,
    y: labels::LabelsEnum,
    weights: Vec<f64>,
    label_map_str: Option<(String, String)>,
    label_map_float: Option<(f64, f64)>,
    model_type: String,
    loss: Vec<f64>,
}

#[pymethods]
impl LinearModel {
    
    /// Creates a new LinearModel instance.
    /// The `x` parameter is a 2D vector representing the input features,
    /// and the `y` parameter can be a vector of strings or floats representing the labels.
    /// The function normalizes the labels to -1 and 1 for classification tasks.
    
    #[new]
    fn new(x: Vec<Vec<f64>>, y: labels::LabelsEnum) -> Self {
        let mut rng = rand::rng();
        let mut weights = (0..x[0].len()).map(|_| rng.random_range(-1.0..1.0)).collect::<Vec<f64>>();
        weights.push(0.0); // biais

        LinearModel {
            x,
            y,
            weights,
            label_map_str: None,
            label_map_float: None,
            model_type: "".to_string(),
            loss: vec![],
        }
    }

    #[getter]
    fn weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    #[getter]
    fn loss(&self) -> Vec<f64> {
        self.loss.clone()
    }

    fn calc_log_loss(&mut self, y: Vec<f64>, sig_val: Vec<f64>) -> f64 {
        let mut loss = 0.0;

        for i in 0..sig_val.len() {
            loss += y[i] * sig_val[i].ln() + (1.0 - y[i]) * (1.0 - sig_val[i]).ln();
        }

        (- 1.0 / sig_val.len() as f64) * loss
    }

    fn calc_gradients(&mut self, sig_val: Vec<f64>, x: Vec<Vec<f64>>, y: Vec<f64>) -> Vec<f64> {
        let n = y.len();
        let m = x[0].len();

        let mut weights_grad = vec![0.0; x[0].len()];
        let mut bias_grad = 0.0;

        let mut diff;

        for i in 0..n {
            diff = sig_val[i] - y[i];

            for j in 0..m {
                weights_grad[j] += x[i][j] * diff;
            }

            bias_grad += diff;
        }
        
        for j in 0..m {
            weights_grad[j] /= n as f64;
        }
        bias_grad /= n as f64;

        weights_grad.push(bias_grad);

        weights_grad
    }

    fn update_weights(&mut self, weights_grad: Vec<f64>, learning_rate: f64){
        for i in 0..self.weights.len() {
            self.weights[i] -= weights_grad[i] * learning_rate;
        }
    }

    /// Trains the model for a specified number of epochs using the given learning rate.
    /// The training process involves adjusting the weights based on the input features and labels.
    fn train_classification(&mut self, epochs: usize, learning_rate: f64) {
        
        self.model_type = "classification".to_string();
        
        let mut label_map_str = None;
        let mut label_map_float = None;

        let labels_norm = match &self.y {
            labels::LabelsEnum::Str(labels) => {
                let first_label = labels[0].clone();
                let second_label = labels
                    .iter()
                    .find(|l| **l != first_label)
                    .unwrap_or(&first_label)
                    .clone();
                label_map_str = Some((first_label.clone(), second_label.clone()));

                labels
                    .iter()
                    .map(|label| if *label == first_label { 1.0 } else { 0.0 })
                    .collect()
            }
            labels::LabelsEnum::Float(values) => {
                let mut uniq: Vec<f64> = values.clone();
                uniq.sort_by(|a, b| a.partial_cmp(b).unwrap());
                uniq.dedup();

                if uniq.len() == 2 {
                    label_map_float = Some((uniq[1], uniq[0]));
                    values
                        .iter()
                        .map(|v| if *v == uniq[1] { 1.0 } else { 0.0 })
                        .collect()
                } else {
                    values.clone()
                }
            }
        };

        self.label_map_str = label_map_str;
        self.label_map_float = label_map_float;

        let mut loss = Vec::with_capacity(labels_norm.len());
        let mut a;
        let mut l;
        let mut weights_grad;

        for _ in 0..epochs {
            a = self.model(self.x.clone());
            l = self.calc_log_loss(labels_norm.clone(), a.clone());
            loss.push(l);
            weights_grad = self.calc_gradients(a.clone(), self.x.clone(), labels_norm.clone());
            self.update_weights(weights_grad, learning_rate);
        }

        self.loss = loss;
    }

    /// Trains the model using linear regression.
    /// The training process involves calculating the weights using the normal equation.
    fn train_regression(&mut self) {

        self.model_type = "regression".to_string();

        let mut x = self.x.clone();

        // Ajout du biais
        for i in 0..x.len(){
            x[i].push(1.0);
        }
        let y = self.y.clone();
        let y: Vec<f64> = match y {
            labels::LabelsEnum::Str(_labels) => {
                panic!("La regression ne fonctionne pas avec des labels de type String")
            }
            labels::LabelsEnum::Float(values) => values,
        };

        let xt: Vec<Vec<f64>> = utils::transpose(&x);

        let xtx: Vec<Vec<f64>> = utils::matmatmul(&xt, &x);
        let call: &str = "Moore-Penrose";
        let xtx_inv: Vec<Vec<f64>> = utils::inverse(xtx.clone(), &call);
        let xtx_inv_xt: Vec<Vec<f64>> = utils::matmatmul(&xtx_inv, &xt);
        self.weights = utils::matvecmul(&xtx_inv_xt, &y);
    }

    /// Predicts the output for a given input vector `x` using the trained model.
    /// The function calculates the weighted sum of the input features and applies the activation function.
    fn model(&self, x: Vec<Vec<f64>>) -> Vec<f64> {
        let mut weights_without_bias = self.weights.clone();
        weights_without_bias.pop();
        let mut z = utils::matvecmul(&x, &weights_without_bias);

        let bias = self.weights[self.weights.len() - 1];
        for val in z.iter_mut() {
            *val += bias;
        }

        // activation function (sigmoid)
        let mut result = Vec::with_capacity(z.len());

        let eps = 1e-15;
        for i in 0..z.len() {
            let neg_x = -z[i];
            let sigmoid = 1.0 / (1.0 + neg_x.exp());
            // car si valeur trèèès grande de z, exp(-z) = 1 et si trèèès petit = 0
            let clamped = sigmoid.clamp(eps, 1.0 - eps);
            result.push(clamped);
        }

        result
    }

    pub fn predict_proba(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<PyObject> {
        let proba = self.model(vec![x])[0];
        let result = PyDict::new(py);

        if let Some((pos, neg)) = &self.label_map_str {
            result.set_item(pos, proba)?;
            result.set_item(neg, 1.0 - proba)?;
        } else {
            let (pos, neg) = self
                .label_map_float
                .as_ref()
                .expect("label_map_float should never be None if label_map_str is None");
            result.set_item(pos.to_string(), proba)?;
            result.set_item(neg.to_string(), 1.0 - proba)?;
        }

        Ok(result.into())
    }

    fn predict_classification(&self, x: Vec<f64>) -> f64 {
        let val = self.model(vec![x])[0];
        match val {
            tmp if tmp >= 0.5 => {
                return 1.0
            }
            _ => {
                return 0.0
            }
        }
    }

    fn predict_regression(&self, x: Vec<f64>) -> f64 {
        let mut x_with_bias: Vec<f64> = x.clone();
        x_with_bias.push(1.0);
        let y = utils::vecvecmul(&self.weights,&x_with_bias);
        let mut value:f64=0.0;

        for i in 0..y.len(){
            value+=y[i];
        }

        value
    }
    
    pub fn predict(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<PyObject> {
        match self.model_type.as_str() {
            "regression" => {
                let value = self.predict_regression(x);
                Ok(PyFloat::new(py, value).into())
            }
            _ => {
                let prediction = self.predict_classification(x);

                if let Some((pos, neg)) = &self.label_map_str {
                    let label_str = if prediction > 0.0 { pos } else { neg };
                    Ok(PyString::new(py, label_str).into())
                } else {
                    let (pos, neg) = self
                        .label_map_float
                        .as_ref()
                        .expect("label_map_float should never be None if label_map_str is None");
                    let label_float = if prediction > 0.0 { pos } else { neg };
                    Ok(PyFloat::new(py, *label_float).into())
                }
                
            }
        }
    }
}