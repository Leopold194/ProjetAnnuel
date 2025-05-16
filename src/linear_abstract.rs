use crate::utils;
use crate::labels;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyString};
use rand::Rng;

pub trait LinearModelAbstract {
    fn weights(&self) -> Vec<f64>;
    fn loss(&self) -> Vec<f64>;

    fn get_x(&self) -> &Vec<Vec<f64>>;
    fn get_y(&self) -> &labels::LabelsEnum;
    fn get_model_type(&self) -> &str;
    fn set_model_type(&mut self, model_type: String);

    fn get_label_map_str(&self) -> Option<(String, String)>;
    fn get_label_map_float(&self) -> Option<(f64, f64)>;
    fn set_label_map_str(&mut self, map: Option<(String, String)>);
    fn set_label_map_float(&mut self, map: Option<(f64, f64)>);

    fn set_weights(&mut self, w: Vec<f64>);
    fn set_loss(&mut self, l: Vec<f64>);
    
    fn model(&self, py: Python<'_>, x: Vec<Vec<f64>>) -> Vec<f64> {
        let mut weights_without_bias = self.weights();
        weights_without_bias.pop();

        // utils::py_print(py, "test2");

        let mut z = utils::matvecmul(&x, &weights_without_bias);

        // utils::py_print(py, "test3");

        let bias = self.weights().last().cloned().unwrap_or(0.0);
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

    fn calc_log_loss(&self, y: Vec<f64>, sig_val: Vec<f64>) -> f64 {
        let mut loss = 0.0;
        for i in 0..sig_val.len() {
            loss += y[i] * sig_val[i].ln() + (1.0 - y[i]) * (1.0 - sig_val[i]).ln();
        }
        -loss / sig_val.len() as f64
    }

    fn calc_gradients(&self, sig_val: Vec<f64>, x: Vec<Vec<f64>>, y: Vec<f64>) -> Vec<f64> {
        let n = y.len();
        let m = x[0].len();
        let mut weights_grad = vec![0.0; m];
        let mut bias_grad = 0.0;

        for i in 0..n {
            let diff = sig_val[i] - y[i];
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

    fn update_weights(&mut self, grad: Vec<f64>, learning_rate: f64) {
        let mut w = self.weights();
        for i in 0..w.len() {
            w[i] -= grad[i] * learning_rate;
        }
        self.set_weights(w);
    }

    fn train_classification(&mut self, py: Python<'_>, epochs: usize, learning_rate: f64, algo: &str) {
        self.set_model_type("classification".to_string());

        let mut label_map_str = None;
        let mut label_map_float = None;

        let y_labels = self.get_y();
        let labels_norm = match y_labels {
            labels::LabelsEnum::Str(labels) => {
                let first = labels[0].clone();
                let second = labels.iter().find(|l| **l != first).unwrap_or(&first).clone();
                label_map_str = Some((first.clone(), second.clone()));
                labels.iter().map(|l| if *l == first { 1.0 } else { 0.0 }).collect()
            }
            labels::LabelsEnum::Float(values) => {
                let mut uniq = values.clone();
                uniq.sort_by(|a, b| a.partial_cmp(b).unwrap());
                uniq.dedup();
                if uniq.len() == 2 {
                    label_map_float = Some((uniq[1], uniq[0]));
                    values.iter().map(|v| if *v == uniq[1] { 1.0 } else { 0.0 }).collect()
                } else {
                    values.clone()
                }
            }
        };

        self.set_label_map_str(label_map_str.clone());
        self.set_label_map_float(label_map_float.clone());

        let mut losses = Vec::with_capacity(epochs);
        
        if algo == "gradient-descent" {
            for _ in 0..epochs {
                let a = self.model(py, self.get_x().clone());
                let l = self.calc_log_loss(labels_norm.clone(), a.clone());
                let grad = self.calc_gradients(a.clone(), self.get_x().clone(), labels_norm.clone());
                self.update_weights(grad, learning_rate);
                losses.push(l);
            }

        } else if algo == "rosenblatt" {
            for _ in 0..epochs {
                let mut rng = rand::rng();
                let i = rng.random_range(0..self.get_x().len());
                let random_x = self.get_x()[i].clone();

                let prediction = self.predict_classification(py, random_x.clone());
                let error = labels_norm[i] - prediction;

                // Mise à jour des poids via getters/setters
                let mut w = self.weights();
                for j in 0..w.len() - 1 {
                    w[j] += learning_rate * error * random_x[j];
                }
                let last_index = w.len() - 1;
                w[last_index] += learning_rate * error;
                self.set_weights(w);

                // Calcul de la loss (ex : log-loss)
                let a = self.model(py, self.get_x().clone());
                let l = self.calc_log_loss(labels_norm.clone(), a);
                losses.push(l);
            }
        } else {
            panic!("Cet algorithme n'est pas pris en charge.")
        }
        self.set_loss(losses);
    }

    fn train_regression(&mut self) {
        self.set_model_type("regression".to_string());

        let mut x = self.get_x().clone();
        for row in &mut x {
            row.push(1.0);
        }

        let y = match self.get_y() {
            labels::LabelsEnum::Str(_) => panic!("Regression incompatible avec labels String"),
            labels::LabelsEnum::Float(v) => v.clone(),
        };

        let xt = utils::transpose(&x);
        let xtx = utils::matmatmul(&xt, &x);
        let xtx_inv = utils::inverse(xtx, "Moore-Penrose");
        let xtx_inv_xt = utils::matmatmul(&xtx_inv, &xt);
        let weights = utils::matvecmul(&xtx_inv_xt, &y);
        self.set_weights(weights);
    }

    fn predict_proba(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<PyObject> {
        let proba = self.model(py, vec![x])[0];
        let result = PyDict::new(py);

        if let Some((pos, neg)) = self.get_label_map_str() {
            result.set_item(pos, proba)?;
            result.set_item(neg, 1.0 - proba)?;
        } else if let Some((pos, neg)) = self.get_label_map_float() {
            result.set_item(pos.to_string(), proba)?;
            result.set_item(neg.to_string(), 1.0 - proba)?;
        }

        Ok(result.into())
    }

    fn predict_classification(&self, py: Python<'_>, x: Vec<f64>) -> f64 {
        if self.model(py, vec![x])[0] >= 0.5 {
            1.0
        } else {
            0.0
        }
    }

    fn predict_regression(&self, x: Vec<f64>) -> f64 {
        let mut x_bias = x;
        x_bias.push(1.0);
        utils::vecvecmul(&self.weights(), &x_bias).iter().sum()
    }

    fn predict(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<PyObject> {
        match self.get_model_type() {
            "regression" => Ok(PyFloat::new(py, self.predict_regression(x)).into()),
            _ => {
                let prediction = self.predict_classification(py, x);
                if let Some((pos, neg)) = self.get_label_map_str() {
                    let label = if prediction > 0.0 { pos } else { neg };
                    Ok(PyString::new(py, &label).into())
                } else if let Some((pos, neg)) = self.get_label_map_float() {
                    let label = if prediction > 0.0 { pos } else { neg };
                    Ok(PyFloat::new(py, label).into())
                } else {
                    Ok(PyFloat::new(py, prediction).into())
                }
            }
        }
    }
}
