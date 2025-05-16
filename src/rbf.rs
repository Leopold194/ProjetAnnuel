use crate::utils;
use crate::labels;
use crate::lloyd::lloyd;
use crate::linear_abstract::LinearModelAbstract;

use pyo3::prelude::*;
use pyo3::types::{ PyString, PyFloat, PyDict };
use rand::Rng;

#[pyclass]
pub struct RBF {
    #[pyo3(get, set)]
    pub weights: Vec<f64>,
    #[pyo3(get, set)]
    pub loss: Vec<f64>,
    #[pyo3(get, set)]
    pub x: Vec<Vec<f64>>,
    #[pyo3(get, set)]
    pub y: labels::LabelsEnum,
    #[pyo3(get, set)]
    pub model_type: String,
    #[pyo3(get, set)]
    pub centers: Vec<Vec<f64>>,
    #[pyo3(get, set)]
    pub gamma: f64,
    pub label_map_str: Option<(String, String)>,
    pub label_map_float: Option<(f64, f64)>,
}

#[pymethods]
impl RBF {
    #[new]
    pub fn new(py: Python<'_>, x: Vec<Vec<f64>>, y: labels::LabelsEnum, gamma: f64, k: i32) -> Self {
        
        let centers = lloyd(x.clone(), k, 2.22e-16);
        
        let mut phi: Vec<Vec<f64>> = Vec::with_capacity(x.len());
        for row in x.clone() {
            phi.push(utils::convert_x_to_phi(row, centers.clone(), gamma))
        }

        let dim = phi[0].len() + 1;
        let weights = (0..dim).map(|_| rand::random::<f64>()).collect();
        
        RBF {
            weights,
            loss: vec![],
            x: phi,
            y,
            model_type: String::new(),
            centers,
            gamma,
            label_map_str: None,
            label_map_float: None,
        }
    }

    pub fn train_classification(&mut self, py: Python<'_>, epochs: usize, learning_rate: f64, algo: &str) {
        <Self as LinearModelAbstract>::train_classification(self, py, epochs, learning_rate, algo)
    }

    pub fn train_regression(&mut self) {
        <Self as LinearModelAbstract>::train_regression(self)
    }

    pub fn predict(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<PyObject> {
        <Self as LinearModelAbstract>::predict(self, py, utils::convert_x_to_phi(x, self.centers.clone(), self.gamma))
    }

    pub fn predict_proba(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<PyObject> {
        <Self as LinearModelAbstract>::predict_proba(self, py, utils::convert_x_to_phi(x, self.centers.clone(), self.gamma))
    }
}

impl LinearModelAbstract for RBF {
    fn weights(&self) -> Vec<f64> { self.weights.clone() }
    fn loss(&self) -> Vec<f64> { self.loss.clone() }
    fn get_x(&self) -> &Vec<Vec<f64>> { &self.x }
    fn get_y(&self) -> &labels::LabelsEnum { &self.y }
    fn get_model_type(&self) -> &str { &self.model_type }
    fn set_model_type(&mut self, model_type: String) { self.model_type = model_type; }
    fn get_label_map_str(&self) -> Option<(String, String)> { self.label_map_str.clone() }
    fn get_label_map_float(&self) -> Option<(f64, f64)> { self.label_map_float.clone() }
    fn set_label_map_str(&mut self, map: Option<(String, String)>) { self.label_map_str = map; }
    fn set_label_map_float(&mut self, map: Option<(f64, f64)>) { self.label_map_float = map; }
    fn set_weights(&mut self, w: Vec<f64>) { self.weights = w; }
    fn set_loss(&mut self, l: Vec<f64>) { self.loss = l; }
}
